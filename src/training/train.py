# Standard library
import argparse
import gc
import logging
import sys
import tempfile
# NumPy
import numpy as np
# TensorFlow
import tensorflow as tf
# Structured logging
from pythonjsonlogger.json import JsonFormatter
# Config
import config
# DB
from src.db import jobs
# Data
from src.data.loader import load_manifest
from src.data.splitter import split_dataset
from src.data.image_loader import load_image
from src.data.augment import augment_image
from src.data.preprocessor import preprocess_image
# Models
from src.models.backbone import build_backbone
from src.models.head import build_head
# Training
from src.training.callbacks import FirestoreEpochCallback
from src.training.callbacks import send_job_completion_email
from src.training.adversarial import fine_tune_with_fgsm
# Schemas
from src.schemas.experiment import ExperimentConfig
# Utils
from src.utils.seed import set_global_seed
from src.utils.gcs import upload_bytes_to_blob

handler = logging.StreamHandler()
handler.setFormatter( JsonFormatter(
	'%(levelname)s %(name)s %(message)s',
	rename_fields = { 'levelname': 'severity' },
	timestamp = True,
) )
logging.basicConfig( level=logging.INFO, handlers=[ handler ], force=True )

log = logging.getLogger( __name__ )

# FAKE is the positive class (target=1) — the natural framing for a deepfake detector.
# Nothing else in the codebase pins this for the neural-net path (baseline.py's SVM path
# keeps raw string labels throughout, sidestepping the question); binary_crossentropy
# needs a numeric target, so it's decided here.
LABEL_TO_TARGET = { 'REAL': 0.0, 'FAKE': 1.0 }


def build_batches( split_df, img_size: int, patch_grid_size: int, batch_size: int, augment: bool, shuffle: bool, augmentation_seed: int | None = None ):
	'''
	Generator yielding (image_batch, label_batch) per batch, where image_batch has shape
	(batch_size, patch_grid_size ** 2, img_size, img_size, 3) and label_batch holds numeric
	targets (LABEL_TO_TARGET) in matching row order. The final batch of a split may be
	smaller than batch_size if the split size isn't a multiple of it.

	Batching is plain Python-generator batching, not tf.data.Dataset — wrapped in
	tf.data.Dataset.from_generator() by train_member() below (the seam 3.9's docstring
	already flagged for "once there's a model to feed"), not rebuilt as one internally, so
	this generator itself stays simple and independently testable.

	Shuffling (row order only) happens here, not in splitter.py (which deliberately leaves
	row order untouched). Draws from whatever global seed is currently active via seed.py —
	no local RNG instantiated for shuffling.

	augmentation_seed seeds a local, this-call-only np.random.Generator that hands each
	image its own sub-seed for augment_image() — deterministic given the same
	augmentation_seed (needed for job reruns to reproduce identical loss curves), but
	varying image-to-image within the call (not one repeated transform for the whole
	batch). None (default, and always used for augment=False callers) leaves augmentation
	off entirely — irrelevant/unused in that case.
	'''

	indices = np.arange( len( split_df ) )

	if shuffle:
		np.random.shuffle( indices )

	augmentation_rng = np.random.default_rng( augmentation_seed )

	for start in range( 0, len( indices ), batch_size ):
		batch_rows = split_df.iloc[ indices[ start : start + batch_size ] ]
		patches = []

		for _, row in batch_rows.iterrows():
			image = load_image( row[ 'image_path' ], img_size )

			if augment:
				image_seed = int( augmentation_rng.integers( 0, 2 ** 31 - 1 ) )
				image = augment_image( image, seed=image_seed )

			patches.append( preprocess_image( image, patch_grid_size, img_size ) )

		labels = np.array( [ LABEL_TO_TARGET[ label ] for label in batch_rows[ 'label' ] ], dtype=np.float32 )
		batch_images = np.stack( patches, axis=0 )

		yield batch_images, labels

		# Explicit cleanup (Task 5.5 memory-mitigation amendment) — drop the intermediate
		# per-image patches list (not batch_images/labels, already handed to the consumer
		# above) and the last loop's image buffer immediately once the consumer has this
		# batch, then force a collection, rather than waiting on this generator's own
		# scope/next-iteration reassignment to eventually free them. Every batch here is
		# ~1,300 patches' worth of same-sized float32 arrays cycling through — exactly the
		# allocation pattern that fragments glibc's allocator over many sustained steps.
		del patches
		del image
		gc.collect()


def build_patch_averaged_model( member_model: tf.keras.Model, patch_grid_size: int, img_size: int ) -> tf.keras.Model:
	'''
	Wrap a single-image backbone+head model (5.1/5.2: input (img_size, img_size, 3) ->
	scalar sigmoid) so it accepts a full patch grid per training example and averages the
	per-patch scores into one image-level prediction — steps 5-6 of the DFT patch pipeline
	(CLAUDE.md): "XCeption processes each patch independently -> scalar score. Average pool
	across all patch_grid_size**2 scores -> image-level prediction."

	TimeDistributed applies member_model to all patch_grid_size**2 patches via one batched
	call (Keras reshapes (batch, patches, H, W, 3) -> (batch*patches, H, W, 3) internally,
	not a Python loop over patches), and preserves member_model's own frozen/trainable
	layer split — freezing is a property of member_model's layers, unaffected by this
	wrapping.
	'''

	patch_count = patch_grid_size ** 2
	patch_input = tf.keras.layers.Input( shape=( patch_count, img_size, img_size, 3 ) )
	patch_scores = tf.keras.layers.TimeDistributed( member_model )( patch_input )
	image_score = tf.keras.layers.Lambda( lambda scores: tf.reduce_mean( scores, axis=1 ) )( patch_scores )

	return tf.keras.Model( inputs=patch_input, outputs=image_score )


def train_member( experiment: ExperimentConfig, job_id: str, member_index: int, splits: dict, class_weights: dict ) -> None:
	'''
	Build, train, optionally adversarially fine-tune, and persist one ensemble member —
	extracted out of main()'s per-member loop (5.5) rather than kept inline, since it's now
	called config.ensemble_size times instead of once (5.4's single-member scope). Every
	side effect (Firestore epoch writes, GCS checkpoint/log upload) is tagged with
	member_index, so members never overwrite each other's records or artefacts.

	Raises straight through on any failure — main()'s outer try/except is what turns one
	member's exception into the whole job's FAILED status; this function doesn't swallow
	errors to let sibling members continue, per 5.5's requirement that one member failing
	fails the whole job rather than silently completing with fewer members than requested.
	'''

	member_seed = set_global_seed( experiment.random_seed + member_index )
	log.info( 'Ensemble member seeded', extra={
		'job_id': job_id, 'ensemble_member': member_index, 'seed': member_seed,
	} )

	# Model — backbone + head (5.1/5.2), wrapped for the patch grid (see
	# build_patch_averaged_model docstring)
	backbone = build_backbone( experiment )
	head_output = build_head( backbone.output, experiment )
	member_model = tf.keras.Model( inputs=backbone.input, outputs=head_output )
	model = build_patch_averaged_model( member_model, experiment.patch_grid_size, experiment.img_size )

	model.compile(
		optimizer = tf.keras.optimizers.Adam( learning_rate=experiment.learning_rate ),
		loss = 'binary_crossentropy',
		metrics = [ 'accuracy' ],
	)

	# class_weight, keyed by the model's own {0.0, 1.0} target encoding rather than the
	# raw label strings splitter.py computed it from
	numeric_class_weights = { LABEL_TO_TARGET[ label ]: weight for label, weight in class_weights.items() }

	# tf.data wrapping — the seam 3.9's build_batches docstring already flagged for
	# "once there's a model to feed" (there is, now). A plain Python generator is
	# exhausted after one pass; .repeat() re-invokes the factory functions below each
	# epoch to get a fresh pass. train's factory closes over a mutable counter so
	# successive epochs get a distinct-but-deterministic augmentation_seed
	# (member_seed + epoch index) — reruns of the whole job are reproducible (same
	# member_seed -> same per-epoch seed sequence), while different epochs within one
	# run still see different augmentations, not the same one repeated every epoch.
	#
	# sample_weight (not model.fit's class_weight= kwarg) carries the class balancing —
	# class_weight's support alongside a tf.data.Dataset input varies across Keras
	# versions, while sample_weight as a third yielded array is universally supported
	# for any input type. Only the training stream is weighted; validation intentionally
	# is not, so val_loss reflects the real, unweighted loss.
	train_epoch_counter = { 'value': 0 }

	def train_generator():
		seed_for_epoch = member_seed + train_epoch_counter[ 'value' ]
		train_epoch_counter[ 'value' ] += 1

		for images, labels in build_batches(
			splits[ 'train' ], experiment.img_size, experiment.patch_grid_size,
			experiment.batch_size, augment=True, shuffle=True, augmentation_seed=seed_for_epoch,
		):
			sample_weight = np.array( [ numeric_class_weights[ label ] for label in labels ], dtype=np.float32 )

			yield images, labels, sample_weight

	def val_generator():
		yield from build_batches(
			splits[ 'val' ], experiment.img_size, experiment.patch_grid_size,
			experiment.batch_size, augment=False, shuffle=False,
		)

	patch_count = experiment.patch_grid_size ** 2
	image_spec = tf.TensorSpec( shape=( None, patch_count, experiment.img_size, experiment.img_size, 3 ), dtype=tf.float32 )
	label_spec = tf.TensorSpec( shape=( None, ), dtype=tf.float32 )

	train_dataset = tf.data.Dataset.from_generator(
		train_generator, output_signature=( image_spec, label_spec, label_spec ),
	).repeat()
	val_dataset = tf.data.Dataset.from_generator(
		val_generator, output_signature=( image_spec, label_spec ),
	).repeat()

	# Ceiling division — the final, possibly-smaller batch each pass through a split
	# still counts as one step, matching build_batches' own range(0, len, batch_size)
	steps_per_epoch = -( -len( splits[ 'train' ] ) // experiment.batch_size )
	validation_steps = -( -len( splits[ 'val' ] ) // experiment.batch_size )

	log.info( 'Model built, starting fit', extra={
		'job_id': job_id, 'ensemble_member': member_index,
		'steps_per_epoch': steps_per_epoch, 'validation_steps': validation_steps,
		'epochs': experiment.epochs, 'batch_size': experiment.batch_size,
	} )

	# Checkpoint + training log both need a real local file path (Keras' save/CSVLogger
	# APIs are file-path-based, not in-memory) — this tempdir is transient scratch, not
	# durable state. GCS (below, after the block) is the durable resting place, so the
	# ephemeral container filesystem never holds the only copy.
	with tempfile.TemporaryDirectory() as tmp_dir:
		log_csv_path = f'{ tmp_dir }/member_{ member_index }.csv'
		checkpoint_path = f'{ tmp_dir }/member_{ member_index }.keras'

		callbacks = [
			tf.keras.callbacks.EarlyStopping(
				monitor = 'val_loss', patience = experiment.early_stopping_patience, restore_best_weights = True,
			),
			tf.keras.callbacks.CSVLogger( log_csv_path ),
			FirestoreEpochCallback( job_id, member_index ),
		]

		model.fit(
			train_dataset,
			validation_data = val_dataset,
			epochs = experiment.epochs,
			steps_per_epoch = steps_per_epoch,
			validation_steps = validation_steps,
			callbacks = callbacks,
			verbose = 2,
		)

		model.save( checkpoint_path )

		with open( checkpoint_path, 'rb' ) as f:
			checkpoint_bytes = f.read()

		with open( log_csv_path, 'rb' ) as f:
			log_csv_bytes = f.read()

	# Pre-adversarial checkpoint + log uploaded and released *before* fine-tuning starts
	# (Task 5.6 memory-mitigation amendment) — previously both checkpoint byte-blobs were
	# held simultaneously in memory across one tempdir scope spanning both base training
	# and fine-tuning, uploaded only after fine-tuning finished. Uploading and dropping
	# checkpoint_bytes/log_csv_bytes here, before fine-tuning even starts, turns that into
	# two sequential checkpoint-sized allocations instead of two simultaneous ones.
	checkpoint_uri = f'gs://{ config.GCS_BUCKET }/checkpoints/{ job_id }/member_{ member_index }.keras'
	upload_bytes_to_blob( checkpoint_uri, checkpoint_bytes )
	log.info( 'Checkpoint uploaded', extra={ 'job_id': job_id, 'ensemble_member': member_index, 'checkpoint_uri': checkpoint_uri } )

	log_csv_uri = f'gs://{ config.GCS_BUCKET }/logs/{ job_id }/member_{ member_index }.csv'
	upload_bytes_to_blob( log_csv_uri, log_csv_bytes )
	log.info( 'Training log uploaded', extra={ 'job_id': job_id, 'ensemble_member': member_index, 'log_csv_uri': log_csv_uri } )

	del checkpoint_bytes
	del log_csv_bytes
	gc.collect()

	# FGSM fine-tuning (5.6) — per member, after that member's own initial training
	# converges, never on the fused ensemble output (System Architecture Doc 9.5). Saved
	# as a *separate* checkpoint (not an overwrite of the one just uploaded above) — Phase
	# 6 Stage 1 evaluation needs the pre-adversarial model's clean accuracy, Stage 2 needs
	# this adversarially-fine-tuned one; overwriting would destroy the one Stage 1 needs.
	if experiment.adversarial_training:

		# Release base-training's optimizer state before fine-tuning starts (Task 5.6
		# amendment) — model.compile() above left Adam's per-variable momentum/variance
		# accumulators resident in model.optimizer; fine_tune_with_fgsm() below builds its
		# own fresh optimizer rather than reusing this one (a distinct phase, not a
		# continuation — see adversarial.py's docstring), so the old one is dead weight
		# otherwise. Verified this doesn't break inference or model.save() afterward — the
		# checkpoints here never needed the base-training optimizer's state anyway.
		model.optimizer = None
		gc.collect()

		log.info( 'Starting adversarial fine-tuning', extra={
			'job_id': job_id, 'ensemble_member': member_index, 'epsilon': experiment.adversarial_epsilon,
		} )

		fine_tune_with_fgsm( model, train_generator(), experiment.learning_rate, experiment.adversarial_epsilon )

		log.info( 'Adversarial fine-tuning complete', extra={ 'job_id': job_id, 'ensemble_member': member_index } )

		with tempfile.TemporaryDirectory() as adversarial_tmp_dir:
			adversarial_checkpoint_path = f'{ adversarial_tmp_dir }/member_{ member_index }_adversarial.keras'
			model.save( adversarial_checkpoint_path )

			with open( adversarial_checkpoint_path, 'rb' ) as f:
				adversarial_checkpoint_bytes = f.read()

		adversarial_checkpoint_uri = f'gs://{ config.GCS_BUCKET }/checkpoints/{ job_id }/member_{ member_index }_adversarial.keras'
		upload_bytes_to_blob( adversarial_checkpoint_uri, adversarial_checkpoint_bytes )
		log.info( 'Adversarial checkpoint uploaded', extra={
			'job_id': job_id, 'ensemble_member': member_index, 'checkpoint_uri': adversarial_checkpoint_uri,
		} )

		del adversarial_checkpoint_bytes
		gc.collect()


def main() -> None:

	parser = argparse.ArgumentParser()
	parser.add_argument( '--job-id', required=True )
	args = parser.parse_args()

	job_id = args.job_id

	# Fetch job document — config comes from Firestore (written by POST /train), never
	# re-parsed from CLI args, per Section 7.4 step 6 of the architecture doc
	job = jobs.get_job( job_id )

	if job is None:
		log.error( 'Job not found in Firestore', extra={ 'job_id': job_id } )
		sys.exit( 1 )

	jobs.update_job_status( job_id, 'RUNNING' )
	log.info( 'Training job started', extra={ 'job_id': job_id } )

	experiment = job.config

	try:
		# Manifest + split
		manifest_uri = f'gs://{ config.GCS_BUCKET }/{ experiment.dataset_path }'
		manifest = load_manifest( manifest_uri, experiment.max_samples_per_split )
		log.info( 'Manifest loaded', extra={ 'job_id': job_id, 'rows': len( manifest ) } )

		splits, class_weights = split_dataset( manifest, is_truncated = experiment.max_samples_per_split is not None )
		log.info( 'Split complete', extra={
			'job_id': job_id,
			'class_weights': class_weights,
			'train_rows': len( splits[ 'train' ] ),
			'val_rows': len( splits[ 'val' ] ),
			'test_rows': len( splits[ 'test' ] ),
		} )

		# Ensemble loop (5.5) — sequential, not parallel (single GPU, no need to complicate
		# this). Each member is fully independent (own seed, own augmentation stream, own
		# checkpoint/log path — see train_member()'s docstring); an exception from any
		# member propagates straight to the except block below rather than being caught
		# per-member, so one failure fails the whole job instead of silently completing
		# with fewer members than config.ensemble_size requested.
		for member_index in range( experiment.ensemble_size ):
			train_member( experiment, job_id, member_index, splits, class_weights )

		jobs.update_job_status( job_id, 'COMPLETED' )
		send_job_completion_email( job_id, 'COMPLETED' )

	except Exception as exc:
		jobs.update_job_error( job_id, str( exc ) )
		jobs.update_job_status( job_id, 'FAILED' )
		send_job_completion_email( job_id, 'FAILED' )
		log.exception( 'Training pipeline failed', extra={ 'job_id': job_id } )
		sys.exit( 1 )

	log.info( 'Training job complete', extra={ 'job_id': job_id } )


if __name__ == '__main__':
	main()
