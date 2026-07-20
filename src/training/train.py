# Standard library
import argparse
import gc
import logging
import os
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
from src.utils.gcs import download_blob_to_bytes
from src.utils.vertex import submit_vertex_job

handler = logging.StreamHandler()
handler.setFormatter( JsonFormatter(
	'%(levelname)s %(name)s %(message)s',
	rename_fields = { 'levelname': 'severity' },
	timestamp = True,
) )
logging.basicConfig( level=logging.INFO, handlers=[ handler ], force=True )

log = logging.getLogger( __name__ )

# Git SHA (baked in at build time by the Dockerfile's GIT_SHA build arg) — logged first,
# before any GPU/model work, so "is this the code we think it is" is answerable from this
# job's log even if everything after this line fails.
log.info( 'Training container started', extra={ 'git_sha': os.getenv( 'GIT_SHA', 'unknown' ) } )

# GPU memory-growth configuration (fine-tuning-phase GPU-VRAM OOM amendment) — must run
# before any other TF operation that touches a GPU device (model construction included),
# since TF raises once physical devices are already initialized; nothing above this point
# does. Default BFC behaviour claims a large upfront memory pool per process and doesn't
# hand any of it back mid-process — the OOM evidence this responds to was a ~22MB
# allocation failing on a 20GB GPU during fine-tuning, after base training's own compiled
# tf.function graphs had already run at a different input shape (chunked FGSM forward
# passes vs. base training's full-batch steps), which points at near-total pool exhaustion
# left over from training rather than any single operation being too large (already ruled
# out — see FGSM_CHUNK_SIZE in adversarial.py). Incremental growth trades this away for
# allocating only what's actually requested, at the cost of not having the single big
# pool's allocation-pattern predictability — an acceptable trade here since correctness
# (not fitting in 20GB at all) matters more than that predictability.
gpus = tf.config.experimental.list_physical_devices( 'GPU' )

for gpu in gpus:
	tf.config.experimental.set_memory_growth( gpu, True )

log.info( 'GPU memory-growth configured', extra={
	'gpu_count': len( gpus ), 'gpu_devices': [ gpu.name for gpu in gpus ],
} )

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


def log_gpu_memory( phase: str, job_id: str, member_index: int ) -> None:
	'''
	Best-effort GPU memory snapshot (current/peak allocated bytes), taken around
	model.fit() and the cached-compiled-function release that follows it — added to
	confirm or rule out the fine-tuning-phase OOM hypothesis (base training's compiled
	tf.function graphs never releasing GPU memory back to the pool) directly from the
	next real Vertex AI run's logs, rather than inferring it from a stack trace alone.

	Silently skipped, never raises, if no GPU is visible (true on this dev machine) or if
	get_memory_info doesn't recognise the device name — diagnostic only, must never break
	an otherwise-working training step over a logging call.
	'''

	gpus = tf.config.list_physical_devices( 'GPU' )

	if not gpus:
		return

	try:
		memory_info = tf.config.experimental.get_memory_info( 'GPU:0' )
	except ValueError:
		return

	log.info( 'GPU memory snapshot', extra={
		'job_id': job_id, 'ensemble_member': member_index, 'phase': phase,
		'current_bytes': memory_info[ 'current' ], 'peak_bytes': memory_info[ 'peak' ],
	} )


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


def build_generators( experiment: ExperimentConfig, member_seed: int, splits: dict, class_weights: dict ):
	'''
	Shared by both stages — train_member_base() wraps these in tf.data.Dataset for
	model.fit(); train_member_adversarial() calls train_generator() directly as
	fine_tune_with_fgsm()'s raw batch iterable (fine-tuning was never wrapped in
	tf.data.Dataset, even before the two-stage split). Same member_seed-derived
	augmentation-seed sequencing in both stages, so a given member_index draws from the
	same seeded augmentation stream regardless of which stage calls this.

	train_epoch_counter always starts fresh at 0 whenever this is called. In Stage 2 (a
	separate process from Stage 1) that means fine-tuning's single pass reuses the same
	augmentation_seed as Stage 1's first base-training epoch, rather than continuing the
	sequence Stage 1 left off at (only possible for the in-process version, before this
	split, since the counter lived in the same Python object across both phases). Accepted
	as a minor, deliberate behaviour change: FGSM's perturbation is computed fresh from the
	model's current (post-base-training) gradients regardless of which valid augmented
	version of an image it's given, so this only affects augmentation-sequence novelty, not
	correctness.

	class_weight, keyed by the model's own {0.0, 1.0} target encoding rather than the raw
	label strings splitter.py computed it from. sample_weight (not model.fit's
	class_weight= kwarg) carries the class balancing — class_weight's support alongside a
	tf.data.Dataset input varies across Keras versions, while sample_weight as a third
	yielded array is universally supported for any input type. Only the training stream is
	weighted; validation intentionally is not, so val_loss reflects the real, unweighted
	loss.
	'''

	numeric_class_weights = { LABEL_TO_TARGET[ label ]: weight for label, weight in class_weights.items() }
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

	return train_generator, val_generator


def train_member_base( experiment: ExperimentConfig, job_id: str, member_index: int, splits: dict, class_weights: dict ) -> None:
	'''
	Stage 1 worker — build, base-train, and persist one ensemble member's checkpoint. Never
	runs adversarial fine-tuning itself: the two-stage split (see run_base_stage /
	run_adversarial_stage below) moved that to a separate Vertex AI job so it always starts
	in a fresh container. In-process cleanup between base training and fine-tuning
	(model.optimizer = None, train_function/test_function release, GPU memory-growth) was
	tried first and wasn't reliable enough to depend on — the allocator dump from the real
	OOM this responds to showed LargestFreeBlock: 0B at the moment fine-tuning issued its
	first op, meaning base training's own retained graph state had already saturated the
	pool before fine-tuning got a chance to request anything.

	The train_function/test_function/optimizer release below still matters here, just for a
	different transition than before this split — member-to-member within this same
	stage/process, not base-to-adversarial (that boundary is now a process exit).

	Raises straight through on any failure, same as before the split — one member's
	exception still fails the whole Stage 1 job (see run_base_stage).
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

	train_generator, val_generator = build_generators( experiment, member_seed, splits, class_weights )

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

		log_gpu_memory( 'before_fit', job_id, member_index )

		model.fit(
			train_dataset,
			validation_data = val_dataset,
			epochs = experiment.epochs,
			steps_per_epoch = steps_per_epoch,
			validation_steps = validation_steps,
			callbacks = callbacks,
			verbose = 2,
		)

		log_gpu_memory( 'after_fit', job_id, member_index )

		# Release model.fit()'s cached compiled tf.function graphs (fine-tuning-phase
		# GPU-VRAM OOM amendment) — model.fit()/its validation_data pass trace and cache
		# train_function/test_function shaped for this call's batch; TF's allocator claims
		# GPU memory for their kernel workspace and doesn't hand it back on its own.
		# Fine-tuning below calls this same model directly at a different input shape
		# (FGSM's per-chunk forward/backward passes — see FGSM_CHUNK_SIZE in
		# adversarial.py), needing its own headroom on top of whatever these cached graphs
		# are still holding. Neither is needed again regardless of whether fine-tuning
		# runs: this member's model.fit() call is already done, and fine_tune_with_fgsm()
		# never calls .fit()/.evaluate(). Confirmed against Keras 3's own TensorFlow
		# trainer (src/backend/tensorflow/trainer.py) — both attributes default to None
		# and are lazily rebuilt via make_train_function()/make_test_function() the next
		# time fit()/evaluate() is called, so this is the same reset the framework itself
		# does, not an internals hack.
		model.train_function = None
		model.test_function = None
		model.optimizer = None
		gc.collect()

		log_gpu_memory( 'after_cached_function_release', job_id, member_index )

		model.save( checkpoint_path )

		with open( checkpoint_path, 'rb' ) as f:
			checkpoint_bytes = f.read()

		with open( log_csv_path, 'rb' ) as f:
			log_csv_bytes = f.read()

	checkpoint_uri = f'gs://{ config.GCS_BUCKET }/checkpoints/{ job_id }/member_{ member_index }.keras'
	upload_bytes_to_blob( checkpoint_uri, checkpoint_bytes )
	jobs.update_job_member_checkpoint( job_id, member_index, checkpoint_uri )
	log.info( 'Checkpoint uploaded', extra={ 'job_id': job_id, 'ensemble_member': member_index, 'checkpoint_uri': checkpoint_uri } )

	log_csv_uri = f'gs://{ config.GCS_BUCKET }/logs/{ job_id }/member_{ member_index }.csv'
	upload_bytes_to_blob( log_csv_uri, log_csv_bytes )
	log.info( 'Training log uploaded', extra={ 'job_id': job_id, 'ensemble_member': member_index, 'log_csv_uri': log_csv_uri } )

	del checkpoint_bytes
	del log_csv_bytes
	gc.collect()


def train_member_adversarial( experiment: ExperimentConfig, job_id: str, member_index: int, checkpoint_uri: str, splits: dict, class_weights: dict ) -> None:
	'''
	Stage 2 worker — reload one ensemble member's base-trained checkpoint fresh from GCS
	(this always runs in a brand-new process/container, never the one that ran Stage 1 —
	see run_adversarial_stage), FGSM fine-tune it, upload the adversarial checkpoint as a
	*separate* artefact from the pre-adversarial one (Phase 6 Stage 1 evaluation needs the
	pre-adversarial model's clean accuracy, Stage 2 needs this one; overwriting would
	destroy the one Stage 1 needs — same reasoning as before the two-stage split).

	Reconstructs the architecture via build_backbone/build_head/build_patch_averaged_model
	and loads only the weights (model.load_weights), rather than
	tf.keras.models.load_model() on the full .keras archive — build_patch_averaged_model's
	Lambda layer needs safe_mode=False to deserialize via load_model, since Keras 3 treats
	arbitrary Lambda function bytecode as unsafe to deserialize by default. Reconstructing
	the (already-known, already-tested) architecture and loading weights only sidesteps
	that, no safe_mode override needed. Verified robust to Keras's auto-generated layer
	names differing between the checkpoint's original build and this fresh one (which they
	will, across members processed in one Stage 2 loop, since Keras's naming counter keeps
	incrementing within one process) — load_weights() on a .keras file matches by
	topological order, not by name.

	clear_session() at the start resets Keras's global layer-naming counters and releases
	the previous member's graph/session state — memory hygiene for the member-to-member
	loop within this stage, mirroring train_member_base()'s own between-member cleanup.
	'''

	tf.keras.backend.clear_session()

	member_seed = set_global_seed( experiment.random_seed + member_index )
	log.info( 'Ensemble member seeded for adversarial fine-tuning', extra={
		'job_id': job_id, 'ensemble_member': member_index, 'seed': member_seed,
	} )

	backbone = build_backbone( experiment )
	head_output = build_head( backbone.output, experiment )
	member_model = tf.keras.Model( inputs=backbone.input, outputs=head_output )
	model = build_patch_averaged_model( member_model, experiment.patch_grid_size, experiment.img_size )

	with tempfile.TemporaryDirectory() as tmp_dir:
		local_checkpoint_path = f'{ tmp_dir }/member_{ member_index }.keras'
		checkpoint_bytes = download_blob_to_bytes( checkpoint_uri )

		with open( local_checkpoint_path, 'wb' ) as f:
			f.write( checkpoint_bytes )

		del checkpoint_bytes
		gc.collect()

		model.load_weights( local_checkpoint_path )

	log.info( 'Checkpoint loaded from GCS', extra={ 'job_id': job_id, 'ensemble_member': member_index, 'checkpoint_uri': checkpoint_uri } )

	train_generator, _ = build_generators( experiment, member_seed, splits, class_weights )

	log_gpu_memory( 'before_finetune', job_id, member_index )

	log.info( 'Starting adversarial fine-tuning', extra={
		'job_id': job_id, 'ensemble_member': member_index, 'epsilon': experiment.adversarial_epsilon,
	} )

	fine_tune_with_fgsm( model, train_generator(), experiment.learning_rate, experiment.adversarial_epsilon )

	log.info( 'Adversarial fine-tuning complete', extra={ 'job_id': job_id, 'ensemble_member': member_index } )

	log_gpu_memory( 'after_finetune', job_id, member_index )

	with tempfile.TemporaryDirectory() as adversarial_tmp_dir:
		adversarial_checkpoint_path = f'{ adversarial_tmp_dir }/member_{ member_index }_adversarial.keras'
		model.save( adversarial_checkpoint_path )

		with open( adversarial_checkpoint_path, 'rb' ) as f:
			adversarial_checkpoint_bytes = f.read()

	adversarial_checkpoint_uri = f'gs://{ config.GCS_BUCKET }/checkpoints/{ job_id }/member_{ member_index }_adversarial.keras'
	upload_bytes_to_blob( adversarial_checkpoint_uri, adversarial_checkpoint_bytes )
	jobs.update_job_member_adversarial_checkpoint( job_id, member_index, adversarial_checkpoint_uri )
	log.info( 'Adversarial checkpoint uploaded', extra={
		'job_id': job_id, 'ensemble_member': member_index, 'checkpoint_uri': adversarial_checkpoint_uri,
	} )

	del adversarial_checkpoint_bytes
	gc.collect()


def run_base_stage( experiment: ExperimentConfig, job_id: str, splits: dict, class_weights: dict ) -> None:
	'''
	Stage 1: base-train every ensemble member, then either finish the job outright
	(adversarial_training=False — this job never wanted a Stage 2 at all, mirroring the
	pre-split code's own `if experiment.adversarial_training:` gate) or self-submit Stage 2
	as a brand-new Vertex AI job and return, leaving that job to finish things. Sequential,
	not parallel, across members — single GPU, no need to complicate this. Each member is
	fully independent (own seed, own augmentation stream, own checkpoint/log path — see
	train_member_base()'s docstring); an exception from any member propagates straight out
	of this function to main()'s except block, so one member failing fails the whole Stage
	1 job rather than silently completing with fewer members than requested.
	'''

	jobs.update_job_training_stage( job_id, 'base_training' )

	for member_index in range( experiment.ensemble_size ):
		train_member_base( experiment, job_id, member_index, splits, class_weights )

	log.info( 'Base training complete for all members', extra={
		'job_id': job_id, 'ensemble_size': experiment.ensemble_size,
	} )

	if not experiment.adversarial_training:
		jobs.update_job_training_stage( job_id, 'complete' )
		jobs.update_job_status( job_id, 'COMPLETED' )
		send_job_completion_email( job_id, 'COMPLETED' )
		log.info( 'adversarial_training disabled — job finished after Stage 1', extra={ 'job_id': job_id } )
		return

	try:
		vertex_job_id = submit_vertex_job( job_id, 'adversarial' )
		jobs.update_job_vertex_job_id( job_id, 'adversarial', vertex_job_id )
		jobs.update_job_training_stage( job_id, 'adversarial_finetuning' )
		log.info( 'Stage 2 (adversarial fine-tuning) submitted', extra={
			'job_id': job_id, 'vertex_job_id': vertex_job_id,
		} )

	except Exception as exc:
		# Base training itself succeeded — every member's checkpoint is already uploaded
		# and durable in GCS. Only the Stage 2 submission call failed (transient GCP API
		# error, quota, etc.), so this gets a status distinct from FAILED: training didn't
		# fail, the automatic hand-off to Stage 2 did. Deliberately not re-raised — letting
		# this propagate to main()'s generic except block below would overwrite this more
		# specific status with plain FAILED, losing exactly the distinction this exists to
		# make. A manual re-submission path (reusing these checkpoints, no retraining
		# needed) is flagged as a follow-up, not built here — see the handoff brief this
		# responds to.
		jobs.update_job_error( job_id, str( exc ) )
		jobs.update_job_status( job_id, 'STAGE2_SUBMISSION_FAILED' )
		send_job_completion_email( job_id, 'STAGE2_SUBMISSION_FAILED' )
		log.exception( 'Stage 2 submission failed after Stage 1 completed successfully', extra={ 'job_id': job_id } )


def run_adversarial_stage( experiment: ExperimentConfig, job_id: str, member_checkpoints: dict, splits: dict, class_weights: dict ) -> None:
	'''
	Stage 2: reload each member's checkpoint fresh from GCS (a separate process from Stage
	1 — nothing survives a process exit, which is the whole point of this split), FGSM
	fine-tune, upload the adversarial checkpoint. Which members exist and where to load
	them from comes from member_checkpoints (the job document's own field, written by Stage
	1) — never from CLI args/env vars, per the two-stage split's Firestore-only hand-off
	design. An exception from any member propagates straight out of this function to
	main()'s except block, same failure semantics as Stage 1.
	'''

	jobs.update_job_training_stage( job_id, 'adversarial_finetuning' )

	if not member_checkpoints:
		raise RuntimeError( f'No member checkpoints found on job { job_id } — Stage 2 cannot run before Stage 1 has uploaded at least one' )

	for member_index_str, checkpoint_uri in member_checkpoints.items():
		train_member_adversarial( experiment, job_id, int( member_index_str ), checkpoint_uri, splits, class_weights )

	jobs.update_job_training_stage( job_id, 'complete' )
	jobs.update_job_status( job_id, 'COMPLETED' )
	send_job_completion_email( job_id, 'COMPLETED' )
	log.info( 'Adversarial fine-tuning complete for all members — job finished', extra={ 'job_id': job_id } )


def main() -> None:

	parser = argparse.ArgumentParser()
	parser.add_argument( '--job-id', required=True )
	parser.add_argument( '--stage', required=True, choices=[ 'base', 'adversarial' ] )
	args = parser.parse_args()

	job_id = args.job_id
	stage = args.stage

	# Fetch job document — config comes from Firestore (written by POST /train), never
	# re-parsed from CLI args, per Section 7.4 step 6 of the architecture doc. Stage 2
	# additionally depends on this same read to discover member_checkpoints, written by
	# Stage 1's own run in a separate process/container.
	job = jobs.get_job( job_id )

	if job is None:
		log.error( 'Job not found in Firestore', extra={ 'job_id': job_id } )
		sys.exit( 1 )

	jobs.update_job_status( job_id, 'RUNNING' )
	log.info( 'Training job started', extra={ 'job_id': job_id, 'stage': stage } )

	experiment = job.config

	try:
		# Manifest + split — reloaded independently by whichever stage is running. Stage 2
		# can't reuse Stage 1's in-memory splits (separate process), only job.config, which
		# is identical either way since both stages read the same Firestore job_id.
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

		if stage == 'base':
			run_base_stage( experiment, job_id, splits, class_weights )
		else:
			run_adversarial_stage( experiment, job_id, job.member_checkpoints, splits, class_weights )

	except Exception as exc:
		jobs.update_job_error( job_id, str( exc ) )
		jobs.update_job_status( job_id, 'FAILED' )
		send_job_completion_email( job_id, 'FAILED' )
		log.exception( 'Training pipeline failed', extra={ 'job_id': job_id, 'stage': stage } )
		sys.exit( 1 )

	log.info( 'Training job process complete', extra={ 'job_id': job_id, 'stage': stage } )


if __name__ == '__main__':
	main()
