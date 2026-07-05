# Standard library
import argparse
import logging
import sys
# NumPy
import numpy as np
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
# Utils
from src.utils.seed import set_global_seed

handler = logging.StreamHandler()
handler.setFormatter( JsonFormatter(
	'%(levelname)s %(name)s %(message)s',
	rename_fields = { 'levelname': 'severity' },
	timestamp = True,
) )
logging.basicConfig( level=logging.INFO, handlers=[ handler ], force=True )

log = logging.getLogger( __name__ )


def build_batches( split_df, img_size: int, patch_grid_size: int, batch_size: int, augment: bool, shuffle: bool ):
	'''
	Generator yielding (image_batch, label_batch) per batch, where image_batch has shape
	(batch_size, patch_grid_size ** 2, img_size, img_size, 3) and label_batch holds the
	raw string labels ('REAL' / 'FAKE') for that batch, in matching row order. The final
	batch of a split may be smaller than batch_size if the split size isn't a multiple of it.

	Batching is plain Python-generator batching, not tf.data.Dataset — sufficient for this
	plumbing sub-step (Phase 3.9). The per-image FFT + 81-patch resize cost flagged in 3.6
	(~85ms/image) means this generator is the natural seam for Phase 5 to wrap in
	tf.data.Dataset.from_generator with a parallel .map() if real training throughput
	demands it — not done here since there's no model yet to feed.

	Shuffling (row order only) happens here, not in splitter.py (which deliberately leaves
	row order untouched). Draws from whatever global seed is currently active via seed.py —
	no local RNG instantiated in this function.
	'''

	indices = np.arange( len( split_df ) )

	if shuffle:
		np.random.shuffle( indices )

	for start in range( 0, len( indices ), batch_size ):
		batch_rows = split_df.iloc[ indices[ start : start + batch_size ] ]
		patches = []

		for _, row in batch_rows.iterrows():
			image = load_image( row[ 'image_path' ], img_size )

			if augment:
				image = augment_image( image )

			patches.append( preprocess_image( image, patch_grid_size, img_size ) )

		yield np.stack( patches, axis=0 ), batch_rows[ 'label' ].to_numpy()


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
		manifest = load_manifest( manifest_uri )
		log.info( 'Manifest loaded', extra={ 'job_id': job_id, 'rows': len( manifest ) } )

		splits, class_weights = split_dataset( manifest )
		log.info( 'Split complete', extra={
			'job_id': job_id,
			'class_weights': class_weights,
			'train_rows': len( splits[ 'train' ] ),
			'val_rows': len( splits[ 'val' ] ),
			'test_rows': len( splits[ 'test' ] ),
		} )

		# Ensemble loop — members differentiated by seed, per Section 9.3 of the
		# architecture doc. Re-seeded once per member, not once for the whole run.
		for member_index in range( experiment.ensemble_size ):
			member_seed = set_global_seed( experiment.random_seed + member_index )
			log.info( 'Ensemble member seeded', extra={
				'job_id': job_id, 'ensemble_member': member_index, 'seed': member_seed,
			} )

			train_batches = build_batches(
				splits[ 'train' ], experiment.img_size, experiment.patch_grid_size,
				experiment.batch_size, augment=True, shuffle=True,
			)
			val_batches = build_batches(
				splits[ 'val' ], experiment.img_size, experiment.patch_grid_size,
				experiment.batch_size, augment=False, shuffle=False,
			)

			# One batch pulled from each here to prove the pipeline shape end to end.
			# Phase 5's real model.fit call consumes the full generators for
			# experiment.epochs epochs — that loop lives inside the model call, not here.
			train_images, train_labels = next( train_batches )
			val_images, val_labels = next( val_batches )

			log.info( 'Batch produced', extra={
				'job_id': job_id, 'ensemble_member': member_index,
				'split': 'train', 'batch_shape': list( train_images.shape ),
			} )
			log.info( 'Batch produced', extra={
				'job_id': job_id, 'ensemble_member': member_index,
				'split': 'val', 'batch_shape': list( val_images.shape ),
			} )

			# --- Phase 5 placeholder — model construction and training not yet built ---
			# model = build_ensemble_member( experiment, member_index )
			# model.fit(
			#     train_dataset, validation_data = val_dataset,
			#     epochs = experiment.epochs, class_weight = class_weights,
			#     callbacks = build_callbacks( job_id, member_index ),
			# )
			log.info( 'Phase 5 placeholder reached — model training not yet implemented', extra={
				'job_id': job_id, 'ensemble_member': member_index, 'class_weights': class_weights,
			} )

	except Exception as exc:
		jobs.update_job_error( job_id, str( exc ) )
		jobs.update_job_status( job_id, 'FAILED' )
		log.exception( 'Training pipeline failed', extra={ 'job_id': job_id } )
		sys.exit( 1 )

	# Status is deliberately left at RUNNING, not COMPLETED — see handoff notes.
	log.info( 'Data pipeline plumbing check complete — model training is Phase 5', extra={ 'job_id': job_id } )


if __name__ == '__main__':
	main()
