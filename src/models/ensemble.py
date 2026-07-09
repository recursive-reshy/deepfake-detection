# Standard library
import logging
# TensorFlow
import tensorflow as tf
# Schemas
from src.schemas.experiment import ExperimentConfig
# Models
from src.models.backbone import build_backbone
from src.models.head import build_head
# Utils
from src.utils.seed import set_global_seed

log = logging.getLogger( __name__ )


def build_ensemble( config: ExperimentConfig ) -> list[ dict ]:
	'''
	Build config.ensemble_size independently-instantiated backbone+head models (never
	hardcoded to 3 — the schema supports 1-5, even though deepfake-detection's production
	config always uses 3), differentiated by seed per System Architecture Doc 9.3.

	Returns a list of per-member dicts: { 'model': tf.keras.Model, 'seed': int,
	'augmentation_seed': int } — plain dicts rather than a wrapper class, consistent with
	the rest of this codebase preferring plain data structures over custom classes
	(splitter.py/loader.py return tuples/dicts, not wrapper objects). Every member is
	independently accessible, not just the fused prediction — 5.4's per-member training
	loop and Phase 6's per-classifier loss-curve plot both need this.

	Seed differentiation: member_index's seed is config.random_seed + member_index,
	deterministic (not random.random() at construction time) so reruns reproduce
	identical results, per Phase 4's reproducibility bar. set_global_seed() is called once
	per member, immediately before that member's layers are constructed — this is the same
	per-member reseed point train.py's own ensemble loop already uses (Section 9.3), not a
	new scheme. It only actually changes each member's *head* weights: the backbone's
	weights come from the same fixed ImageNet checkpoint regardless of seed (weights are
	loaded from disk, not randomly drawn), so backbone weights are identical across
	members by design — only head.py's freshly-initialized Dense layers diverge, and that
	divergence is what makes ensembling non-degenerate.

	Augmentation-stream differentiation: each member's augmentation_seed is threaded
	through to augment.py's augment_image(image, seed=...) by the caller (train.py, 5.4) —
	this module only assigns the seed, it does not call augment_image itself (no data/file
	I/O happens here, consistent with backbone.py/head.py). Reusing the member's own seed
	value for both weight init and augmentation is safe: TF's global RNG (weight init) and
	Albumentations' A.Compose RNG (augment.py) are entirely separate, independent
	generators — see augment.py's own note — so the same integer seeding both carries no
	cross-contamination risk.

	No file I/O, no GCS/Firestore calls — pure model construction, same as backbone.py and
	head.py.
	'''

	members = []

	for member_index in range( config.ensemble_size ):
		member_seed = set_global_seed( config.random_seed + member_index )

		backbone = build_backbone( config )
		output_tensor = build_head( backbone.output, config )
		model = tf.keras.Model( inputs=backbone.input, outputs=output_tensor )

		members.append( {
			'model': model,
			'seed': member_seed,
			'augmentation_seed': member_seed,
		} )

	log.info( f'Ensemble built — { len( members ) } member(s), seeds: { [ member[ "seed" ] for member in members ] }' )

	return members


def fuse_predictions( member_predictions: list[ tf.Tensor ] ) -> tf.Tensor:
	'''
	Late fusion — simple average of member sigmoid outputs, per the locked System
	Architecture Doc 9.3 decision (no learned weighting, considered and rejected — see
	CLAUDE.md). Kept as its own function, separate from member construction and from
	predict_ensemble()'s per-member forward passes, so the fusion rule could be swapped
	later without touching either — not on the table for this coursework, but the
	separation costs nothing now.
	'''

	return tf.reduce_mean( tf.stack( member_predictions, axis=0 ), axis=0 )


def predict_ensemble( members: list[ dict ], x: tf.Tensor, training: bool = False ) -> tf.Tensor:
	'''
	Run x through every member's model and fuse the results. training is passed straight
	through to each member's model call — True during a real fit step, False at
	eval/inference (see head.py's dropout behaviour under each).
	'''

	member_predictions = [ member[ 'model' ]( x, training=training ) for member in members ]

	return fuse_predictions( member_predictions )
