# Standard library
import argparse
import logging
# NumPy
import numpy as np
# Schemas
from src.schemas.experiment import ExperimentConfig
# Data
from src.data.augment import augment_image
# Models
from src.models.ensemble import build_ensemble
from src.models.ensemble import predict_ensemble

logging.basicConfig( level=logging.INFO )
log = logging.getLogger( __name__ )

# build_head() always appends exactly 4 layers, in this fixed order, to whatever backbone
# it's attached to: GlobalAveragePooling2D, Dense(512), Dropout, Dense(1). That makes
# model.layers[ -3 ] reliably "the Dense(512) layer" regardless of backbone choice — used
# below as the one layer whose kernel is actually seed-differentiated per member (see
# ensemble.py's build_ensemble() docstring: backbone weights are identical across members
# by design, only the head diverges).
HEAD_DENSE_512_LAYER_INDEX = -3


def build_smoke_test_config( ensemble_size: int ) -> ExperimentConfig:

	# ensemble_size is passed at construction time, not set post-hoc on an already-built
	# ExperimentConfig — Pydantic v2 doesn't re-run field validators (the schema's ge=1,
	# le=5 bound on ensemble_size) on plain attribute assignment unless validate_assignment
	# is turned on, which ExperimentConfig doesn't do.
	return ExperimentConfig(
		backbone = 'xception',
		img_size = 299,
		dropout_rate = 0.5,
		random_seed = 42,
		ensemble_size = ensemble_size,
		dataset_path = 'unused-verify-ensemble',
	)


def main() -> None:

	parser = argparse.ArgumentParser()
	parser.add_argument( '--ensemble-size', type=int, default=3 )
	args = parser.parse_args()

	config = build_smoke_test_config( args.ensemble_size )

	# Build — confirms N distinct member models are created without error
	members = build_ensemble( config )

	assert len( members ) == config.ensemble_size, f'Expected { config.ensemble_size } members, got { len( members ) }'

	model_ids = [ id( member[ 'model' ] ) for member in members ]

	assert len( set( model_ids ) ) == len( members ), 'Two or more members point at the same model instance'

	log.info( f'Build check passed — { len( members ) } distinct member model(s) created (ensemble_size={ config.ensemble_size })' )

	if len( members ) < 2:
		log.info( 'ensemble_size < 2 — skipping divergence/fusion checks, which need >= 2 members to compare' )
		return

	# Weight-divergence check — the 3 members' head Dense(512) kernels must all differ from
	# each other, proving seed differentiation actually took effect. A bug that dropped the
	# per-member reseed would silently produce identical models, defeating ensembling.
	head_kernels = [ member[ 'model' ].layers[ HEAD_DENSE_512_LAYER_INDEX ].get_weights()[ 0 ] for member in members ]

	divergent_pairs = [
		not np.array_equal( head_kernels[ i ], head_kernels[ j ] )
		for i in range( len( head_kernels ) ) for j in range( i + 1, len( head_kernels ) )
	]

	if all( divergent_pairs ):
		log.info( 'Weight-divergence check passed — every pair of members has a distinct head Dense(512) kernel' )
	else:
		log.error( 'Weight-divergence check FAILED — at least two members have an identical head Dense(512) kernel' )

	# Augmentation-stream check — the same single image run through each member's
	# augmentation_seed must produce a different result, proving distinct augmentation
	# streams. Uses a plain random uint8 array, not a real training image — augment_image's
	# contract only cares about dtype/value-range (image_loader.py's output shape), not content.
	dummy_image = np.random.randint( 0, 256, size=( 299, 299, 3 ), dtype=np.uint8 )
	augmented_outputs = [ augment_image( dummy_image, seed=member[ 'augmentation_seed' ] ) for member in members ]

	augmentation_divergent_pairs = [
		not np.array_equal( augmented_outputs[ i ], augmented_outputs[ j ] )
		for i in range( len( augmented_outputs ) ) for j in range( i + 1, len( augmented_outputs ) )
	]

	if all( augmentation_divergent_pairs ):
		log.info( 'Augmentation-stream check passed — every pair of members produced a different augmented image' )
	else:
		log.error( 'Augmentation-stream check FAILED — at least two members produced an identical augmented image' )

	# Fusion check — fused output must equal the manual average of the individual members'
	# outputs, computed independently here (fresh forward passes, plain np.mean) rather
	# than reusing ensemble.py's own fuse_predictions(), so this can't share a bug with the
	# code it's checking.
	dummy_input = np.zeros( ( 1, config.img_size, config.img_size, 3 ), dtype=np.float32 )

	individual_predictions = [ member[ 'model' ]( dummy_input, training=False ).numpy() for member in members ]
	manual_average = np.mean( np.stack( individual_predictions, axis=0 ), axis=0 )

	fused_output = predict_ensemble( members, dummy_input, training=False ).numpy()

	if np.allclose( fused_output, manual_average, atol=1e-6 ):
		log.info( f'Fusion check passed — ensemble output { fused_output[ 0, 0 ]:.6f} matches manual average { manual_average[ 0, 0 ]:.6f}' )
	else:
		log.error( f'Fusion check FAILED — ensemble output { fused_output[ 0, 0 ]:.6f} != manual average { manual_average[ 0, 0 ]:.6f}' )

	# Reproducibility check — rebuilding with the same random_seed must reproduce identical
	# head weights for every member, confirming determinism (not just "differs from
	# siblings", but "reproducible across independent builds").
	rebuilt_members = build_ensemble( config )
	rebuilt_head_kernels = [ member[ 'model' ].layers[ HEAD_DENSE_512_LAYER_INDEX ].get_weights()[ 0 ] for member in rebuilt_members ]

	reproducible = all(
		np.array_equal( head_kernels[ index ], rebuilt_head_kernels[ index ] )
		for index in range( len( members ) )
	)

	if reproducible:
		log.info( 'Reproducibility check passed — rebuilding with the same random_seed reproduced identical head weights for every member' )
	else:
		log.error( 'Reproducibility check FAILED — rebuilding with the same random_seed produced different head weights for at least one member' )


if __name__ == '__main__':
	main()
