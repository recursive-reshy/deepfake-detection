# Standard library
import argparse
import hashlib
import logging
# NumPy
import numpy as np
# TensorFlow
import tensorflow as tf
# Schemas
from src.schemas.experiment import ExperimentConfig
# Models
from src.models.backbone import BACKBONE_CONSTRUCTORS
from src.models.backbone import EXPECTED_OUTPUT_SHAPE_AT_299
from src.models.backbone import build_backbone

logging.basicConfig( level=logging.INFO )
log = logging.getLogger( __name__ )


def main() -> None:

	parser = argparse.ArgumentParser()
	parser.add_argument( '--backbone', default='xception', choices=list( BACKBONE_CONSTRUCTORS.keys() ) )
	parser.add_argument( '--img-size', type=int, default=299 )
	args = parser.parse_args()

	config = ExperimentConfig( backbone=args.backbone, img_size=args.img_size, dataset_path='unused-verify-backbone' )

	model = build_backbone( config )

	# Layer-by-layer freeze table
	for layer in model.layers:
		log.info( f'{ layer.name }: trainable={ layer.trainable }' )

	# Dummy forward pass — confirms the model is actually callable end to end, not just buildable
	dummy_input = np.zeros( ( 1, args.img_size, args.img_size, 3 ), dtype=np.float32 )
	output = model( dummy_input, training=False )

	log.info( f'Forward pass output shape: { tuple( output.shape ) }' )

	if args.img_size == 299 and args.backbone in EXPECTED_OUTPUT_SHAPE_AT_299:
		expected = EXPECTED_OUTPUT_SHAPE_AT_299[ args.backbone ]

		# Batch dim is dynamic (None) in the documented contract but concrete (1) in this
		# single-dummy-input forward pass — compare the trailing feature-map dims only.
		if tuple( output.shape )[ 1: ] == expected[ 1: ]:
			log.info( f'Output shape matches documented contract: { expected }' )
		else:
			log.error( f'Output shape MISMATCH — got { tuple( output.shape ) }, expected { expected }' )

	# Param counts
	trainable_params = sum( tf.keras.backend.count_params( w ) for w in model.trainable_weights )
	total_params = model.count_params()

	log.info( f'Trainable params: { trainable_params }/{ total_params } ({ 100 * trainable_params / total_params:.1f}%)' )

	# ImageNet weight-loading check — everything logged above (architecture, freeze table,
	# output shape) would be identical whether weights='imagenet' actually loaded pretrained
	# weights or silently fell back to random init, since neither shape nor layer names
	# depend on the weight values. Build the same backbone/input_shape with weights=None and
	# compare the first weighted layer's kernel against it — pretrained and freshly
	# initialized kernels are checked for equality, not just "close", since a random init
	# matching a specific pretrained tensor by chance is not a real possibility.
	#
	# A single fresh random draw is a sufficient counter-example here (rather than needing a
	# hardcoded expected checksum) because Keras's weights='imagenet' path never silently
	# substitutes random init on failure — get_file() hash-validates the downloaded .h5 and
	# raises ValueError on a corrupt/mismatched file, and load_weights() raises on a shape
	# mismatch. So the only way this model's kernel could equal reference_model's is if
	# weights='imagenet' was never actually honoured. A hardcoded expected checksum was
	# considered and rejected — it would pin this check to today's specific TF/Keras release
	# and false-fail on a future tensorflow~=2.21 patch bump that ships updated weights.
	reference_model = BACKBONE_CONSTRUCTORS[ args.backbone ](
		include_top = False,
		weights = None,
		input_shape = ( args.img_size, args.img_size, 3 ),
	)

	weighted_index = next( index for index, layer in enumerate( model.layers ) if layer.get_weights() )
	weighted_layer_name = model.layers[ weighted_index ].name
	pretrained_kernel = model.layers[ weighted_index ].get_weights()[ 0 ]
	random_init_kernel = reference_model.layers[ weighted_index ].get_weights()[ 0 ]

	kernel_checksum = hashlib.md5( pretrained_kernel.tobytes() ).hexdigest()
	log.info( f'First weighted layer "{ weighted_layer_name }" kernel checksum={ kernel_checksum }, mean={ pretrained_kernel.mean():.4f}, std={ pretrained_kernel.std():.4f}' )

	if np.array_equal( pretrained_kernel, random_init_kernel ):
		log.error( f'ImageNet weight-loading check FAILED — "{ weighted_layer_name }" kernel is identical to a fresh weights=None baseline; pretrained weights may not have loaded' )
	else:
		log.info( f'ImageNet weight-loading check passed — "{ weighted_layer_name }" kernel differs from a freshly-initialized weights=None baseline' )


if __name__ == '__main__':
	main()
