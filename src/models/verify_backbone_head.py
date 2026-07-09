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
from src.models.head import build_head

logging.basicConfig( level=logging.INFO )
log = logging.getLogger( __name__ )


def main() -> None:

	parser = argparse.ArgumentParser()
	parser.add_argument( '--backbone', default='xception', choices=list( BACKBONE_CONSTRUCTORS.keys() ) )
	parser.add_argument( '--img-size', type=int, default=299 )
	parser.add_argument( '--dropout-rate', type=float, default=0.5 )
	args = parser.parse_args()

	config = ExperimentConfig(
		backbone = args.backbone, img_size = args.img_size, dropout_rate = args.dropout_rate,
		dataset_path = 'unused-verify-backbone-head',
	)

	# Backbone — 5.1's checks, unchanged
	backbone = build_backbone( config )
	backbone_trainable_params = sum( tf.keras.backend.count_params( w ) for w in backbone.trainable_weights )
	backbone_total_params = backbone.count_params()

	for layer in backbone.layers:
		log.info( f'{ layer.name }: trainable={ layer.trainable }' )

	dummy_input = np.zeros( ( 1, args.img_size, args.img_size, 3 ), dtype=np.float32 )
	backbone_output = backbone( dummy_input, training=False )
	log.info( f'Backbone forward pass output shape: { tuple( backbone_output.shape ) }' )

	if args.img_size == 299 and args.backbone in EXPECTED_OUTPUT_SHAPE_AT_299:
		expected = EXPECTED_OUTPUT_SHAPE_AT_299[ args.backbone ]

		# Batch dim is dynamic (None) in the documented contract but concrete (1) here
		if tuple( backbone_output.shape )[ 1: ] == expected[ 1: ]:
			log.info( f'Backbone output shape matches documented contract: { expected }' )
		else:
			log.error( f'Backbone output shape MISMATCH — got { tuple( backbone_output.shape ) }, expected { expected }' )

	# ImageNet weight-loading check — 5.1's checks, unchanged (see backbone.py for rationale)
	reference_backbone = BACKBONE_CONSTRUCTORS[ args.backbone ](
		include_top = False,
		weights = None,
		input_shape = ( args.img_size, args.img_size, 3 ),
	)

	weighted_index = next( index for index, layer in enumerate( backbone.layers ) if layer.get_weights() )
	weighted_layer_name = backbone.layers[ weighted_index ].name
	pretrained_kernel = backbone.layers[ weighted_index ].get_weights()[ 0 ]
	random_init_kernel = reference_backbone.layers[ weighted_index ].get_weights()[ 0 ]

	kernel_checksum = hashlib.md5( pretrained_kernel.tobytes() ).hexdigest()
	log.info( f'First weighted layer "{ weighted_layer_name }" kernel checksum={ kernel_checksum }, mean={ pretrained_kernel.mean():.4f}, std={ pretrained_kernel.std():.4f}' )

	if np.array_equal( pretrained_kernel, random_init_kernel ):
		log.error( f'ImageNet weight-loading check FAILED — "{ weighted_layer_name }" kernel is identical to a fresh weights=None baseline; pretrained weights may not have loaded' )
	else:
		log.info( f'ImageNet weight-loading check passed — "{ weighted_layer_name }" kernel differs from a freshly-initialized weights=None baseline' )

	# Head — 5.2, wired onto the backbone's output tensor via the functional API
	output_tensor = build_head( backbone.output, config )
	model = tf.keras.Model( inputs=backbone.input, outputs=output_tensor )

	prediction = model( dummy_input, training=False )
	prediction_value = float( prediction[ 0, 0 ] )
	log.info( f'Full model forward pass output shape: { tuple( prediction.shape ) }, value={ prediction_value:.4f}' )

	if tuple( prediction.shape ) == ( 1, 1 ) and 0.0 <= prediction_value <= 1.0:
		log.info( 'Output shape (1, 1) and sigmoid value-range [0, 1] check passed' )
	else:
		log.error( f'Output shape/range check FAILED — shape={ tuple( prediction.shape ) }, value={ prediction_value }' )

	# Dropout active-under-training — two training=True passes on the same input should
	# differ, since Dropout draws a fresh random mask on every call when training=True.
	# Frozen (trainable=False) BatchNorm layers in the backbone always run in inference
	# mode regardless of the outer training flag, so this isolates dropout's own randomness.
	#
	# Side effect worth knowing about when reading the log below: the *trainable*
	# block13/14 BatchNorm layers DO run in training mode here, which updates their
	# moving_mean/moving_variance in place — same as a real model.fit() batch would. That
	# permanently shifts this model instance's normalization statistics, so the upcoming
	# "inactive-at-inference" value will not match the earlier "Full model forward pass"
	# value above (both are training=False, but one ran before this BN update, one after).
	# That's expected, not a bug — the inactive-at-inference check only asserts the two
	# post-update inference passes match each other, which they do.
	train_pass_1 = model( dummy_input, training=True ).numpy()
	train_pass_2 = model( dummy_input, training=True ).numpy()

	if np.array_equal( train_pass_1, train_pass_2 ):
		log.error( 'Dropout active-under-training check FAILED — two training=True forward passes produced identical output' )
	else:
		log.info( f'Dropout active-under-training check passed — training=True outputs differ ({ float( train_pass_1[ 0, 0 ] ):.4f} vs { float( train_pass_2[ 0, 0 ] ):.4f})' )

	# Dropout inactive-at-inference — two training=False passes should match exactly,
	# since Dropout is an identity op and BatchNorm uses moving (not batch) statistics
	infer_pass_1 = model( dummy_input, training=False ).numpy()
	infer_pass_2 = model( dummy_input, training=False ).numpy()

	if np.array_equal( infer_pass_1, infer_pass_2 ):
		log.info( f'Dropout inactive-at-inference check passed — training=False outputs match exactly ({ float( infer_pass_1[ 0, 0 ] ):.4f})' )
	else:
		log.error( f'Dropout inactive-at-inference check FAILED — two training=False forward passes produced different output ({ float( infer_pass_1[ 0, 0 ] ):.4f} vs { float( infer_pass_2[ 0, 0 ] ):.4f})' )

	# Param counts — backbone-only baseline (5.1) vs full backbone+head
	full_trainable_params = sum( tf.keras.backend.count_params( w ) for w in model.trainable_weights )
	full_total_params = model.count_params()

	log.info( f'Backbone-only params (5.1 baseline): trainable={ backbone_trainable_params }/{ backbone_total_params }' )
	log.info( f'Head-added params: trainable={ full_trainable_params - backbone_trainable_params }, total={ full_total_params - backbone_total_params }' )
	log.info( f'Full model params: trainable={ full_trainable_params }/{ full_total_params }' )


if __name__ == '__main__':
	main()
