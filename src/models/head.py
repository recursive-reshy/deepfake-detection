# Standard library
import logging
# TensorFlow
import tensorflow as tf
# Schemas
from src.schemas.experiment import ExperimentConfig

log = logging.getLogger( __name__ )


def build_head( feature_tensor: tf.Tensor, config: ExperimentConfig ) -> tf.Tensor:
	'''
	Attach the fixed classification head (System Architecture Doc 9.2) to a backbone's
	output feature tensor: GlobalAveragePooling2D -> Dense(512, relu) -> Dropout(rate) ->
	Dense(1, sigmoid). Same architecture regardless of backbone choice — only
	config.dropout_rate varies it, read from config rather than hardcoded so the field
	on ExperimentConfig is actually tunable.

	Takes and returns a tensor, not a Model, deliberately — composes with backbone.py's
	build_backbone() output via the Keras functional API
	(tf.keras.Model(inputs=backbone.input, outputs=build_head(backbone.output, config)))
	rather than assuming backbone and head are only ever built together in one script.
	5.3's ensemble.py needs 3 independent backbone+head instances, each wired this way.

	No file I/O, no GCS/Firestore calls — pure model construction, same as backbone.py.
	'''

	x = tf.keras.layers.GlobalAveragePooling2D()( feature_tensor )
	x = tf.keras.layers.Dense( 512, activation='relu' )( x )
	x = tf.keras.layers.Dropout( config.dropout_rate )( x )
	x = tf.keras.layers.Dense( 1, activation='sigmoid' )( x )

	return x
