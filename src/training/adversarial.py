# Standard library
import gc
import logging
# TensorFlow
import tensorflow as tf

log = logging.getLogger( __name__ )


def compute_fgsm_perturbation( model: tf.keras.Model, images: tf.Tensor, labels: tf.Tensor, sample_weight: tf.Tensor, epsilon: float ) -> tf.Tensor:
	'''
	Fast Gradient Sign Method (Goodfellow et al., 2015): x_adv = x + epsilon * sign(grad_x
	loss(model(x), y)). Manual tf.GradientTape, not a library shortcut, per the folder
	structure's own note that adversarial.py is GradientTape-based.

	images is the model's actual input tensor — the DFT patch grid (batch, patch_count,
	img_size, img_size, 3), not the raw pixel image. Raw pixels never reach the model
	directly (preprocessor.py's DFT transform sits between them per the patch pipeline),
	so the only input FGSM can meaningfully perturb is the one the model's gradient graph
	actually runs through: this patch tensor.

	training=False for this forward pass — the perturbation should reflect the model's
	genuine, deterministic sensitivity to the input, not noise introduced by dropout.

	Clipped to [0, 1], matching preprocessor.py's own min-max-scaled output range for this
	tensor (not [0, 255] — that range belongs to the raw pixel image, several steps
	upstream of what this function perturbs).
	'''

	images = tf.convert_to_tensor( images )

	with tf.GradientTape() as tape:
		tape.watch( images )
		predictions = model( images, training=False )
		loss = tf.keras.losses.binary_crossentropy( labels, predictions )
		loss = tf.reduce_mean( loss * sample_weight )

	gradient = tape.gradient( loss, images )
	perturbation = epsilon * tf.sign( gradient )

	return tf.clip_by_value( images + perturbation, 0.0, 1.0 )


def build_mixed_batch( model: tf.keras.Model, images: tf.Tensor, labels: tf.Tensor, sample_weight: tf.Tensor, epsilon: float ) -> tf.Tensor:
	'''
	Mix clean and FGSM-perturbed examples *within* one batch (first half clean, second half
	adversarial) rather than alternating whole clean/adversarial batches — the interpretation
	System Architecture Doc 9.5's table specifies for "50% clean / 50% adversarial batches".
	Labels/sample_weight are unchanged and stay aligned to the same example order; only the
	second half of images is replaced with its adversarial counterpart.
	'''

	adversarial_images = compute_fgsm_perturbation( model, images, labels, sample_weight, epsilon )

	half = images.shape[ 0 ] // 2

	return tf.concat( [ images[ :half ], adversarial_images[ half: ] ], axis=0 )


def fine_tune_with_fgsm( model: tf.keras.Model, batches, learning_rate: float, epsilon: float ) -> None:
	'''
	Post-training fine-tuning pass for one ensemble member — one full pass over `batches`
	(whatever the caller supplies; train.py passes one epoch's worth of the same train
	split used for initial training), each step replacing half of that batch's images with
	their FGSM-perturbed counterpart before a normal gradient-descent update. Applied per
	member individually (called once per member, after that member's own initial training
	converges), never on the fused ensemble output — per the already-locked System
	Architecture Doc 9.5 design.

	A fresh Adam optimizer at config.learning_rate — fine-tuning is a distinct phase from
	initial training, not a continuation of the same optimizer state (whose momentum/
	variance accumulators were shaped by clean-only gradients).
	'''

	optimizer = tf.keras.optimizers.Adam( learning_rate=learning_rate )

	for step, ( images, labels, sample_weight ) in enumerate( batches ):
		# Reshaped to (batch, 1) to match the model's own (batch, 1) sigmoid output —
		# model.fit()'s compiled loss path reconciles a bare (batch,) target
		# automatically, but a raw tf.keras.losses.binary_crossentropy() call in a manual
		# GradientTape loop (this module, throughout) does not and raises on the rank
		# mismatch otherwise.
		labels = tf.reshape( tf.convert_to_tensor( labels, dtype=tf.float32 ), [ -1, 1 ] )
		sample_weight = tf.reshape( tf.convert_to_tensor( sample_weight, dtype=tf.float32 ), [ -1, 1 ] )

		mixed_images = build_mixed_batch( model, images, labels, sample_weight, epsilon )

		with tf.GradientTape() as tape:
			predictions = model( mixed_images, training=True )
			loss = tf.keras.losses.binary_crossentropy( labels, predictions )
			loss = tf.reduce_mean( loss * sample_weight )

		gradients = tape.gradient( loss, model.trainable_variables )
		optimizer.apply_gradients( zip( gradients, model.trainable_variables ) )

		log.info( f'Adversarial fine-tuning step { step } — loss={ float( loss ):.4f}' )

		# Explicit cleanup (Task 5.6 memory-mitigation amendment) — same discipline as
		# build_batches()/preprocess_image()'s Tier 1 cleanup, applied here too since this
		# loop hits the identical per-image patch-materialization cost a second time (via
		# `batches`, train.py's train_generator() reused for fine-tuning) on top of
		# base training's own resident state, plus this loop's own per-step tensors
		# (mixed_images, predictions, gradients, the GradientTape's captured graph) that a
		# raw eager GradientTape loop doesn't get the same buffer-reuse Keras's compiled
		# model.fit() path gets for free.
		del images, labels, sample_weight, mixed_images, predictions, gradients, tape
		gc.collect()
