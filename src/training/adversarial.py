# Standard library
import gc
import logging
# TensorFlow
import tensorflow as tf

log = logging.getLogger( __name__ )

# GPU-VRAM mitigation (fine-tuning-phase OOM amendment) — original-batch rows processed
# per GradientTape in compute_fgsm_perturbation, not the full incoming batch at once. Each
# row expands internally to patch_grid_size**2 patches once the caller's TimeDistributed
# backbone runs (e.g. 8 rows * 81 patches = 648 images through XCeption in one tape, which
# a real Vertex AI L4 run OOM'd on: Limit 20.34GiB, InUse 20.20GiB at crash time, inside
# this exact function). The identically-sized forward+backward pass fits fine inside
# model.fit()'s compiled training step (base training completes with no OOM at the same
# batch_size), so the gap is this function's uncompiled, eager GradientTape not getting
# that step's graph-level memory planning/buffer reuse — not a hard "648 patches" ceiling.
# Independent of patch_grid_size — chunking happens on the original-image batch dimension,
# before the caller's TimeDistributed reshape.
FGSM_CHUNK_SIZE = 2


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

	Processed FGSM_CHUNK_SIZE original rows at a time (see module comment) rather than the
	whole batch through one GradientTape. This is exact, not an approximation: FGSM only
	needs sign(gradient), and binary_crossentropy's per-example loss terms don't couple
	across examples, so sign(gradient) computed on a sub-batch is identical to sign(gradient)
	computed within the full batch — reduce_mean's scaling constant is the only thing that
	differs between chunk-local and full-batch averaging, and sign() is invariant to a
	positive scaling constant. Splitting and concatenating the result changes peak memory
	only, never the perturbation actually produced.
	'''

	images = tf.convert_to_tensor( images )
	batch_size = images.shape[ 0 ]
	perturbed_chunks = []

	for start in range( 0, batch_size, FGSM_CHUNK_SIZE ):
		end = min( start + FGSM_CHUNK_SIZE, batch_size )
		images_chunk = images[ start : end ]

		with tf.GradientTape() as tape:
			tape.watch( images_chunk )
			predictions = model( images_chunk, training=False )
			loss = tf.keras.losses.binary_crossentropy( labels[ start : end ], predictions )
			loss = tf.reduce_mean( loss * sample_weight[ start : end ] )

		gradient = tape.gradient( loss, images_chunk )
		perturbation = epsilon * tf.sign( gradient )
		perturbed_chunks.append( tf.clip_by_value( images_chunk + perturbation, 0.0, 1.0 ) )

		del images_chunk, predictions, loss, gradient, tape

	adversarial_images = tf.concat( perturbed_chunks, axis=0 )
	del perturbed_chunks
	gc.collect()

	return adversarial_images


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
