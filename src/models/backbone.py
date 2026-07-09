# Standard library
import logging
# TensorFlow
import tensorflow as tf
# Schemas
from src.schemas.experiment import ExperimentConfig

log = logging.getLogger( __name__ )

BACKBONE_CONSTRUCTORS = {
	'xception': tf.keras.applications.Xception,
	'efficientnetb0': tf.keras.applications.EfficientNetB0,
	'resnet50': tf.keras.applications.ResNet50,
	'mobilenetv2': tf.keras.applications.MobileNetV2,
}

# Output feature-map shape contract at img_size=299 (include_top=False, no pooling) — the
# expected-output reference verify_backbone.py's exit criteria checks against. Channel
# depth is architecture-specific; spatial dims land on 10x10 for all four at this
# resolution.
EXPECTED_OUTPUT_SHAPE_AT_299 = {
	'xception': ( None, 10, 10, 2048 ),
	'efficientnetb0': ( None, 10, 10, 1280 ),
	'resnet50': ( None, 10, 10, 2048 ),
	'mobilenetv2': ( None, 10, 10, 1280 ),
}

# Xception-only: name prefix of the first layer belonging to the exit flow's last 2
# blocks (block13, block14) — the freezing boundary. Applied by layer *index*, not by a
# name-prefix filter over every layer, because Xception's functional graph threads
# block13's residual-shortcut conv/BN/add layers (generically named conv2d_N,
# batch_normalization_N, add_N) between the named block13_* layers. Filtering by name
# alone would leave that shortcut path frozen while its own block13 main path unfreezes —
# an inconsistent split of one residual block. Finding the index of the first
# 'block13'-named layer and unfreezing everything from there onward captures the shortcut
# layers too, since they sit after that index in the layer list.
#
# This name-matching rule is deliberately Xception-specific, not a pattern intended to
# generalize — 'block13'/'block14' are Xception's own layer-naming convention and have no
# equivalent in the other 3 backbones' naming schemes (e.g. EfficientNet's 'blockNa',
# ResNet50's 'convN_blockN', MobileNetV2's 'block_N'). Those 3 instead take the
# freeze-everything fallback below.
#
# Test coverage note: as of Task 5.1, verify_backbone.py has only been run against
# 'xception' (this boundary) and 'efficientnetb0' (fallback). 'resnet50' and
# 'mobilenetv2' share the same fallback code path but have not individually been run —
# deferred, not assumed safe, since this project only exercises 'xception' in practice.
# Run `python -m src.models.verify_backbone --backbone=resnet50` /
# `--backbone=mobilenetv2` before relying on either for real training.
XCEPTION_UNFREEZE_BLOCK_PREFIX = 'block13'


def build_backbone( config: ExperimentConfig ) -> tf.keras.Model:
	'''
	Build the ImageNet-pretrained feature-extraction backbone named by config.backbone, at
	config.img_size resolution, with a single-phase (no progressive unfreeze) freezing
	scheme applied. include_top=False throughout, so the returned model's output is the
	raw backbone feature map — no pooling or dense layers, that is head.py's job (5.2).
	Expected output shape at the default img_size=299 is EXPECTED_OUTPUT_SHAPE_AT_299
	above, e.g. (None, 10, 10, 2048) for 'xception'.

	Freezing: for 'xception', every layer is frozen except the last 2 exit-flow blocks
	(block13, block14, plus their shortcut layers — see XCEPTION_UNFREEZE_BLOCK_PREFIX
	above) — matches the Tracker's Phase 5 risk mitigation and needs no unfreeze-epoch
	field on ExperimentConfig. The other 3 backbones in the menu (efficientnetb0,
	resnet50, mobilenetv2) use unrelated block-naming conventions with no equivalent
	"last 2 blocks" boundary defined for this project, so they fall back to freezing the
	entire base model rather than guessing a per-architecture equivalent.

	No file I/O, no GCS/Firestore calls — pure model construction. Per-layer freeze
	inspection and forward-pass verification live in verify_backbone.py, not here, since
	logging every layer on every real ensemble-member build would spam Cloud Logging.
	'''

	constructor = BACKBONE_CONSTRUCTORS[ config.backbone ]

	base_model = constructor(
		include_top = False,
		weights = 'imagenet',
		input_shape = ( config.img_size, config.img_size, 3 ),
	)

	# Freezing
	if config.backbone == 'xception':
		boundary_index = next(
			index for index, layer in enumerate( base_model.layers )
			if layer.name.startswith( XCEPTION_UNFREEZE_BLOCK_PREFIX )
		)

		for index, layer in enumerate( base_model.layers ):
			layer.trainable = index >= boundary_index
	else:
		for layer in base_model.layers:
			layer.trainable = False

	trainable_params = sum( tf.keras.backend.count_params( w ) for w in base_model.trainable_weights )
	total_params = base_model.count_params()

	log.info( f'Backbone "{ config.backbone }" built — output_shape={ base_model.output_shape }, trainable_params={ trainable_params }/{ total_params }' )

	return base_model
