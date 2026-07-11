# Standard library
from typing import Literal
# Pydantic
from pydantic import BaseModel
from pydantic import Field

class ExperimentConfig( BaseModel ):
	backbone: Literal[ 'xception', 'efficientnetb0', 'resnet50', 'mobilenetv2' ] = 'xception'
	ensemble_size: int = Field( default = 3, ge = 1, le = 5 )

	use_patch_pipeline: bool = True
	patch_grid_size: int = 9
	img_size: int = 299

	batch_size: int = 8
	epochs: int = 30
	learning_rate: float = Field( default = 0.0001, gt = 0 )
	dropout_rate: float = Field( default = 0.5, ge = 0, le = 1 )
	early_stopping_patience: int = 5
	random_seed: int = 42

	adversarial_training: bool = False
	adversarial_epsilon: float = Field( default = 0.01, gt = 0 )

	dataset_ingestion: Literal[ 'presplit', 'flat', 'csv' ] = 'presplit'
	dataset_path: str

	# Smoke-test only — truncates each split to a small, class-balanced sample so Phase 5
	# code can be validated on CPU ahead of T4 quota. Must be None in every production config.
	max_samples_per_split: int | None = None
