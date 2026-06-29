# Standard library
from typing import Literal
# Pydantic
from pydantic import BaseModel

class DatasetManifest( BaseModel ):
	train_path: str
	val_path: str
	test_path: str

	num_classes: int = 2
	class_names: list[ str ]
	class_weights: dict[ str, float ]

	total_samples: int
	ingestion_mode: Literal[ 'presplit', 'flat', 'csv' ]
