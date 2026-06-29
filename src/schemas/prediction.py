# Standard library
from typing import Literal
# Pydantic
from pydantic import BaseModel

class PatchScore( BaseModel ):
	patch_index: int
	score: float

class EnsemblePrediction( BaseModel ):
	image_hash: str
	patch_scores: list[ PatchScore ]
	member_predictions: list[ float ]

	final_score: float
	label: Literal[ 'real', 'fake' ]
	confidence: float
