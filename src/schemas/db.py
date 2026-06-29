# Standard library
from datetime import datetime
from typing import Literal
# Pydantic
from pydantic import BaseModel
# Schemas
from src.schemas.experiment import ExperimentConfig

class JobDocument( BaseModel ):
	job_id: str
	stage: Literal[ 'train', 'evaluate', 'predict' ]
	status: Literal[ 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED' ]

	config: ExperimentConfig

	created_at: datetime
	updated_at: datetime
	error: str | None = None

class EpochRecord( BaseModel ):
	ensemble_member: int
	epoch: int
	loss: float
	val_loss: float
	accuracy: float
	val_accuracy: float
	recorded_at: datetime

class ResultsSummary( BaseModel ):
	accuracy: float
	precision: float
	recall: float
	f1: float
	auc_roc: float
	checkpoint_path: str
	plots: dict[ str, str ]
	completed_at: datetime

class PredictionRecord( BaseModel ):
	job_id: str
	image_hash: str
	prediction: float
	label: Literal[ 'real', 'fake' ]
	confidence: float
	predicted_at: datetime
