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
