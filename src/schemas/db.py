# Standard library
from datetime import datetime
from typing import Literal
# Pydantic
from pydantic import BaseModel
from pydantic import Field
# Schemas
from src.schemas.experiment import ExperimentConfig

class JobDocument( BaseModel ):
	job_id: str
	# Pipeline category (train/evaluate/predict) — distinct from training_stage below,
	# which only exists for stage == 'train' and tracks progress *within* a training job's
	# own base/adversarial split.
	stage: Literal[ 'train', 'evaluate', 'predict' ]
	status: Literal[ 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'STAGE2_SUBMISSION_FAILED' ]

	config: ExperimentConfig

	created_at: datetime
	updated_at: datetime
	error: str | None = None

	# Bumped once per completed epoch by training/callbacks.py's FirestoreEpochCallback —
	# lets a client poll job state for live progress instead of only the epochs
	# subcollection, without waiting for the job to reach a terminal status.
	current_epoch: int | None = None

	# Two-stage training split (base training -> adversarial fine-tuning), each stage a
	# separate Vertex AI job/container — see train.py. None for non-training jobs (stage !=
	# 'train') and for job documents written before this field existed, so old Firestore
	# documents still validate.
	training_stage: Literal[ 'base_training', 'adversarial_finetuning', 'complete' ] | None = None

	# Stage 1 -> Stage 2 hand-off surface: per-member checkpoint GCS URIs, keyed by
	# str(member_index) (Firestore map keys must be strings). Stage 2 runs in a separate
	# process from Stage 1, so it can't rely on Stage 1's in-memory state — it reads these
	# from the job document at startup instead, per the "Firestore as sole job state"
	# principle already locked for epoch tracking.
	member_checkpoints: dict[ str, str ] = Field( default_factory=dict )
	member_adversarial_checkpoints: dict[ str, str ] = Field( default_factory=dict )

	# The underlying Vertex AI CustomJob resource name for each stage — for cross-
	# referencing this Firestore job_id (which never changes across the split) against the
	# two separate Vertex AI job submissions when debugging. Not read by any application
	# code, debugging aid only.
	vertex_job_id_base: str | None = None
	vertex_job_id_adversarial: str | None = None

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

class BaselineResultsDocument( BaseModel ):
	run_id: str
	model_type: str

	accuracy: float
	auc_roc: float
	auc_roc_positive_class: Literal[ 'REAL', 'FAKE' ]

	feature_dim: int
	feature_method: str

	artifact_paths: dict[ str, str ]
	completed_at: datetime
