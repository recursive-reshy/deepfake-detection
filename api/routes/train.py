# Standard library
import logging
from datetime import datetime
from datetime import timezone
# FastAPI
from fastapi import APIRouter
from fastapi.responses import JSONResponse
# Schemas
from src.schemas.experiment import ExperimentConfig
from src.schemas.db import JobDocument
# DB
from src.db import jobs
# Utils
from src.utils.vertex import submit_vertex_job

router = APIRouter()

log = logging.getLogger( __name__ )


@router.post( '/train' )
async def submit_training_job( payload: ExperimentConfig ):

	now = datetime.now( timezone.utc )
	job_id = f'train_{ now.strftime( "%Y%m%d_%H%M%S" ) }'

	# Write PENDING job to Firestore — every submission starts at Stage 1 (base training);
	# train.py's own Stage 1 -> Stage 2 hand-off (see run_base_stage) is what moves
	# training_stage on from here, never this route.
	job_doc = JobDocument(
		job_id = job_id,
		stage = 'train',
		status = 'PENDING',
		training_stage = 'base_training',
		config = payload,
		created_at = now,
		updated_at = now,
	)
	jobs.create_job( job_doc )

	# Submit Stage 1 (base training) to Vertex AI
	try:
		vertex_job_id = submit_vertex_job( job_id, 'base' )
		jobs.update_job_vertex_job_id( job_id, 'base', vertex_job_id )

	except Exception as exc:
		jobs.update_job_error( job_id, str( exc ) )
		jobs.update_job_status( job_id, 'FAILED' )
		return JSONResponse( status_code=500, content={ 'detail': str( exc ) } )

	return JSONResponse( status_code=202, content={ 'job_id': job_id, 'status': 'PENDING' } )
