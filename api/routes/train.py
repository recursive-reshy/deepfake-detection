# Standard library
import logging
from datetime import datetime
from datetime import timezone
# YAML
import yaml
# FastAPI
from fastapi import APIRouter
from fastapi.responses import JSONResponse
# Vertex AI
from google.cloud import aiplatform
# Config
import config
# Schemas
from src.schemas.experiment import ExperimentConfig
from src.schemas.db import JobDocument
# DB
from src.db import jobs

router = APIRouter()

log = logging.getLogger( __name__ )


@router.post( '/train' )
async def submit_training_job( payload: ExperimentConfig ):

	now = datetime.now( timezone.utc )
	job_id = f'train_{ now.strftime( "%Y%m%d_%H%M%S" ) }'

	# Write PENDING job to Firestore
	job_doc = JobDocument(
		job_id = job_id,
		stage = 'train',
		status = 'PENDING',
		config = payload,
		created_at = now,
		updated_at = now,
	)
	jobs.create_job( job_doc )

	# Load and substitute YAML template
	with open( 'infra/vertex_job.yaml' ) as f:
		spec_text = f.read()

	spec_text = spec_text \
		.replace( '{job_id}', job_id ) \
		.replace( '{image_uri}', config.IMAGE_URI or '' )

	spec = yaml.safe_load( spec_text )

	# Build worker pool specs in snake_case (Vertex AI SDK format)
	raw_pools = spec[ 'jobSpec' ][ 'workerPoolSpecs' ]

	worker_pool_specs = [
		{
			'machine_spec': {
				'machine_type': pool[ 'machineSpec' ][ 'machineType' ],
				**(
					{
						'accelerator_type': pool[ 'machineSpec' ][ 'acceleratorType' ],
						'accelerator_count': pool[ 'machineSpec' ][ 'acceleratorCount' ],
					}
					if 'acceleratorType' in pool[ 'machineSpec' ] else {}
				),
			},
			'replica_count':  pool[ 'replicaCount' ],
			'container_spec': {
				'image_uri': pool[ 'containerSpec' ][ 'imageUri' ],
				'command': pool[ 'containerSpec' ][ 'command' ],
				'args': pool[ 'containerSpec' ][ 'args' ],
				'env': [
					{ 'name': e[ 'name' ], 'value': e[ 'value' ] }
					for e in pool[ 'containerSpec' ].get( 'env', [] )
				],
			},
		}
		for pool in raw_pools
	]

	# Submit to Vertex AI
	try:
		aiplatform.init( project=config.GCP_PROJECT_ID, location=config.GCP_REGION )

		custom_job = aiplatform.CustomJob(
			display_name = spec[ 'displayName' ],
			worker_pool_specs = worker_pool_specs,
			staging_bucket = f"gs://{ config.GCS_BUCKET }"
		)
		custom_job.submit()

	except Exception as exc:
		jobs.update_job_error( job_id, str( exc ) )
		jobs.update_job_status( job_id, 'FAILED' )
		return JSONResponse( status_code=500, content={ 'detail': str( exc ) } )

	return JSONResponse( status_code=202, content={ 'job_id': job_id, 'status': 'PENDING' } )
