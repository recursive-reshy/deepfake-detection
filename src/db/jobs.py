# Standard library
from datetime import datetime
from datetime import timezone
# Schemas
from src.schemas.db import JobDocument
# Db
from src.db.client import get_client

JOBS_COLLECTION = 'jobs'

def create_job( job_doc: JobDocument ) -> None:
	client = get_client()
	job_ref = client.collection( JOBS_COLLECTION ).document( job_doc.job_id )

	job_ref.set( job_doc.model_dump() )

def get_job( job_id: str ) -> JobDocument | None:
	client = get_client()
	job_ref = client.collection( JOBS_COLLECTION ).document( job_id )
	snapshot = job_ref.get()

	if not snapshot.exists:
		return None

	return JobDocument.model_validate( snapshot.to_dict() )

def update_job_status( job_id: str, status: str ) -> None:
	client = get_client()
	job_ref = client.collection( JOBS_COLLECTION ).document( job_id )

	job_ref.update( {
		'status': status,
		'updated_at': datetime.now( timezone.utc ),
	} )

def update_job_error( job_id: str, error: str ) -> None:
	client = get_client()
	job_ref = client.collection( JOBS_COLLECTION ).document( job_id )

	job_ref.update( {
		'error': error,
		'updated_at': datetime.now( timezone.utc ),
	} )
