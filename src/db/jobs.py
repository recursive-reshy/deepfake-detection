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

def update_job_progress( job_id: str, current_epoch: int ) -> None:
	client = get_client()
	job_ref = client.collection( JOBS_COLLECTION ).document( job_id )

	job_ref.update( {
		'current_epoch': current_epoch,
		'updated_at': datetime.now( timezone.utc ),
	} )

def update_job_training_stage( job_id: str, training_stage: str ) -> None:
	client = get_client()
	job_ref = client.collection( JOBS_COLLECTION ).document( job_id )

	job_ref.update( {
		'training_stage': training_stage,
		'updated_at': datetime.now( timezone.utc ),
	} )

def update_job_member_checkpoint( job_id: str, member_index: int, checkpoint_uri: str ) -> None:
	client = get_client()
	job_ref = client.collection( JOBS_COLLECTION ).document( job_id )

	# Dotted field path — updates one key of the member_checkpoints map without touching
	# the others, rather than overwriting the whole map (other members may still be
	# training when this write happens).
	job_ref.update( {
		f'member_checkpoints.{ member_index }': checkpoint_uri,
		'updated_at': datetime.now( timezone.utc ),
	} )

def update_job_member_adversarial_checkpoint( job_id: str, member_index: int, checkpoint_uri: str ) -> None:
	client = get_client()
	job_ref = client.collection( JOBS_COLLECTION ).document( job_id )

	job_ref.update( {
		f'member_adversarial_checkpoints.{ member_index }': checkpoint_uri,
		'updated_at': datetime.now( timezone.utc ),
	} )

def update_job_vertex_job_id( job_id: str, stage: str, vertex_job_id: str ) -> None:
	client = get_client()
	job_ref = client.collection( JOBS_COLLECTION ).document( job_id )
	field = f'vertex_job_id_{ stage }'

	job_ref.update( {
		field: vertex_job_id,
		'updated_at': datetime.now( timezone.utc ),
	} )
