# Schemas
from src.schemas.db import EpochRecord
# Db
from src.db.client import get_client

def write_epoch( job_id: str, record: EpochRecord ) -> None:
	client = get_client()
	epochs_ref = client.collection( 'jobs' ).document( job_id ).collection( 'epochs' )

	epochs_ref.add( record.model_dump() )
