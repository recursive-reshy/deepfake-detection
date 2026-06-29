# Schemas
from src.schemas.db import ResultsSummary
# Db
from src.db.client import get_client

RESULTS_DOCUMENT_ID = 'summary'

def write_results( job_id: str, summary: ResultsSummary ) -> None:
	client = get_client()
	results_ref = client.collection( 'jobs' ).document( job_id ).collection( 'results' ).document( RESULTS_DOCUMENT_ID )

	results_ref.set( summary.model_dump() )

def get_results( job_id: str ) -> ResultsSummary | None:
	client = get_client()
	results_ref = client.collection( 'jobs' ).document( job_id ).collection( 'results' ).document( RESULTS_DOCUMENT_ID )
	snapshot = results_ref.get()

	if not snapshot.exists:
		return None

	return ResultsSummary.model_validate( snapshot.to_dict() )
