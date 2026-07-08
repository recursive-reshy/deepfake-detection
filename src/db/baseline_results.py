# Schemas
from src.schemas.db import BaselineResultsDocument
# Db
from src.db.client import get_client

BASELINE_RESULTS_COLLECTION = 'baseline_results'

def write_baseline_results( results: BaselineResultsDocument ) -> None:
	client = get_client()
	results_ref = client.collection( BASELINE_RESULTS_COLLECTION ).document( results.run_id )

	results_ref.set( results.model_dump() )

def get_baseline_results( run_id: str ) -> BaselineResultsDocument | None:
	client = get_client()
	results_ref = client.collection( BASELINE_RESULTS_COLLECTION ).document( run_id )
	snapshot = results_ref.get()

	if not snapshot.exists:
		return None

	return BaselineResultsDocument.model_validate( snapshot.to_dict() )
