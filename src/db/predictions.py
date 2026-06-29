# Schemas
from src.schemas.db import PredictionRecord
# Db
from src.db.client import get_client

def log_prediction( record: PredictionRecord ) -> None:
	client = get_client()
	predictions_ref = client.collection( 'predictions' )

	predictions_ref.add( record.model_dump() )
