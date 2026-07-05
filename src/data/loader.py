# Standard library
import io
import logging
# Pandas
import pandas as pd
# Utils
from src.utils.gcs import download_blob_to_bytes

log = logging.getLogger( __name__ )

REQUIRED_COLUMNS = [ 'image_path', 'label', 'dataset_split' ]
VALID_SPLITS = { 'train', 'val', 'test' }
EXPECTED_ROW_COUNT = 6557


def load_manifest( dataset_path: str ) -> pd.DataFrame:

	# Fetch manifest CSV from GCS
	csv_bytes = download_blob_to_bytes( dataset_path )
	df = pd.read_csv( io.BytesIO( csv_bytes ) )

	# Validate shape
	missing_columns = [ column for column in REQUIRED_COLUMNS if column not in df.columns ]

	if missing_columns:
		raise ValueError( f'Manifest is missing required columns: { missing_columns }' )

	df = df[ REQUIRED_COLUMNS ]

	invalid_splits = set( df[ 'dataset_split' ].unique() ) - VALID_SPLITS

	if invalid_splits:
		raise ValueError( f'Manifest contains invalid dataset_split values: { invalid_splits }' )

	if len( df ) != EXPECTED_ROW_COUNT:
		log.warning( f'Manifest row count { len( df ) } does not match expected { EXPECTED_ROW_COUNT }' )

	log.info( f'Loaded manifest with { len( df ) } rows from { dataset_path }' )

	return df
