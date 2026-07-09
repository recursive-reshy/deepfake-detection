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

# Fixed seed for smoke-test sampling only — kept separate from ExperimentConfig.random_seed,
# which governs TF/ensemble reproducibility, not this one-off truncation.
SMOKE_TEST_SAMPLE_SEED = 42


def load_manifest( dataset_path: str, max_samples_per_split: int | None = None ) -> pd.DataFrame:

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

	# Smoke-test truncation — stratified per (dataset_split, label), sampling each label in
	# proportion to its actual share of that split rather than forcing an even 50/50 split.
	# An even split would hide a broken inverse-frequency class_weights formula (splitter.py)
	# behind a coincidental {'FAKE': 1.0, 'REAL': 1.0} that a real ~58/42 split would never
	# produce — matching real class balance is what makes the smoke test actually exercise
	# that code path. Each label's quota is rounded from its real ratio and floored at 1, so
	# a small max_samples_per_split still guarantees both classes are represented (the
	# original class-balance risk this truncation exists to avoid). None (production
	# default) leaves the manifest above untouched.
	if max_samples_per_split is not None:
		sampled_groups = []

		for _, split_df in df.groupby( 'dataset_split' ):
			split_total = len( split_df )

			for label, label_count in split_df[ 'label' ].value_counts().items():
				quota = max( 1, round( max_samples_per_split * label_count / split_total ) )
				group = split_df[ split_df[ 'label' ] == label ]
				sampled_groups.append( group.sample( n=min( len( group ), quota ), random_state=SMOKE_TEST_SAMPLE_SEED ) )

		df = pd.concat( sampled_groups, ignore_index=True )

		log.info( f'Truncated manifest to { len( df ) } rows for smoke test (max_samples_per_split={ max_samples_per_split }), class ratio preserved' )

	return df
