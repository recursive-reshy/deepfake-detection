# Standard library
import argparse
import logging
# Config
import config
# Data
from src.data.loader import load_manifest
from src.data.splitter import split_dataset

logging.basicConfig( level=logging.INFO )
log = logging.getLogger( __name__ )


def main() -> None:

	parser = argparse.ArgumentParser()
	parser.add_argument( '--max-samples-per-split', type=int, default=None )
	args = parser.parse_args()

	manifest_uri = f'gs://{ config.GCS_BUCKET }/{ config.GCS_DATASET_PATH }'
	manifest = load_manifest( manifest_uri, args.max_samples_per_split )
	splits, class_weights = split_dataset( manifest, is_truncated = args.max_samples_per_split is not None )

	for split_name, split_df in splits.items():
		total = len( split_df )
		counts = split_df[ 'label' ].value_counts()

		log.info( f'{ split_name }: total={ total }' )

		for label, count in counts.items():
			log.info( f'{ split_name } — { label }: { count } ({ 100 * count / total:.1f}%)' )

	log.info( f'Class weights (train partition, inverse frequency): { class_weights }' )


if __name__ == '__main__':
	main()
