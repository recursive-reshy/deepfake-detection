# Standard library
import logging
from datetime import datetime
from datetime import timezone
# Structured logging
from pythonjsonlogger.json import JsonFormatter
# Config
import config
# Data
from src.data.loader import load_manifest
from src.data.splitter import split_dataset
# Models
from src.models.baseline import build_feature_matrices
from src.models.baseline import train_baseline_svm
# Utils
from src.utils.seed import set_global_seed

handler = logging.StreamHandler()
handler.setFormatter( JsonFormatter(
	'%(levelname)s %(name)s %(message)s',
	rename_fields = { 'levelname': 'severity' },
	timestamp = True,
) )
logging.basicConfig( level=logging.INFO, handlers=[ handler ], force=True )

log = logging.getLogger( __name__ )

# Not sourced from ExperimentConfig — this script has no Firestore job behind it, per
# the async job pattern (baseline runs are local/Cloud Shell only, never Vertex AI).
# Matches ExperimentConfig's own defaults for img_size and random_seed so the baseline
# stays comparable to the ensemble's default configuration.
IMG_SIZE = 299
RANDOM_SEED = 42


def main() -> None:

	# Baseline runs have no Vertex AI job_id — this is the lightweight identifier 4.4
	# will use to tag the Firestore results document. Not persisted here; log only.
	run_id = f'baseline-svm-{ datetime.now( timezone.utc ).strftime( "%Y%m%d-%H%M%S" ) }'
	log.info( f'Baseline run: { run_id }' )

	set_global_seed( RANDOM_SEED )

	# Manifest + split — same loader.py / splitter.py used by the ensemble pipeline
	manifest_uri = f'gs://{ config.GCS_BUCKET }/{ config.GCS_DATASET_PATH }'
	manifest = load_manifest( manifest_uri )
	splits, class_weights = split_dataset( manifest )

	# Feature extraction (cached to GCS) + SVM training (model + scaler persisted to GCS)
	features = build_feature_matrices( splits, IMG_SIZE )
	model, scaler = train_baseline_svm( features, class_weights, RANDOM_SEED )

	log.info( f'Baseline run { run_id } complete — model classes_: { list( model.classes_ ) }' )


if __name__ == '__main__':
	main()
