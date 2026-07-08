# Standard library
import argparse
import logging
from datetime import datetime
from datetime import timezone
# Structured logging
from pythonjsonlogger.json import JsonFormatter
# Models
from src.models.baseline import load_baseline_artifacts
from src.models.baseline import load_cached_features
from src.models.baseline import evaluate_baseline_svm
from src.models.baseline import get_artifact_uris
# Schemas
from src.schemas.db import BaselineResultsDocument
# Db
from src.db.baseline_results import write_baseline_results

handler = logging.StreamHandler()
handler.setFormatter( JsonFormatter(
	'%(levelname)s %(name)s %(message)s',
	rename_fields = { 'levelname': 'severity' },
	timestamp = True,
) )
logging.basicConfig( level=logging.INFO, handlers=[ handler ], force=True )

log = logging.getLogger( __name__ )

MODEL_TYPE = 'svm_dft_baseline'
FEATURE_METHOD = 'azimuthal_power_spectrum'


def main() -> None:

	parser = argparse.ArgumentParser()
	parser.add_argument( '--run-id', required=True, help='Run identifier logged by train_baseline.py (e.g. baseline-svm-20260709-091500)' )
	args = parser.parse_args()

	run_id = args.run_id

	model, scaler = load_baseline_artifacts()

	# Val is loaded and its shape logged purely as a build-time sanity check — confirms
	# build_feature_matrices() (4.2) produced consistent dimensionality across every
	# split, not just train. Not evaluated against here; that stays test-only.
	load_cached_features( 'val' )
	X_test, y_test = load_cached_features( 'test' )

	results = evaluate_baseline_svm( model, scaler, X_test, y_test )

	# Write once, after everything is already known — no partial-write/poll pattern like
	# the Vertex AI job documents use, since a baseline run produces its final result in
	# one shot rather than progressing through epochs.
	document = BaselineResultsDocument(
		run_id = run_id,
		model_type = MODEL_TYPE,
		accuracy = results[ 'accuracy' ],
		auc_roc = results[ 'auc_roc' ],
		auc_roc_positive_class = results[ 'auc_roc_positive_class' ],
		feature_dim = X_test.shape[ 1 ],
		feature_method = FEATURE_METHOD,
		artifact_paths = get_artifact_uris(),
		completed_at = datetime.now( timezone.utc ),
	)
	write_baseline_results( document )

	log.info( f'Baseline run { run_id } results persisted to Firestore ("baseline_results" collection)' )


if __name__ == '__main__':
	main()
