# Standard library
import logging
# Structured logging
from pythonjsonlogger.json import JsonFormatter
# Models
from src.models.baseline import load_baseline_artifacts
from src.models.baseline import load_cached_features
from src.models.baseline import evaluate_baseline_svm

handler = logging.StreamHandler()
handler.setFormatter( JsonFormatter(
	'%(levelname)s %(name)s %(message)s',
	rename_fields = { 'levelname': 'severity' },
	timestamp = True,
) )
logging.basicConfig( level=logging.INFO, handlers=[ handler ], force=True )

log = logging.getLogger( __name__ )


def main() -> None:

	model, scaler = load_baseline_artifacts()

	# Val is loaded and its shape logged purely as a build-time sanity check — confirms
	# build_feature_matrices() (4.2) produced consistent dimensionality across every
	# split, not just train. Not evaluated against here; that stays test-only.
	load_cached_features( 'val' )
	X_test, y_test = load_cached_features( 'test' )

	evaluate_baseline_svm( model, scaler, X_test, y_test )


if __name__ == '__main__':
	main()
