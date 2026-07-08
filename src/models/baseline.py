# Standard library
import io
import logging
# NumPy
import numpy as np
# Pandas
import pandas as pd
# scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
# joblib
import joblib
# Config
import config
# Data
from src.data.image_loader import load_image
from src.data.dft import extract_azimuthal_power_spectrum
# Utils
from src.utils.gcs import upload_bytes_to_blob
from src.utils.gcs import download_blob_to_bytes

log = logging.getLogger( __name__ )

FEATURE_CACHE_PREFIX = 'baseline/features'
ARTIFACT_PREFIX = 'baseline/artifacts'


def get_artifact_uris() -> dict[ str, str ]:
	'''
	Single source of truth for where the trained model and fitted scaler live in GCS —
	used by train_baseline_svm() (write), load_baseline_artifacts() (read), and 4.4's
	Firestore document (reported as artifact_paths), so the path convention only exists
	in one place.
	'''

	return {
		'model': f'gs://{ config.GCS_BUCKET }/{ ARTIFACT_PREFIX }/svm_model.joblib',
		'scaler': f'gs://{ config.GCS_BUCKET }/{ ARTIFACT_PREFIX }/scaler.joblib',
	}


def build_feature_matrices( splits: dict[ str, pd.DataFrame ], img_size: int ) -> dict[ str, tuple[ np.ndarray, np.ndarray ] ]:
	'''
	Load every image in each split (train/val/test) via image_loader.py, run it through
	dft.py's azimuthal power spectrum extraction, and assemble per-split feature/label
	matrices. Runs once here rather than per-task — 4.3 (evaluation) reads the cached
	.npy files this writes to GCS instead of recomputing them.

	No augmentation — matches dft.py's contract of clean images only. Labels are kept as
	the raw manifest strings ('REAL' / 'FAKE'), unconverted, so class_weights from
	splitter.py (keyed by those same strings) can be passed straight into SVC without
	remapping.

	Cached to gs://{ GCS_BUCKET }/baseline/features/{ split }_X.npy and _y.npy — read
	back directly with np.load(io.BytesIO(download_blob_to_bytes(uri))) by any
	downstream step that needs them, no dedicated reader in this module.
	'''

	features = {}

	for split_name, split_df in splits.items():
		vectors = [
			extract_azimuthal_power_spectrum( load_image( row[ 'image_path' ], img_size ) )
			for _, row in split_df.iterrows()
		]

		X = np.stack( vectors, axis=0 ).astype( np.float32 )
		y = split_df[ 'label' ].to_numpy().astype( str )

		features[ split_name ] = ( X, y )

		log.info( 'Testing' )

		for name, array in ( ( 'X', X ), ( 'y', y ) ):
			buffer = io.BytesIO()
			np.save( buffer, array )
			upload_bytes_to_blob( f'gs://{ config.GCS_BUCKET }/{ FEATURE_CACHE_PREFIX }/{ split_name }_{ name }.npy', buffer.getvalue() )

		log.info( f'Feature matrix built and cached for split "{ split_name }": { X.shape[ 0 ] } samples, { X.shape[ 1 ] }-dim features' )

	return features


def train_baseline_svm( features: dict[ str, tuple[ np.ndarray, np.ndarray ] ], class_weights: dict[ str, float ], random_seed: int ) -> tuple[ SVC, StandardScaler ]:
	'''
	Fit a StandardScaler on the train split's features only, then train an RBF-kernel SVC
	on the scaled train features. Kernel and hyperparameters are fixed architect
	decisions, not paper-mandated — see report scope note: sklearn defaults (C=1.0,
	gamma='scale'), no grid search, since this baseline's job is "a number to beat", not
	a tuned competitor to the ensemble.

	probability=False, deliberately — SVC's probability calibration runs its own internal
	5-fold CV with independent randomness that set_global_seed (seed.py) cannot reach,
	which would silently break determinism. AUC-ROC (4.3) uses decision_function() output
	directly instead of predict_proba(); roc_auc_score accepts raw decision scores.
	model.classes_ gives the class order decision_function's sign is relative to.

	class_weights is passed straight through from splitter.py — already a { label:
	weight } dict keyed by the same raw label strings used here, no transformation.

	Trained model and fitted scaler are persisted to GCS via joblib (in-memory buffer,
	no local disk) at gs://{ GCS_BUCKET }/baseline/artifacts/svm_model.joblib and
	scaler.joblib — 4.3 and 4.4 both need the scaler applied consistently, and 4.4 will
	need to serve the model too.
	'''

	X_train, y_train = features[ 'train' ]

	# Feature scaling — fit on train only, dft.py's output is raw and unscaled
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform( X_train )

	# RBF-kernel SVM — see docstring for why probability=False and defaults
	model = SVC(
		kernel = 'rbf',
		class_weight = class_weights,
		probability = False,
		random_state = random_seed,
	)
	model.fit( X_train_scaled, y_train )

	log.info( f'Baseline SVM trained on { X_train_scaled.shape[ 0 ] } samples, { X_train_scaled.shape[ 1 ] }-dim features, classes_={ list( model.classes_ ) }' )

	artifact_uris = get_artifact_uris()

	for obj, uri in ( ( model, artifact_uris[ 'model' ] ), ( scaler, artifact_uris[ 'scaler' ] ) ):
		buffer = io.BytesIO()
		joblib.dump( obj, buffer )
		upload_bytes_to_blob( uri, buffer.getvalue() )

	return model, scaler


def load_baseline_artifacts() -> tuple[ SVC, StandardScaler ]:
	'''
	Read back the SVM and scaler persisted by train_baseline_svm() — joblib.load() via an
	in-memory buffer, no local disk touched either direction.
	'''

	artifact_uris = get_artifact_uris()

	model_bytes = download_blob_to_bytes( artifact_uris[ 'model' ] )
	scaler_bytes = download_blob_to_bytes( artifact_uris[ 'scaler' ] )

	model = joblib.load( io.BytesIO( model_bytes ) )
	scaler = joblib.load( io.BytesIO( scaler_bytes ) )

	return model, scaler


def load_cached_features( split_name: str ) -> tuple[ np.ndarray, np.ndarray ]:
	'''
	Read back a split's feature/label matrices cached by build_feature_matrices() — the
	reader promised but deliberately left out of 4.2, now that 4.3 actually needs it.
	'''

	X_bytes = download_blob_to_bytes( f'gs://{ config.GCS_BUCKET }/{ FEATURE_CACHE_PREFIX }/{ split_name }_X.npy' )
	y_bytes = download_blob_to_bytes( f'gs://{ config.GCS_BUCKET }/{ FEATURE_CACHE_PREFIX }/{ split_name }_y.npy' )

	X = np.load( io.BytesIO( X_bytes ) )
	y = np.load( io.BytesIO( y_bytes ) )

	log.info( f'Loaded "{ split_name }" features: { X.shape }' )

	return X, y


def evaluate_baseline_svm( model: SVC, scaler: StandardScaler, X_test: np.ndarray, y_test: np.ndarray ) -> dict:
	'''
	Score the trained baseline SVM against a held-out split's cached features.

	scaler.transform() only — never re-fit. The post-scale mean/std logged here should
	sit close to but not exactly 0/1: it was fit on the train split's parameters, not
	test's own, so an exact 0/1 here would actually indicate leakage (scaler re-fit on
	test), not a healthy pipeline.

	roc_auc_score has no pos_label override for binary input — checked directly against
	the installed sklearn (1.7.x), not assumed from memory. For binary y_true it always
	treats the lexicographically greater class as positive, which for { 'FAKE', 'REAL' }
	is 'REAL' — and that happens to be exactly the class decision_function() itself
	scores positively toward (classes_[1], confirmed empirically: SVC.classes_ sorts to
	['FAKE', 'REAL'] and decision_function() runs positive for classes_[1]). So the raw
	decision_function() output is passed straight through, unmodified — negating it
	would silently flip the AUC to its complement (1 - AUC), not an equivalent
	orientation, since roc_auc_score can't be told to treat the negated scores as
	pointing at 'FAKE' instead. The resulting AUC-ROC is therefore with respect to
	detecting 'REAL', not 'FAKE' — worth carrying that framing into the report if this
	number sits next to precision/recall computed the other way round. That positive
	class is derived here (np.unique(y_test)[-1], the same rule roc_auc_score applies
	internally) rather than hardcoded, and returned as 'auc_roc_positive_class' so 4.4
	records it in Firestore instead of it needing to be re-derived later from memory.
	predict_proba() is not used — probability=False upstream (train_baseline_svm) rules
	it out for determinism.
	'''

	X_test_scaled = scaler.transform( X_test )

	log.info( f'Test features post-scale: mean={ X_test_scaled.mean():.4f}, std={ X_test_scaled.std():.4f}' )

	y_pred = model.predict( X_test_scaled )
	log.info( f'Test predictions: { dict( zip( *np.unique( y_pred, return_counts=True ) ) ) }' )

	accuracy = accuracy_score( y_test, y_pred )
	auc_roc = roc_auc_score( y_test, model.decision_function( X_test_scaled ) )
	auc_roc_positive_class = np.unique( y_test )[ -1 ]

	log.info( f'Baseline evaluation — accuracy={ accuracy:.4f}, auc_roc={ auc_roc:.4f}, auc_roc_positive_class={ auc_roc_positive_class }' )

	return {
		'accuracy': accuracy,
		'auc_roc': auc_roc,
		'auc_roc_positive_class': str( auc_roc_positive_class ),
	}
