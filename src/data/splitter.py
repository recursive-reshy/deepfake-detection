# Standard library
import logging
# Pandas
import pandas as pd

log = logging.getLogger( __name__ )

EXPECTED_SPLIT_COUNTS = { 'train': 4593, 'val': 641, 'test': 1323 }


def split_dataset( manifest: pd.DataFrame ) -> tuple[ dict[ str, pd.DataFrame ], dict[ str, float ] ]:
	'''
	Partition the manifest DataFrame (from loader.py) by dataset_split, and compute class
	weights from the train partition's own label distribution only — not the full dataset.

	Class weight formula: inverse frequency, weight[label] = n_samples / (n_classes * count),
	the standard "balanced" heuristic (equivalent to
	sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', ...)), computed
	by hand here rather than adding scikit-learn as a dependency before Phase 4/6 needs it.

	Returns (splits, class_weights):
	  splits        — { 'train': df, 'val': df, 'test': df }, filtered by dataset_split with
	                  row order preserved. No shuffling — that is train.py's concern.
	  class_weights — { label: weight }, derived only from splits[ 'train' ].
	'''

	splits = {}

	for split_name in ( 'train', 'val', 'test' ):
		split_df = manifest[ manifest[ 'dataset_split' ] == split_name ]

		if split_df.empty:
			raise ValueError( f'Split "{ split_name }" is empty after filtering — check dataset_split values in the manifest' )

		expected = EXPECTED_SPLIT_COUNTS[ split_name ]

		if len( split_df ) != expected:
			log.warning( f'Split "{ split_name }" has { len( split_df ) } rows, expected { expected }' )

		splits[ split_name ] = split_df

		log.info( f'Split "{ split_name }": { len( split_df ) } rows, label distribution: { split_df[ "label" ].value_counts().to_dict() }' )

	# Class weights — train partition only
	train_labels = splits[ 'train' ][ 'label' ]
	label_counts = train_labels.value_counts()
	n_samples = len( train_labels )
	n_classes = len( label_counts )

	class_weights = { label: n_samples / ( n_classes * count ) for label, count in label_counts.items() }

	log.info( f'Class weights (train partition, inverse frequency): { class_weights }' )

	return splits, class_weights
