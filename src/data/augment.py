# NumPy
import numpy as np
# Albumentations
import albumentations as A


def augment_image( image: np.ndarray, seed: int | None = None ) -> np.ndarray:
	'''
	Apply the training-split augmentation pipeline to a single image.

	Expects and returns a uint8 array with pixel values in [0, 255] — the same contract
	produced by image_loader.py. Never called on val/test images; the caller decides that.

	seed — threaded straight through to A.Compose. A.Compose builds its own independent
	numpy.random.Generator via np.random.default_rng(seed) rather than inheriting NumPy's
	global random state, so set_global_seed() (seed.py) alone cannot make this
	reproducible or differentiate callers — this parameter is how a caller (e.g.
	ensemble.py, giving each member a distinct augmentation stream) does that instead.
	None (default) leaves this OS-entropy-seeded, unchanged from before this parameter
	existed.
	'''

	pipeline = A.Compose( [
		A.HorizontalFlip( p=0.5 ),
		A.ColorJitter( p=0.5 ),
		A.ImageCompression( quality_range=( 30, 90 ), p=0.5 ),
	], seed=seed )

	return pipeline( image=image )[ 'image' ]
