# NumPy
import numpy as np
# Albumentations
import albumentations as A


def augment_image( image: np.ndarray ) -> np.ndarray:
	'''
	Apply the training-split augmentation pipeline to a single image.

	Expects and returns a uint8 array with pixel values in [0, 255] — the same contract
	produced by image_loader.py. Never called on val/test images; the caller decides that.
	'''

	pipeline = A.Compose( [
		A.HorizontalFlip( p=0.5 ),
		A.ColorJitter( p=0.5 ),
		A.ImageCompression( quality_range=( 30, 90 ), p=0.5 ),
	] )

	return pipeline( image=image )[ 'image' ]
