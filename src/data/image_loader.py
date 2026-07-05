# TensorFlow
import tensorflow as tf
# NumPy
import numpy as np
# Utils
from src.utils.gcs import download_blob_to_bytes


def load_image( gcs_uri: str, img_size: int ) -> np.ndarray:
	'''
	Fetch a single image from GCS, decode it, and resize it to (img_size, img_size, 3).

	Returns a uint8 array with pixel values in [0, 255] — the raw decoded range expected
	by the Albumentations pipeline (augment.py) and DFT preprocessing (dft.py). Any
	float normalisation for the model happens downstream, not here.
	'''

	image_bytes = download_blob_to_bytes( gcs_uri )

	image = tf.io.decode_jpeg( image_bytes, channels=3 )
	image = tf.image.resize( image, [ img_size, img_size ], method=tf.image.ResizeMethod.BILINEAR )
	image = tf.cast( tf.round( image ), tf.uint8 )

	return image.numpy()
