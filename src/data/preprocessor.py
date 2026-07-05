# NumPy
import numpy as np
# OpenCV
import cv2


def preprocess_image( image: np.ndarray, patch_grid_size: int, img_size: int ) -> np.ndarray:
	'''
	Convert a single (img_size, img_size, 3) image into the DFT patch-grid representation
	the ensemble trains on: (patch_grid_size ** 2, img_size, img_size, 3) float32.

	Complex-to-real convention (locked decision — see report methodology):
	per-channel 2D DFT (np.fft.fft2), fftshift to centre the zero-frequency component so
	patches correspond to distinct radial frequency bands rather than an arbitrary corner
	split, then log-magnitude (log1p(abs(freq))) to compress the DC-dominated dynamic
	range, then per-image min-max scaling to [0, 1]. Output is float32 throughout — no
	complex dtype leaves this function.

	Patch grid: img_size does not divide evenly by patch_grid_size (299 / 9 = 33 remainder
	2), so row/column boundaries come from np.array_split, which distributes the remainder
	across a few patches (34px) rather than cropping pixels off the image. Each patch is
	then independently upsampled back to (img_size, img_size) via bilinear interpolation
	(cv2.INTER_LINEAR, explicit — not a library default).
	'''

	# Full-image frequency-domain representation, per channel
	spectrum = np.zeros( image.shape, dtype=np.float32 )

	for channel in range( image.shape[ -1 ] ):
		freq = np.fft.fft2( image[ :, :, channel ] )
		freq = np.fft.fftshift( freq )
		spectrum[ :, :, channel ] = np.log1p( np.abs( freq ) )

	spectrum_min = spectrum.min()
	spectrum_max = spectrum.max()
	spectrum = ( spectrum - spectrum_min ) / ( spectrum_max - spectrum_min + 1e-8 )

	# Non-overlapping patch grid, upsampled independently
	row_bounds = np.array_split( np.arange( img_size ), patch_grid_size )
	col_bounds = np.array_split( np.arange( img_size ), patch_grid_size )
	patches = []

	for rows in row_bounds:
		for cols in col_bounds:
			patch = spectrum[ rows[ 0 ] : rows[ -1 ] + 1, cols[ 0 ] : cols[ -1 ] + 1, : ]
			patch = cv2.resize( patch, ( img_size, img_size ), interpolation=cv2.INTER_LINEAR )
			patches.append( patch )

	return np.stack( patches, axis=0 ).astype( np.float32 )
