# NumPy
import numpy as np
# OpenCV
import cv2


def extract_azimuthal_power_spectrum( image: np.ndarray ) -> np.ndarray:
	'''
	Extract a 1D azimuthally-averaged power spectrum feature vector from a single image,
	per Durall et al. (2020) — feeds the classical Durall DFT + SVM baseline, not the
	ensemble. Separate technique and separate file from preprocessor.py (the DFT
	patch-grid pipeline the ensemble trains on) — do not conflate the two.

	Grayscale conversion happens first, before any FFT step — the paper notes this
	explicitly in a footnote and it is not optional. The grayscale image then goes
	through a 2D FFT, fftshift to centre the zero-frequency component, and the power
	spectrum (magnitude squared, not amplitude) is taken. Azimuthal averaging collapses
	the 2D spectrum to 1D by averaging all power values sharing the same integer radial
	distance from the centre.

	Bin count is the distance from the centre to the farthest corner of the frequency
	grid — ceil(sqrt((H/2)^2 + (W/2)^2)) — derived from the actual input shape, never
	hardcoded. The paper's 722-dim vector is specific to their 1024x1024 source images;
	at this pipeline's fixed 299x299 resolution this yields ~212 bins instead. Same
	method, different vector length as a function of input size — a principled
	adaptation, not a deviation, and documented as such in the report. No normalisation
	is applied on top of the radial mean — the paper does not specify one at this stage.
	'''

	# Grayscale conversion — mandatory before FFT, matches paper exactly
	grayscale = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY ).astype( np.float32 )

	# 2D FFT, centred, power spectrum
	freq = np.fft.fft2( grayscale )
	freq = np.fft.fftshift( freq )
	power_spectrum = np.abs( freq ) ** 2

	# Integer radial distance of every pixel from the spectrum centre
	height, width = power_spectrum.shape
	center_y, center_x = height / 2, width / 2

	y_indices, x_indices = np.indices( ( height, width ) )
	radii = np.sqrt( ( y_indices - center_y ) ** 2 + ( x_indices - center_x ) ** 2 )
	radii = radii.astype( np.int64 )

	# Azimuthal averaging — mean power at each integer radius, bin count derived from shape
	num_bins = int( np.ceil( np.sqrt( center_y ** 2 + center_x ** 2 ) ) )
	radial_sum = np.bincount( radii.ravel(), weights=power_spectrum.ravel(), minlength=num_bins )[ :num_bins ]
	radial_count = np.bincount( radii.ravel(), minlength=num_bins )[ :num_bins ]

	return ( radial_sum / ( radial_count + 1e-8 ) ).astype( np.float32 )
