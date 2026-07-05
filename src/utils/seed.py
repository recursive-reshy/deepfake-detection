# Standard library
import logging
import random
# NumPy
import numpy as np
# TensorFlow
import tensorflow as tf

log = logging.getLogger( __name__ )


def set_global_seed( random_seed: int ) -> int:
	'''
	Seed every global source of randomness the pipeline touches: Python's random module,
	NumPy's global state, and TensorFlow's global state. Called once, early, before data
	loading or model construction — per ensemble member if train.py re-seeds between
	members (Section 9.3 of the architecture doc differentiates members partly by seed).

	Albumentations finding (checked directly against the installed 2.3.x, not assumed):
	A.Compose does NOT inherit from NumPy's global random state. Each Compose instance
	builds its own independent numpy.random.Generator via np.random.default_rng(seed), and
	when no seed is passed to Compose, that generator is seeded from OS entropy — entirely
	disconnected from the np.random.seed() call this function makes. Calling
	set_global_seed() alone does NOT make augment.py's transform choices reproducible.
	train.py (3.9) must pass random_seed through explicitly to wherever augment.py builds
	its A.Compose pipeline (e.g. A.Compose( [...], seed=random_seed )) — this function
	stays decoupled from augment.py and only seeds/returns the value for train.py to pass
	along itself.

	Returns random_seed unchanged, so a call site can seed and thread the same value into
	augment.py in one line without re-reading it back out of ExperimentConfig.
	'''

	random.seed( random_seed )
	np.random.seed( random_seed )
	tf.random.set_seed( random_seed )

	log.info( f'Global seed set to { random_seed } (random, numpy, tensorflow)' )

	return random_seed
