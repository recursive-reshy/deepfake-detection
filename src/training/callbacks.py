# Standard library
import logging
from datetime import datetime
from datetime import timezone
# TensorFlow
import tensorflow as tf
# Schemas
from src.schemas.db import EpochRecord
# DB
from src.db import epochs
from src.db import jobs

log = logging.getLogger( __name__ )


class FirestoreEpochCallback( tf.keras.callbacks.Callback ):
	'''
	Bridges Keras's on_epoch_end to Firestore — the epoch -> job-store update this folder's
	own name promises. Writes one EpochRecord per epoch under jobs/{job_id}/epochs/ and
	bumps the job document's current_epoch + updated_at, so Firestore (not something
	reconstructed only at the end of training) is the live source of per-epoch progress —
	a container restart mid-training loses nothing already past this callback, consistent
	with the "Firestore as sole state source" principle already locked from Phase 3/4.

	SendGrid notification is deliberately not wired here — this callback's job for Task 5.4
	is scoped to the epoch -> Firestore bridge only. Job-completion email is out of scope
	for a single-member smoke-test training run (see the 5.4 handoff brief).
	'''

	def __init__( self, job_id: str, ensemble_member: int ):
		super().__init__()

		self.job_id = job_id
		self.ensemble_member = ensemble_member

	def on_epoch_end( self, epoch: int, logs: dict | None = None ) -> None:
		logs = logs or {}

		record = EpochRecord(
			ensemble_member = self.ensemble_member,
			epoch = epoch,
			loss = logs[ 'loss' ],
			val_loss = logs[ 'val_loss' ],
			accuracy = logs[ 'accuracy' ],
			val_accuracy = logs[ 'val_accuracy' ],
			recorded_at = datetime.now( timezone.utc ),
		)

		epochs.write_epoch( self.job_id, record )
		jobs.update_job_progress( self.job_id, epoch )

		log.info( 'Epoch record written to Firestore', extra={
			'job_id': self.job_id, 'ensemble_member': self.ensemble_member, 'epoch': epoch,
			'loss': record.loss, 'val_loss': record.val_loss,
			'accuracy': record.accuracy, 'val_accuracy': record.val_accuracy,
		} )
