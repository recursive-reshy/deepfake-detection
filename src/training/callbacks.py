# Standard library
import logging
import os
from datetime import datetime
from datetime import timezone
# TensorFlow
import tensorflow as tf
# SendGrid
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
# Config
import config
# Schemas
from src.schemas.db import EpochRecord
# DB
from src.db import epochs
from src.db import jobs

log = logging.getLogger( __name__ )

# Must be a SendGrid-verified sender — no dedicated config var for this exists (CLAUDE.md's
# env var table only documents NOTIFY_EMAIL, the recipient), so it's a fixed constant here
# rather than invented config surface nothing else reads.
NOTIFICATION_FROM_EMAIL = 'noreply@deepfake-detection.dev'


class FirestoreEpochCallback( tf.keras.callbacks.Callback ):
	'''
	Bridges Keras's on_epoch_end to Firestore — the epoch -> job-store update this folder's
	own name promises. Writes one EpochRecord per epoch under jobs/{job_id}/epochs/ and
	bumps the job document's current_epoch + updated_at, so Firestore (not something
	reconstructed only at the end of training) is the live source of per-epoch progress —
	a container restart mid-training loses nothing already past this callback, consistent
	with the "Firestore as sole state source" principle already locked from Phase 3/4.
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


def send_job_completion_email( job_id: str, status: str ) -> None:
	'''
	Fires once, after every ensemble member finishes (or the job fails) — not a per-epoch
	Keras callback, called directly from train.py's main() alongside the terminal status
	transition. Lives in this module rather than train.py because callbacks.py's job, per
	the System Architecture Doc's folder structure, is "Keras callbacks + Firestore job
	state writes + SendGrid".

	SENDGRID_API_KEY is read from the environment only (os.getenv), never from .env, never
	hardcoded, per CLAUDE.md's Secrets section. Cloud Run gets it via --set-secrets; the
	Vertex AI training container needs the equivalent secure delivery wired at the infra
	layer (Secret Manager + the job's service account) — outside this module's scope, and
	no application code change is needed either way, since both paths land the value in
	the same environment variable this function reads.

	A missing key/recipient or a SendGrid API failure is logged and swallowed, not raised —
	email is a notification, not a step whose failure should flip an otherwise-successful
	training job to FAILED.
	'''

	api_key = os.getenv( 'SENDGRID_API_KEY' )

	if not api_key or not config.NOTIFY_EMAIL:
		log.warning( 'SendGrid notification skipped — SENDGRID_API_KEY or NOTIFY_EMAIL not configured', extra={ 'job_id': job_id } )
		return

	message = Mail(
		from_email = NOTIFICATION_FROM_EMAIL,
		to_emails = config.NOTIFY_EMAIL,
		subject = f'Training job { job_id } { status.lower() }',
		html_content = f'<p>Training job <strong>{ job_id }</strong> finished with status <strong>{ status }</strong>.</p>',
	)

	try:
		SendGridAPIClient( api_key ).send( message )
		log.info( 'Job completion email sent', extra={ 'job_id': job_id, 'status': status, 'recipient': config.NOTIFY_EMAIL } )
	except Exception as exc:
		log.error( 'Job completion email failed to send', extra={ 'job_id': job_id, 'error': str( exc ) } )
