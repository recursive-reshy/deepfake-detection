# Standard library
import logging
# YAML
import yaml
# Vertex AI
from google.cloud import aiplatform
# Config
import config

log = logging.getLogger( __name__ )


def render_job_spec( job_id: str, stage: str ) -> tuple[ str, list[ dict ] ]:
	'''
	Load infra/vertex_job.yaml, substitute job_id/image_uri/stage, and return
	(display_name, worker_pool_specs) in Vertex AI SDK (snake_case) format — the exact
	payload submit_vertex_job() hands to aiplatform.CustomJob, with no API call made.
	Split out from submit_vertex_job() so scripts/validate_local.py can render and print
	this same payload for both stage values without touching GCP, catching a
	template-vs-submission drift locally instead of via a failed job.

	stage is 'base' or 'adversarial' — train.py's --stage vocabulary, not the same as
	JobDocument.training_stage ('base_training'/'adversarial_finetuning'/'complete'), which
	tracks Firestore-facing progress rather than selecting a container entry point.
	'''

	# Fail loudly, not with a silent '' fallback — a spec submitted with an empty imageUri
	# reaches the Vertex AI API as InvalidArgument: 400 Invalid image URI, which is a much
	# harder failure to trace back to "IMAGE_URI wasn't set" than catching it here. This
	# matters most for Stage 2's self-submission path (run_base_stage in
	# src/training/train.py, calling this from inside Stage 1's own running container):
	# that container only has IMAGE_URI in its environment because vertex_job.yaml's own
	# containerSpec.env carries it in, via the same {image_uri} substitution as
	# containerSpec.imageUri below — never assumed inherited from anywhere else.
	if not config.IMAGE_URI:
		raise RuntimeError(
			'IMAGE_URI is not set — refusing to render a Vertex AI job spec with an empty '
			'imageUri. Run scripts/build_and_push.sh to resolve the current image digest, '
			'then set IMAGE_URI (digest-pinned, e.g. image@sha256:...) in .env before '
			'submitting a job.'
		)

	with open( 'infra/vertex_job.yaml' ) as f:
		spec_text = f.read()

	spec_text = spec_text \
		.replace( '{job_id}', job_id ) \
		.replace( '{image_uri}', config.IMAGE_URI ) \
		.replace( '{stage}', stage )

	spec = yaml.safe_load( spec_text )

	# Build worker pool specs in snake_case (Vertex AI SDK format)
	raw_pools = spec[ 'jobSpec' ][ 'workerPoolSpecs' ]

	worker_pool_specs = [
		{
			'machine_spec': {
				'machine_type': pool[ 'machineSpec' ][ 'machineType' ],
				**(
					{
						'accelerator_type': pool[ 'machineSpec' ][ 'acceleratorType' ],
						'accelerator_count': pool[ 'machineSpec' ][ 'acceleratorCount' ],
					}
					if 'acceleratorType' in pool[ 'machineSpec' ] else {}
				),
			},
			'replica_count':  pool[ 'replicaCount' ],
			'container_spec': {
				'image_uri': pool[ 'containerSpec' ][ 'imageUri' ],
				'command': pool[ 'containerSpec' ][ 'command' ],
				'args': pool[ 'containerSpec' ][ 'args' ],
				'env': [
					{ 'name': e[ 'name' ], 'value': e[ 'value' ] }
					for e in pool[ 'containerSpec' ].get( 'env', [] )
				],
			},
		}
		for pool in raw_pools
	]

	return spec[ 'displayName' ], worker_pool_specs


def submit_vertex_job( job_id: str, stage: str ) -> str:
	'''
	Render infra/vertex_job.yaml via render_job_spec() and submit a Vertex AI CustomJob.
	Shared by api/routes/train.py (Stage 1, API-triggered) and src/training/train.py
	(Stage 2, self-submitted by Stage 1's own container on success) — same image, same
	entry point, different --stage value (train.py's own CLI arg reads it), per the
	base/adversarial training split always starting each stage in a brand-new
	Vertex AI-provisioned container.

	Returns the submitted job's Vertex AI resource name, for the caller to persist as a
	debugging cross-reference — the Firestore job_id itself never changes across stages.
	'''

	display_name, worker_pool_specs = render_job_spec( job_id, stage )

	# Log the literal command/args/worker_pool_specs actually being submitted (not the
	# source YAML) — this is the payload the CustomJob API receives, so a template-vs-
	# submission drift (e.g. a fix landing in infra/vertex_job.yaml but a stale cached
	# spec still going out) is visible from this job's own log, not just a repo diff.
	log.info( 'Submitting Vertex AI CustomJob', extra={
		'job_id': job_id, 'stage': stage, 'worker_pool_specs': worker_pool_specs,
	} )

	aiplatform.init( project=config.GCP_PROJECT_ID, location=config.GCP_REGION )

	custom_job = aiplatform.CustomJob(
		display_name = display_name,
		worker_pool_specs = worker_pool_specs,
		staging_bucket = f"gs://{ config.GCS_BUCKET }"
	)
	custom_job.submit()

	return custom_job.resource_name
