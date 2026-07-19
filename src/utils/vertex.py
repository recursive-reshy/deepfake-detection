# YAML
import yaml
# Vertex AI
from google.cloud import aiplatform
# Config
import config


def submit_vertex_job( job_id: str, stage: str ) -> str:
	'''
	Load infra/vertex_job.yaml, substitute job_id/image_uri/stage, and submit a Vertex AI
	CustomJob. Shared by api/routes/train.py (Stage 1, API-triggered) and
	src/training/train.py (Stage 2, self-submitted by Stage 1's own container on success) —
	same image, same entry point, different --stage value (train.py's own CLI arg reads
	it), per the base/adversarial training split always starting each stage in a brand-new
	Vertex AI-provisioned container.

	stage is 'base' or 'adversarial' — train.py's --stage vocabulary, not the same as
	JobDocument.training_stage ('base_training'/'adversarial_finetuning'/'complete'), which
	tracks Firestore-facing progress rather than selecting a container entry point.

	Returns the submitted job's Vertex AI resource name, for the caller to persist as a
	debugging cross-reference — the Firestore job_id itself never changes across stages.
	'''

	with open( 'infra/vertex_job.yaml' ) as f:
		spec_text = f.read()

	spec_text = spec_text \
		.replace( '{job_id}', job_id ) \
		.replace( '{image_uri}', config.IMAGE_URI or '' ) \
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

	aiplatform.init( project=config.GCP_PROJECT_ID, location=config.GCP_REGION )

	custom_job = aiplatform.CustomJob(
		display_name = spec[ 'displayName' ],
		worker_pool_specs = worker_pool_specs,
		staging_bucket = f"gs://{ config.GCS_BUCKET }"
	)
	custom_job.submit()

	return custom_job.resource_name
