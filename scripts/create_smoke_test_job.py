# Standard library
from datetime import datetime
from datetime import timezone
# Config
import config
# Schemas
from src.schemas.experiment import ExperimentConfig
from src.schemas.db import JobDocument
# DB
from src.db import jobs

# Writes a PENDING job document straight to Firestore via the same jobs.create_job() path
# POST /train uses internally, deliberately skipping that route's Vertex AI submission —
# Task 5.4's train.py run happens locally on CPU, not on a real Vertex AI job, so there's
# nothing for a submitted CustomJob to do here. See the Task 5.4 handoff brief's own note
# on this ("directly for this test if 5.7 isn't done yet").


def main() -> None:

	now = datetime.now( timezone.utc )
	job_id = f'train_{ now.strftime( "%Y%m%d_%H%M%S" ) }'

	smoke_test_config = ExperimentConfig(
		backbone = 'xception',
		img_size = 299,
		use_patch_pipeline = True,
		max_samples_per_split = 10,
		ensemble_size = 1,
		epochs = 1,
		early_stopping_patience = 1,
		adversarial_training = True,
		dataset_path = config.GCS_DATASET_PATH,
	)

	job_doc = JobDocument(
		job_id = job_id,
		stage = 'train',
		status = 'PENDING',
		config = smoke_test_config,
		created_at = now,
		updated_at = now,
	)
	jobs.create_job( job_doc )

	print( job_id )


if __name__ == '__main__':
	main()
