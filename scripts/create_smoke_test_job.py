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

	# Task 5.5 memory-mitigation amendment — max_samples_per_split=500, batch_size=8 gives
	# ceil(500/8)=63 steps/epoch, enough sustained iteration to actually exercise the
	# allocator-fragmentation path that a 2-step run (the old max_samples_per_split=10)
	# never triggered, without running the full production dataset.
	smoke_test_config = ExperimentConfig(
		backbone = 'xception',
		img_size = 299,
		use_patch_pipeline = True,
		max_samples_per_split = 500,
		ensemble_size = 1,
		epochs = 2,
		batch_size = 8,
		early_stopping_patience = 1,
		adversarial_training = True,
		dataset_path = config.GCS_DATASET_PATH,
	)

	job_doc = JobDocument(
		job_id = job_id,
		stage = 'train',
		status = 'PENDING',
		training_stage = 'base_training',
		config = smoke_test_config,
		created_at = now,
		updated_at = now,
	)
	jobs.create_job( job_doc )

	print( job_id )


if __name__ == '__main__':
	main()
