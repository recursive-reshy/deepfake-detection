# Standard library
import argparse
import logging
import sys
# DB
from src.db import jobs

logging.basicConfig( level=logging.INFO )
log = logging.getLogger( __name__ )


def main() -> None:

	parser = argparse.ArgumentParser()
	parser.add_argument( '--job-id', required=True )
	args = parser.parse_args()

	job_id = args.job_id

	# Fetch job document
	job = jobs.get_job( job_id )

	if job is None:
		log.error( f'Job { job_id } not found in Firestore' )
		sys.exit( 1 )

	# Mark running
	jobs.update_job_status( job_id, 'RUNNING' )

	raise NotImplementedError( 'Training pipeline not yet implemented' )


if __name__ == '__main__':
	main()
