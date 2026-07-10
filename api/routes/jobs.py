# FastAPI
from fastapi import APIRouter
from fastapi.responses import JSONResponse
# DB
from src.db import jobs as jobs_db

router = APIRouter()


@router.get( '/jobs' )
async def list_jobs():
	return JSONResponse( status_code = 501, content = { 'detail': 'Not implemented' } )


@router.get( '/jobs/{job_id}/status' )
async def get_job_status( job_id: str ):

	job = jobs_db.get_job( job_id )

	if job is None:
		return JSONResponse( status_code = 404, content = { 'detail': f'Job { job_id } not found' } )

	return {
		'job_id': job.job_id,
		'status': job.status,
		'current_epoch': job.current_epoch,
		'error': job.error,
		'updated_at': job.updated_at.isoformat(),
	}


@router.get( '/jobs/{job_id}/results' )
async def get_job_results( job_id: str ):
	return JSONResponse( status_code = 501, content = { 'detail': 'Not implemented' } )
