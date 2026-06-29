# FastAPI
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get( '/jobs' )
async def list_jobs():
	return JSONResponse( status_code = 501, content = { 'detail': 'Not implemented' } )


@router.get( '/jobs/{job_id}/status' )
async def get_job_status( job_id: str ):
	return JSONResponse( status_code = 501, content = { 'detail': 'Not implemented' } )


@router.get( '/jobs/{job_id}/results' )
async def get_job_results( job_id: str ):
	return JSONResponse( status_code = 501, content = { 'detail': 'Not implemented' } )
