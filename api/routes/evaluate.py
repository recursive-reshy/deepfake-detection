# FastAPI
from fastapi import APIRouter
from fastapi.responses import JSONResponse
# Schemas
from src.schemas.experiment import ExperimentConfig

router = APIRouter()


@router.post( '/evaluate' )
async def submit_evaluation_job( config: ExperimentConfig ):
	return JSONResponse( status_code = 501, content = { 'detail': 'Not implemented' } )
