# FastAPI
from fastapi import APIRouter
from fastapi import File
from fastapi import UploadFile
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post( '/predict' )
async def predict( file: UploadFile = File( ... ) ):
	return JSONResponse( status_code = 501, content = { 'detail': 'Not implemented' } )
