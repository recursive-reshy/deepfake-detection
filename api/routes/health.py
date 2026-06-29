# FastAPI
from fastapi import APIRouter
from fastapi.responses import JSONResponse
# Db
from src.db.client import get_client

router = APIRouter()


@router.get( '/health' )
async def health():

	try:
		client = get_client()
		client.collection( 'health_check' ).document( 'ping' ).get()
	except Exception as e:
		return JSONResponse( status_code = 503, content = { 'status': 'unhealthy', 'firestore': str( e ) } )

	return { 'status': 'healthy', 'firestore': 'ok' }
