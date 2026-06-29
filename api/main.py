# FastAPI
from fastapi import FastAPI
# Routes
from api.routes.health import router as health_router

app = FastAPI()

app.include_router( health_router )
