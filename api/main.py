# FastAPI
from fastapi import FastAPI
# Routes
from api.routes import health, evaluate, jobs, predict, train

app = FastAPI( title = 'Deepfake Detection API' )

app.include_router( health.router )
app.include_router( train.router )
app.include_router( evaluate.router )
app.include_router( predict.router )
app.include_router( jobs.router )
