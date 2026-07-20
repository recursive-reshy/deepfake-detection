# Standard library
import logging
import os
# FastAPI
from fastapi import FastAPI
# Structured logging
from pythonjsonlogger.json import JsonFormatter
# Routes
from api.routes import health, evaluate, jobs, predict, train

# Root logging config for the whole Cloud Run process — without this, the root logger
# defaults to WARNING with no handler, so every log.info() call across api/routes/*
# and src/utils/vertex.py is silently dropped rather than reaching stdout.
handler = logging.StreamHandler()
handler.setFormatter( JsonFormatter(
	'%(levelname)s %(name)s %(message)s',
	rename_fields = { 'levelname': 'severity' },
	timestamp = True,
) )
logging.basicConfig( level=logging.INFO, handlers=[ handler ], force=True )

log = logging.getLogger( __name__ )

# Git SHA (baked in at build time by the Dockerfile's GIT_SHA build arg) — logged once at
# process startup so any Cloud Run request log can directly answer "is this the code we
# think it is", the same check src/training/train.py does on its own startup.
log.info( 'API container started', extra={ 'git_sha': os.getenv( 'GIT_SHA', 'unknown' ) } )

app = FastAPI( title = 'Deepfake Detection API' )

app.include_router( health.router )
app.include_router( train.router )
app.include_router( evaluate.router )
app.include_router( predict.router )
app.include_router( jobs.router )
