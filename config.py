# Standard library
import os
from dotenv import load_dotenv

load_dotenv()

GCP_PROJECT_ID = os.getenv( 'GCP_PROJECT_ID' )
GCP_REGION = os.getenv( 'GCP_REGION', 'asia-southeast1' )
GCS_BUCKET = os.getenv( 'GCS_BUCKET' )
IMAGE_URI = os.getenv( 'IMAGE_URI' )
GOOGLE_APPLICATION_CREDENTIALS = os.getenv( 'GOOGLE_APPLICATION_CREDENTIALS' )
LOG_PREDICTIONS = True