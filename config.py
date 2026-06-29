# Standard library
import os
from dotenv import load_dotenv

load_dotenv()

GCP_PROJECT_ID = os.getenv( 'GCP_PROJECT_ID' )
GOOGLE_APPLICATION_CREDENTIALS = os.getenv( 'GOOGLE_APPLICATION_CREDENTIALS' )
LOG_PREDICTIONS = True