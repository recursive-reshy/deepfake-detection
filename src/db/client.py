# GCP
from google.cloud import firestore
# Config
import config

def get_client() -> firestore.Client:
	return firestore.Client( project = config.GCP_PROJECT_ID )
