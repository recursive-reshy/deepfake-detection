# Standard library
from urllib.parse import urlparse
# GCP
from google.cloud import storage
# Config
import config


def download_blob_to_bytes( gcs_uri: str ) -> bytes:
	parsed = urlparse( gcs_uri )
	bucket_name = parsed.netloc
	blob_path = parsed.path.lstrip( '/' )

	client = storage.Client( project = config.GCP_PROJECT_ID )
	bucket = client.bucket( bucket_name )
	blob = bucket.blob( blob_path )

	return blob.download_as_bytes()


def upload_bytes_to_blob( gcs_uri: str, data: bytes ) -> None:
	parsed = urlparse( gcs_uri )
	bucket_name = parsed.netloc
	blob_path = parsed.path.lstrip( '/' )

	client = storage.Client( project = config.GCP_PROJECT_ID )
	bucket = client.bucket( bucket_name )
	blob = bucket.blob( blob_path )

	blob.upload_from_string( data )
