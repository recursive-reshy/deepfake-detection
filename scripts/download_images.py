# Standard library
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
# Requests
import requests

CSV_PATH = 'FINAL_DATASET.csv'
IMAGES_DIR = './data/images'
FAILED_CSV_PATH = './data/failed_downloads.csv'
MAX_WORKERS = 16
MAX_RETRIES = 3
BACKOFF_SECONDS = 2
REQUEST_TIMEOUT = 15


def download_one( row: dict ) -> tuple[ str, bool, str ]:

	image_id = row[ 'image_id' ]
	url = row[ 'image_url' ]
	dest_path = os.path.join( IMAGES_DIR, f'{ image_id }.jpg' )

	last_error = ''

	for attempt in range( MAX_RETRIES ):
		print( f'[{ image_id }] attempt { attempt + 1 }/{ MAX_RETRIES } -> { url }', flush=True )

		try:
			response = requests.get( url, timeout=REQUEST_TIMEOUT )
			response.raise_for_status()

			with open( dest_path, 'wb' ) as f:
				f.write( response.content )

			print( f'[{ image_id }] OK', flush=True )

			return ( image_id, True, '' )

		except Exception as e:
			last_error = str( e )

			print( f'[{ image_id }] failed: { last_error }', flush=True )

			if attempt < MAX_RETRIES - 1:
				time.sleep( BACKOFF_SECONDS * ( attempt + 1 ) )

	return ( image_id, False, last_error )


if __name__ == '__main__':

	os.makedirs( IMAGES_DIR, exist_ok=True )

	with open( CSV_PATH, 'r', newline='', encoding='utf-8' ) as f:
		rows = list( csv.DictReader( f ) )

	attempted = len( rows )
	succeeded = 0
	failed_rows = []

	print( f'Starting download of { attempted } images with { MAX_WORKERS } workers', flush=True )

	with ThreadPoolExecutor( max_workers=MAX_WORKERS ) as executor:
		futures = { executor.submit( download_one, row ): row for row in rows }

		completed = 0

		for future in as_completed( futures ):
			row = futures[ future ]
			image_id, ok, error = future.result()

			completed += 1

			if ok:
				succeeded += 1
			else:
				failed_rows.append( { 'image_id': image_id, 'url': row[ 'image_url' ], 'error': error } )

			if completed % 50 == 0 or completed == attempted:
				print( f'Progress: { completed }/{ attempted } done ({ succeeded } succeeded)', flush=True )

	# Log failures
	with open( FAILED_CSV_PATH, 'w', newline='', encoding='utf-8' ) as f:
		writer = csv.DictWriter( f, fieldnames=[ 'image_id', 'url', 'error' ] )
		writer.writeheader()
		writer.writerows( failed_rows )

	# Summary
	print( f'Attempted: { attempted }' )
	print( f'Succeeded: { succeeded }' )
	print( f'Failed: { len( failed_rows ) }' )
