# Standard library
import csv
import os
import time
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
# Requests
import requests

FAILED_CSV_PATH = './data/failed_downloads.csv'
IMAGES_DIR = './data/images'
REQUEST_TIMEOUT = 15

MULTIAVATAR_DOMAIN = 'api.multiavatar.com'
MULTIAVATAR_WORKERS = 16
MULTIAVATAR_RETRIES = 3
MULTIAVATAR_BACKOFF_SECONDS = 2
MULTIAVATAR_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'

DICEBEAR_DOMAIN = 'api.dicebear.com'
DICEBEAR_WORKERS = 2
DICEBEAR_RETRIES = 3
DICEBEAR_BACKOFF_SECONDS = 5


def download_one( row: dict, max_retries: int, backoff_seconds: int, headers: dict ) -> tuple[ str, bool, str ]:

	image_id = row[ 'image_id' ]
	url = row[ 'url' ]
	dest_path = os.path.join( IMAGES_DIR, f'{ image_id }.jpg' )

	last_error = ''

	for attempt in range( max_retries ):
		print( f'[{ image_id }] attempt { attempt + 1 }/{ max_retries } -> { url }', flush=True )

		try:
			response = requests.get( url, headers=headers, timeout=REQUEST_TIMEOUT )
			response.raise_for_status()

			with open( dest_path, 'wb' ) as f:
				f.write( response.content )

			print( f'[{ image_id }] OK', flush=True )

			return ( image_id, True, '' )

		except Exception as e:
			last_error = str( e )

			print( f'[{ image_id }] failed: { last_error }', flush=True )

			if attempt < max_retries - 1:
				time.sleep( backoff_seconds * ( attempt + 1 ) )

	return ( image_id, False, last_error )


def retry_batch( rows: list, max_workers: int, max_retries: int, backoff_seconds: int, headers: dict ) -> tuple[ int, list ]:

	succeeded = 0
	failed_rows = []

	with ThreadPoolExecutor( max_workers=max_workers ) as executor:
		futures = { executor.submit( download_one, row, max_retries, backoff_seconds, headers ): row for row in rows }

		for future in as_completed( futures ):
			row = futures[ future ]
			image_id, ok, error = future.result()

			if ok:
				succeeded += 1
			else:
				failed_rows.append( { 'image_id': image_id, 'url': row[ 'url' ], 'error': error } )

	return ( succeeded, failed_rows )


if __name__ == '__main__':

	os.makedirs( IMAGES_DIR, exist_ok=True )

	with open( FAILED_CSV_PATH, 'r', newline='', encoding='utf-8' ) as f:
		rows = list( csv.DictReader( f ) )

	attempted = len( rows )

	# Split the retry batch by domain
	multiavatar_rows = []
	dicebear_rows = []
	other_rows = []

	for row in rows:
		domain = urlparse( row[ 'url' ] ).netloc

		if domain == MULTIAVATAR_DOMAIN:
			multiavatar_rows.append( row )
		elif domain == DICEBEAR_DOMAIN:
			dicebear_rows.append( row )
		else:
			other_rows.append( row )

	if other_rows:
		print( f'WARNING: { len( other_rows ) } rows are neither multiavatar.com nor dicebear.com and will not be retried', flush=True )

	# multiavatar.com — same concurrency, add a browser User-Agent to fix the 403s
	print( f'Retrying { len( multiavatar_rows ) } multiavatar.com rows with a browser User-Agent, { MULTIAVATAR_WORKERS } workers', flush=True )

	multiavatar_succeeded, multiavatar_failed = retry_batch(
		multiavatar_rows,
		max_workers = MULTIAVATAR_WORKERS,
		max_retries = MULTIAVATAR_RETRIES,
		backoff_seconds = MULTIAVATAR_BACKOFF_SECONDS,
		headers = { 'User-Agent': MULTIAVATAR_USER_AGENT },
	)

	# dicebear.com — low concurrency, longer backoff to work around the 429s
	print( f'Retrying { len( dicebear_rows ) } dicebear.com rows at low concurrency, { DICEBEAR_WORKERS } workers', flush=True )

	dicebear_succeeded, dicebear_failed = retry_batch(
		dicebear_rows,
		max_workers = DICEBEAR_WORKERS,
		max_retries = DICEBEAR_RETRIES,
		backoff_seconds = DICEBEAR_BACKOFF_SECONDS,
		headers = {},
	)

	succeeded = multiavatar_succeeded + dicebear_succeeded
	still_failed = multiavatar_failed + dicebear_failed + [
		{ 'image_id': row[ 'image_id' ], 'url': row[ 'url' ], 'error': 'unexpected domain, not retried' } for row in other_rows
	]

	# Flag rather than retry further if the UA header didn't fix multiavatar.com
	still_403 = [ row for row in multiavatar_failed if '403' in row[ 'error' ] ]

	if still_403:
		print( f'FLAG: { len( still_403 ) } multiavatar.com rows still returned 403 even with a browser User-Agent header.', flush=True )
		print( 'The block is likely not request-signature based (IP block, referer check, etc) — stopping here, needs a different fix.', flush=True )

	# Overwrite failed_downloads.csv with the current true state of failures
	with open( FAILED_CSV_PATH, 'w', newline='', encoding='utf-8' ) as f:
		writer = csv.DictWriter( f, fieldnames=[ 'image_id', 'url', 'error' ] )
		writer.writeheader()
		writer.writerows( still_failed )

	# Summary
	print( f'Attempted: { attempted }' )
	print( f'Succeeded: { succeeded }' )
	print( f'Still failed: { len( still_failed ) }' )
