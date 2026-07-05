# Standard library
import csv
import io
import os
from urllib.parse import urlparse
# Multiavatar (script-local — pip install multiavatar, not a project dependency)
from multiavatar.multiavatar import multiavatar
# resvg-py (script-local — pip install resvg-py, not a project dependency)
import resvg_py
# Pillow (script-local — pip install pillow, not a project dependency)
from PIL import Image

FAILED_CSV_PATH = './data/failed_downloads.csv'
IMAGES_DIR = './data/images'
IMG_SIZE = 299

MULTIAVATAR_DOMAIN = 'api.multiavatar.com'


def seed_from_url( url: str ) -> str:

	path = urlparse( url ).path
	filename = path.rsplit( '/', 1 )[ -1 ]

	return filename.rsplit( '.', 1 )[ 0 ]


if __name__ == '__main__':

	os.makedirs( IMAGES_DIR, exist_ok=True )

	with open( FAILED_CSV_PATH, 'r', newline='', encoding='utf-8' ) as f:
		rows = list( csv.DictReader( f ) )

	multiavatar_rows = [ row for row in rows if urlparse( row[ 'url' ] ).netloc == MULTIAVATAR_DOMAIN ]
	remaining_rows = [ row for row in rows if urlparse( row[ 'url' ] ).netloc != MULTIAVATAR_DOMAIN ]

	attempted = len( multiavatar_rows )
	succeeded = 0

	print( f'Regenerating { attempted } multiavatar.com images locally, no network calls', flush=True )

	for row in multiavatar_rows:
		image_id = row[ 'image_id' ]
		seed = seed_from_url( row[ 'url' ] )
		dest_path = os.path.join( IMAGES_DIR, f'{ image_id }.jpg' )

		svg = multiavatar( seed, None, None )
		png_bytes = bytes( resvg_py.svg_to_bytes( svg_string=svg, width=IMG_SIZE, height=IMG_SIZE ) )

		img = Image.open( io.BytesIO( png_bytes ) ).convert( 'RGB' )
		img.save( dest_path, 'JPEG' )

		succeeded += 1

		print( f'[{ image_id }] generated from seed "{ seed }"', flush=True )

	# Overwrite failed_downloads.csv with these 1,000 rows removed
	with open( FAILED_CSV_PATH, 'w', newline='', encoding='utf-8' ) as f:
		writer = csv.DictWriter( f, fieldnames=[ 'image_id', 'url', 'error' ] )
		writer.writeheader()
		writer.writerows( remaining_rows )

	# Summary
	print( f'Attempted: { attempted }' )
	print( f'Succeeded: { succeeded }' )
	print( f'Remaining in failed_downloads.csv: { len( remaining_rows ) }' )
