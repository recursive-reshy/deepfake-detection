import pandas as pd

df = pd.read_csv( 'data\FINAL_DATASET.csv' )

df[ 'image_path' ] = 'gs://uneeb-zulfiqar-deepfake-detection/datasets/images/' + df[ 'image_id' ].astype( str ) + '.jpg'

print( len( df ) )
print( df.head() )

df.to_csv( 'data\FINAL_DATASET_UPDATED.csv', index=False )