import pandas as pd
import os

# Load the main album and year data
albums_df = pd.read_csv('Album_Lists.csv')  # Ensure this file has columns like 'album', 'year'

# Directory containing the album CSV files
album_dir = 'Album_Lists'

# List to hold track data
track_data = []

# Iterate through each album file and load track data
for album in albums_df['Album']:
    album_file = os.path.join(album, f'{album}.csv')
    #album_file = f'{album}.csv'
    if os.path.exists(album_file):
        album_tracks = pd.read_csv(album_file)
        album_tracks['Album'] = album  # Add the album name to each track entry
        track_data.append(album_tracks)

# Combine all track data into a single DataFrame
#tracks_df = pd.concat(track_data, ignore_index=True)

#print(tracks_df.head())
print(track_data)