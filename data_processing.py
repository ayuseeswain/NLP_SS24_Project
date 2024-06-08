import pandas as pd
import os
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the main album and year data
albums_df = pd.read_csv('dataset/Album_Lists.csv')

# Directory containing the albums
album_dir = 'dataset'

# List to hold track data
track_data = []

# Expected columns in each album CSV
expected_columns = ['ID', 'Tracks']

# Iterate through each album file and load track data
for album in albums_df['Album']:
    album_file = os.path.join(album_dir, album, f'{album}.csv')
    if os.path.exists(album_file):
        album_tracks = pd.read_csv(album_file)
        if 'Unnamed: 0' in album_tracks.columns:
            album_tracks = album_tracks.drop(columns=['Unnamed: 0'])  # Drop the unnecessary column
        for col in expected_columns:
            if col not in album_tracks.columns:
                album_tracks[col] = None
        
        album_tracks = album_tracks[expected_columns]  # Keep only expected columns
        album_tracks['Album'] = album  # Add the album name to each track entry
        track_data.append(album_tracks)
    else:
        print(f"File not found: {album_file}")

# Combine all track data into a single DataFrame
tracks_df = pd.concat(track_data, ignore_index=True)

#print(f"Combined DataFrame:\n{tracks_df.head(100)}")


def format_track_name(track_name):
    # Replace non-alphanumeric characters with underscores and remove spaces
    formatted_name = re.sub(r'[^A-Za-z0-9]+', '_', track_name.replace(' ', ''))
    return formatted_name

# Apply the function to create a new column for formatted track names
tracks_df['formatted_track_name'] = tracks_df['Tracks'].apply(format_track_name)

# Function to load lyrics from a text file based on the formatted track name
def load_lyrics(formatted_track_name):
    for album in albums_df['Album']:
        lyric_file = os.path.join(album_dir, album, f'{formatted_track_name}.txt')
        if os.path.exists(lyric_file):
            with open(lyric_file, 'r', encoding='utf-8') as file:
                return file.read()
    return None

# Apply the function to load lyrics for each track
tracks_df['lyrics'] = tracks_df['formatted_track_name'].apply(load_lyrics)

# Filter tracks with no lyrics
tracks_with_no_lyrics = tracks_df[tracks_df['lyrics'].isna()]
#print(tracks_with_no_lyrics[['Tracks', 'Album']])

#print(tracks_df.head(50))
#print(tracks_df[['Album','Tracks']])

# Merge the album data with track data
merged_df = pd.merge(tracks_df, albums_df, on='Album')

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize necessary tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define cleaning functions
def clean_lyrics(lyrics):
    if lyrics:
        lyrics = lyrics.lower()
        lyrics = re.sub(f'[{re.escape(string.punctuation)}]', '', lyrics)
        lyrics = re.sub(r'\d+', '', lyrics)
        lyrics = re.sub(r'\s+', ' ', lyrics).strip()
    return lyrics

def remove_stopwords(lyrics):
    if lyrics:
        words = lyrics.split()
        filtered_words = [word for word in words if word not in stop_words]
        return ' '.join(filtered_words)
    return lyrics

def tokenize_and_lemmatize(lyrics):
    if lyrics:
        tokens = word_tokenize(lyrics)
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)
    return lyrics

# Apply cleaning functions to the lyrics column
merged_df['cleaned_lyrics'] = merged_df['lyrics'].apply(clean_lyrics)
merged_df['cleaned_lyrics'] = merged_df['cleaned_lyrics'].apply(remove_stopwords)
merged_df['processed_lyrics'] = merged_df['cleaned_lyrics'].apply(tokenize_and_lemmatize)

#print(merged_df[['Album', 'Tracks', 'processed_lyrics']].head())


# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the processed lyrics
X = tfidf_vectorizer.fit_transform(merged_df['processed_lyrics'].dropna())

# Convert the TF-IDF matrix to a DataFrame for easier inspection
tfidf_df = pd.DataFrame(X.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

print(tfidf_df.head(10))