import pandas as pd
import re
import numpy as np

def normalize_title(title):
    if pd.isna(title):
        return ""
    s = title.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)  # remove all punctuation
    return s

def load_and_process_movielens():
    movies = pd.read_csv('data/movies.csv')

    # Extract year as numeric and normalize titles
    movies['title_original'] = movies['title']
    movies['title'] = movies['title'].str.lower().str.strip()
    movies['year'] = pd.to_numeric(movies['title'].str.extract(r'\((\d{4})\)')[0], errors='coerce')
    movies['title_norm'] = movies['title'].apply(normalize_title)

    imdb_metadata = pd.read_pickle('data/imdb_metadata.pkl')
    imdb_metadata['primaryTitle_norm'] = imdb_metadata['primaryTitle'].apply(normalize_title)
    imdb_metadata['startYear'] = pd.to_numeric(imdb_metadata['startYear'], errors='coerce')

    # Do a loose merge on normalized titles
    merged = pd.merge(
        movies,
        imdb_metadata,
        left_on='title_norm',
        right_on='primaryTitle_norm',
        how='left',
        suffixes=('', '_imdb')
    )

    # Calculate year difference and keep matches where difference is <= 1 or startYear missing
    merged['year_diff'] = (merged['year'] - merged['startYear']).abs()
    merged_filtered = merged[(merged['year_diff'] <= 1) | (merged['startYear'].isna())]

    # Fill missing director/actor names with 'Unknown'
    merged_filtered['director_name'] = merged_filtered['director_name'].fillna('Unknown').replace('', 'Unknown')
    merged_filtered['actor_names'] = merged_filtered['actor_names'].fillna('Unknown').replace('', 'Unknown')

    # Keep relevant columns only and reset index
    final_movies = merged_filtered[['title_original', 'title', 'year', 'genres', 'director_name', 'actor_names']].copy()
    final_movies.rename(columns={'title_original': 'title'}, inplace=True)
    final_movies.reset_index(drop=True, inplace=True)

    final_movies.to_pickle('data/enriched_movies.pkl')

    ratings = pd.read_csv('data/ratings.csv')
    ratings.to_pickle('data/processed_ratings.pkl')

    print("Data ingestion complete, enriched movies saved.")
    print(f"Movies with director known: {(final_movies['director_name'] != 'Unknown').sum()} of {len(final_movies)}")
    print(f"Movies with actors known: {(final_movies['actor_names'] != 'Unknown').sum()} of {len(final_movies)}")

if __name__ == "__main__":
    load_and_process_movielens()
