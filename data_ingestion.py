import pandas as pd
import re

def normalize_title(title):
    if pd.isna(title):
        return ""
    s = title.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)  # remove all punctuation
    return s

def load_and_process_movielens():
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')

    # Normalize titles and extract years with better cleaning
    movies['title_original'] = movies['title']
    movies['title'] = movies['title'].str.lower().str.strip()
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')[0]

    # Normalize titles by removing punctuation
    movies['title_norm'] = movies['title'].apply(normalize_title)

    imdb_metadata = pd.read_pickle('data/imdb_metadata.pkl')
    imdb_metadata['primaryTitle_norm'] = imdb_metadata['primaryTitle'].apply(normalize_title)

    # Join on normalized title and year
    movies = movies.merge(
        imdb_metadata,
        left_on=['title_norm', 'year'],
        right_on=['primaryTitle_norm', 'startYear'],
        how='left'
    )

    movies['genres'] = movies['genres'].fillna('').apply(lambda x: x.split('|') if x else [])

    # Backfill missing or blank director and actor names with "Unknown"
    movies['director_name'] = movies['director_name'].fillna('Unknown').replace('', 'Unknown')
    movies['actor_names'] = movies['actor_names'].fillna('Unknown').replace('', 'Unknown')

    # Reset index before saving
    movies.reset_index(drop=True, inplace=True)
    movies.to_pickle('data/enriched_movies.pkl')
    ratings.to_pickle('data/processed_ratings.pkl')

    print("Data ingestion complete, enriched movies saved.")
    print(f"Movies with director known: {(movies['director_name'] != 'Unknown').sum()} of {len(movies)}")
    print(f"Movies with actors known: {(movies['actor_names'] != 'Unknown').sum()} of {len(movies)}")

if __name__ == "__main__":
    load_and_process_movielens()
