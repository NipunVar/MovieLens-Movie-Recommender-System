import pandas as pd

def load_and_process_data():
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')
    genome_tags = pd.read_csv('data/genome-tags.csv')
    genome_scores = pd.read_csv('data/genome-scores.csv')

    movies['genres'] = movies['genres'].fillna('').apply(lambda x: x.split('|') if x else [])

    # Reset index so movies rows correspond 1-to-1 with embeddings rows later
    movies = movies.reset_index(drop=True)
    movies['row_pos'] = movies.index  # essential for alignment

    # Create movie-tag pivot matrix (for embeddings)
    tag_matrix = genome_scores.pivot(index='movieId', columns='tagId', values='relevance').fillna(0)
    # Align tag_matrix rows with movies DataFrame (reindex by movieId as integer)
    tag_matrix = tag_matrix.reindex(movies['movieId']).fillna(0)

    # Save processed data
    movies.to_pickle('data/processed_movies.pkl')
    ratings.to_pickle('data/processed_ratings.pkl')
    tag_matrix.to_pickle('data/movie_tag_matrix.pkl')

    print("Data ingestion complete and saved.")

if __name__ == "__main__":
    load_and_process_data()
