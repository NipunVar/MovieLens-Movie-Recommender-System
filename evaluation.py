import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error

def evaluate():
    # Load preprocessed data and models
    ratings = pd.read_pickle('data/processed_ratings.pkl')
    user_factors = joblib.load('models/user_factors.pkl')
    item_factors = joblib.load('models/item_factors.pkl')
    user_id_map = joblib.load('models/user_id_map.pkl')
    movie_id_map = joblib.load('models/movie_id_map.pkl')
    movies = pd.read_pickle('data/processed_movies.pkl')

    # Create reverse mapping dictionaries for user and movie ids to factor indices
    user_id_to_idx = {v: k for k, v in user_id_map.items()}
    movie_id_to_idx = {v: k for k, v in movie_id_map.items()}

    # Map userId and movieId in ratings to model indices
    ratings['user_idx'] = ratings['userId'].map(user_id_to_idx)
    ratings['movie_idx'] = ratings['movieId'].map(movie_id_to_idx)

    # Drop rows with missing indices (unmapped)
    ratings = ratings.dropna(subset=['user_idx', 'movie_idx'])
    ratings['user_idx'] = ratings['user_idx'].astype(int)
    ratings['movie_idx'] = ratings['movie_idx'].astype(int)

    # Because movie_idx is model index, but you want to cross check with row_pos for safety (optional)
    # Example: map movie_idx to row_pos via movies DataFrame if needed (usually isn't)

    # Vectorized prediction: dot product of user and item factors
    preds = np.sum(user_factors[ratings['user_idx']] * item_factors[ratings['movie_idx']], axis=1)
    truths = ratings['rating'].values

    rmse = np.sqrt(mean_squared_error(truths, preds))

    print(f"Vectorized RMSE on known ratings: {rmse:.4f}")

if __name__ == "__main__":
    evaluate()
