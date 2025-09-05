import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate():
    ratings = pd.read_pickle("data/processed_ratings.pkl")
    user_factors = joblib.load("models/user_factors.pkl")
    item_factors = joblib.load("models/item_factors.pkl")
    user_id_map = joblib.load("models/user_id_map.pkl")
    movie_id_map = joblib.load("models/movie_id_map.pkl")

    user_id_to_idx = {v: k for k, v in user_id_map.items()}
    movie_id_to_idx = {v: k for k, v in movie_id_map.items()}

    ratings['user_idx'] = ratings['userId'].map(user_id_to_idx)
    ratings['movie_idx'] = ratings['movieId'].map(movie_id_to_idx)

    ratings_eval = ratings.dropna(subset=['user_idx', 'movie_idx'])
    ratings_eval['user_idx'] = ratings_eval['user_idx'].astype(int)
    ratings_eval['movie_idx'] = ratings_eval['movie_idx'].astype(int)

    pred_ratings = np.sum(user_factors[ratings_eval['user_idx']] * item_factors[ratings_eval['movie_idx']], axis=1)
    rmse = np.sqrt(mean_squared_error(ratings_eval['rating'], pred_ratings))
    print(f"Evaluation RMSE: {rmse:.4f}")

if __name__ == "__main__":
    evaluate()
