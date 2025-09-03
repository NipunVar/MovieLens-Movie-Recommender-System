import pandas as pd
import joblib
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import os

def build_models():
    os.makedirs('models', exist_ok=True)

    movies = pd.read_pickle('data/processed_movies.pkl')
    ratings = pd.read_pickle('data/processed_ratings.pkl')
    tag_matrix = pd.read_pickle('data/movie_tag_matrix.pkl')

    # Build content embeddings on tag_matrix
    svd = TruncatedSVD(n_components=50, random_state=42)
    content_embeddings = svd.fit_transform(tag_matrix)

    # Map userId and movieId to zero-based codes for sparse matrix
    user_codes = ratings['userId'].astype('category').cat.codes
    movie_codes = ratings['movieId'].astype('category').cat.codes

    user_id_map = dict(enumerate(ratings['userId'].astype('category').cat.categories))
    movie_id_map = dict(enumerate(ratings['movieId'].astype('category').cat.categories))

    # Build sparse user-item rating matrix
    user_movie_matrix = csr_matrix((ratings['rating'], (user_codes, movie_codes)))

    # CF model using SVD on sparse matrix
    cf_svd = TruncatedSVD(n_components=50, random_state=42)
    user_factors = cf_svd.fit_transform(user_movie_matrix)
    item_factors = cf_svd.components_.T

    # Save models and id maps
    joblib.dump(content_embeddings, 'models/content_embeddings.pkl')
    joblib.dump(user_factors, 'models/user_factors.pkl')
    joblib.dump(item_factors, 'models/item_factors.pkl')
    joblib.dump(user_id_map, 'models/user_id_map.pkl')
    joblib.dump(movie_id_map, 'models/movie_id_map.pkl')

    print("Model building complete and saved.")

if __name__ == "__main__":
    build_models()
