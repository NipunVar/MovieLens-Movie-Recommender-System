import os
import pandas as pd
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer

def build_models():
    os.makedirs('models', exist_ok=True)
    movies = pd.read_pickle('data/enriched_movies.pkl')

    # Prepare genre data
    movies['genres'] = movies['genres'].apply(lambda gs: gs if isinstance(gs, list) else [])

    mlb = MultiLabelBinarizer()
    genre_matrix = pd.DataFrame(mlb.fit_transform(movies['genres']), columns=mlb.classes_, index=movies.index)

    n_components = min(50, genre_matrix.shape[1])
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    content_embeddings = svd.fit_transform(genre_matrix)

    # Save content embeddings
    joblib.dump(content_embeddings, 'models/content_embeddings.pkl')

    print("Genre-based content embeddings created and saved.")

if __name__ == "__main__":
    build_models()
