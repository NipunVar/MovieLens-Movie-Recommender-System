from fastapi import FastAPI
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

movies = pd.read_pickle('data/processed_movies.pkl')
content_embeddings = joblib.load('models/content_embeddings.pkl')

@app.get("/recommend/")
def recommend(title: str, n: int = 10):
    try:
        filtered = movies[movies['title'].str.lower().str.strip() == title.lower().strip()]
        if filtered.empty:
            return {"recommendations": [], "error": "Movie not found."}

        row_pos = filtered.iloc[0]['row_pos']
        if row_pos < 0 or row_pos >= content_embeddings.shape[0]:
            return {"recommendations": [], "error": "Movie index mapping error."}

        movie_emb = content_embeddings[row_pos].reshape(1, -1)
        sims = cosine_similarity(content_embeddings, movie_emb).flatten()
        ranked = sims.argsort()[::-1]

        recs = []
        count = 0
        for i in ranked:
            if i == row_pos:
                continue
            rec_movie = movies.iloc[i]
            recs.append({
                "title": rec_movie['title'],
                "genres": "|".join(rec_movie['genres']),
            })
            count += 1
            if count >= n:
                break

        return {"recommendations": recs}

    except Exception as e:
        return {"recommendations": [], "error": f"Server Exception: {str(e)}"}
