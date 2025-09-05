from fastapi import FastAPI, Query
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
import traceback
import numpy as np

app = FastAPI()

movies = pd.read_pickle('data/enriched_movies.pkl')
content_embeddings = joblib.load('models/content_embeddings.pkl')
movies['title'] = movies['title'].str.lower().str.strip()

@app.get("/recommend/")
def recommend(
    title: str,
    n: int = 10
):
    try:
        title_norm = title.lower().strip()
        filtered = movies[movies['title'] == title_norm]
        if filtered.empty:
            return {"recommendations": [], "error": "Movie not found."}

        row_pos = filtered.index[0]
        movie_emb = content_embeddings[row_pos].reshape(1, -1)
        sims = cosine_similarity(content_embeddings, movie_emb).flatten()
        sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)

        sims_filtered = [(i, sims[i]) for i in range(len(movies)) if i != row_pos]
        sims_filtered.sort(key=lambda x: x[1], reverse=True)

        recommendations = []
        for idx, score in sims_filtered[:n]:
            rec = movies.iloc[idx]
            genres = rec['genres']
            if isinstance(genres, list):
                genres_str = "|".join(genres)
            else:
                genres_str = ""
            recommendations.append({
                "title": rec['title'].title(),
                "genres": genres_str,
                "director": "Unknown",
                "actors": "Unknown",
                "similarity": round(float(score), 4)
            })

        return {"recommendations": recommendations}
    except Exception as e:
        print("Exception in recommend endpoint:", e)
        traceback.print_exc()
        return {"recommendations": [], "error": f"Server Exception: {str(e)}"}
