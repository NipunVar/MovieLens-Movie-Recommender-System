MovieLens Movie Recommender System

Project Overview
The MovieLens Movie Recommender System is a complete, end-to-end project designed to explore, build, and deploy a working recommendation engine using the well-known MovieLens dataset. This project showcases how modern recommendation systems work by combining content-based filtering and collaborative filtering to deliver highly personalized movie suggestions.
In today’s streaming-driven world, platforms like Netflix, Amazon Prime, and Disney+ rely on recommender systems to enhance user engagement. By predicting what a user might enjoy watching next, these systems save time, improve satisfaction, and keep audiences hooked. This project replicates that process on a smaller scale while also providing a fully interactive interface for exploration and learning.

The solution is built with a modular architecture:

Backend (FastAPI): Hosts the recommendation logic and exposes it through a lightweight, fast API.
Frontend (Streamlit): Provides an interactive user interface where users can select movies, view recommendations, and analyze insights.
Data & Models: Responsible for ingesting, processing, and transforming the raw MovieLens dataset into useful embeddings and recommendation outputs.

Beyond generating recommendations, the project includes visual analytics to help users explore rating trends, compare movies side by side, and dive into genre-specific insights. This ensures the application is not only useful but also engaging for movie lovers and learners alike.

What Has Been Accomplished So Far

1. Data Ingestion and Processing
Loaded the official MovieLens datasets, including movies.csv (movie metadata), ratings.csv (user ratings), and additional files like genome tags and genome scores for fine-grained features.
Cleaned and normalized the movie dataset by parsing genres, extracting release years from titles, and handling missing or malformed entries.
Created a user-item rating matrix for collaborative filtering and a movie-tag matrix for content-based embeddings.
Saved processed datasets as efficient .pkl (pickle) files (processed_movies.pkl, processed_ratings.pkl, movie_tag_matrix.pkl) for faster future loading.

2. Model Building
Constructed content-based embeddings for movies using metadata such as genres, directors, and actors. Dimensionality reduction with Truncated SVD helped convert sparse matrices into dense, meaningful vectors.
Developed a collaborative filtering model by applying SVD on the user-item matrix, learning latent factors that capture patterns of user behavior.
Combined embeddings with cosine similarity to measure closeness between movies. This allows the system to recommend titles that are semantically or behaviorally similar to the user’s chosen movie.
Built a robust mapping layer between datasets, ensuring that each movie ID consistently links with its embedding index and metadata across different steps.
Saved all important artifacts (embeddings, SVD factors, ID mappings) for reusability and reproducibility.

3. Backend API
Designed a FastAPI backend service exposing a /recommend/ endpoint.
The endpoint accepts a movie title and number of recommendations (n) as parameters, then returns a ranked list of similar movies.
Implemented comprehensive error handling to prevent crashes when a movie title is missing, misspelled, or absent from the dataset.
Ensured that the API is lightweight, responsive, and easy to scale.

4. Frontend Application
The frontend is built with Streamlit, offering a modern, user-friendly interface. Key features include:
Overview Section: Explains the project’s purpose and methodology, along with a dataset snapshot (total movies, users, ratings, and average ratings per movie). It also includes a timeline illustrating how recommender systems have evolved—from collaborative filtering to deep learning.
Recommendations Tab: Lets users pick a movie from a dropdown menu. By querying the backend API, the app displays top recommended movies with metadata such as genres, directors, actors, and similarity scores.
Analytics Dashboard: Provides deep insights into the dataset:
Rating Distribution: Histogram showing how users rated a chosen movie.
Top Movies by Rating Counts: Bar chart comparing the most frequently rated movies.
Top Movies by Average Ratings: Scatter chart showcasing audience favorites.
Genre Insights: Pick a genre to see the top-rated movies within it.
Side-by-Side Movie Comparison: Compare two movies on rating counts and averages.

Together, these features transform the system from a black-box recommender into a transparent, educational tool.