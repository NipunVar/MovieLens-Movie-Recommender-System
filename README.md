MovieLens Movie Recommender System
Project Overview
This project builds a movie recommendation system using the MovieLens dataset. It combines collaborative filtering and content-based approaches to provide personalized movie suggestions based on user preferences and movie features.

What Has Been Accomplished So Far
1. Data Ingestion and Processing
Loaded MovieLens dataset files including movies, ratings, genome tags, and genome scores.

Processed movies dataset to parse genres and reset DataFrame indexing for perfect alignment.

Created a movie-tag matrix from genome scores to represent movies by their tag relevance features.

Saved processed datasets as pickle files (processed_movies.pkl, processed_ratings.pkl, movie_tag_matrix.pkl) for efficient loading.

2. Model Building
Built content-based embeddings for movies by performing dimensionality reduction (Truncated SVD) on the movie-tag matrix.

Built collaborative filtering model using Truncated SVD on a sparse user-item rating matrix.

Used categorical encoding of user and movie IDs to construct a sparse matrix suitable for SVD.

Saved SVD user and item factors, content embeddings, and ID mappings for later use.

3. Backend API
Developed a FastAPI backend exposing a /recommend/ endpoint.

This endpoint accepts a movie title and returns a list of similar movies based on cosine similarity of content embeddings.

Implemented robust mapping between movies DataFrame rows and embedding indices using a dedicated row_pos column to prevent index mismatches.

Handled errors so the API returns meaningful messages instead of crashing.

4. Frontend Application
Built a Streamlit web app with:

Movie selection dropdown.

Button to get recommendations by querying the backend API.

Display of recommended movies along with genres.

Analytics tab showing rating counts and average rating comparisons between the selected movie and its recommendations.

Fixed issues with matching movie IDs between recommendations and rating data to correctly display analytics charts.

Implemented error handling for API failures and user input errors.

Future Directions
Several enhancements can improve the recommender system, including:

Combining collaborative and content-based filtering for hybrid recommendations.

Adding user authentication and personalized recommendation history.

Including more movie metadata (directors, actors) for richer content analysis.

Improving UI/UX with filters, reviews, and explanations.

Training more advanced models using deep learning or matrix factorization techniques.

Deploying the app on cloud platforms for scalability.