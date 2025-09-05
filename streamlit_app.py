import streamlit as st
import pandas as pd
import requests
import altair as alt
import re

st.set_page_config(page_title="MovieLens Movie Recommender", layout="wide")

# Load data
movies = pd.read_pickle('data/enriched_movies.pkl')
ratings = pd.read_pickle('data/processed_ratings.pkl')
movies_csv = pd.read_csv('data/movies.csv')
movie_titles = sorted(movies['title'].dropna().unique())

def normalize_title(title):
    if pd.isna(title):
        return ""
    s = title.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Overview", "Recommendations", "Analytics"])

# ---------------- Overview ----------------
if page == "Overview":
    st.title("MovieLens Movie Recommender – Project Overview")

    overview_text = """
    The MovieLens Movie Recommender System is a data-driven application designed to provide personalized movie 
    recommendations and insightful analytics based on the widely used MovieLens dataset. The primary objective of this 
    project is to showcase how recommendation systems can enhance user experiences by predicting preferences and 
    offering tailored suggestions.

    At its core, the system leverages a combination of **content-based filtering** and **collaborative filtering** 
    approaches. Content-based filtering analyzes movie features such as genres, directors, and actors to find 
    similarities between movies, ensuring that recommendations align closely with the characteristics of a selected 
    film. Collaborative filtering, on the other hand, examines user rating patterns across the dataset to identify 
    movies that users with similar tastes have enjoyed. Together, these techniques create a more robust and accurate 
    recommendation engine.

    The application has been built using **Streamlit** for the frontend interface, providing an interactive and user-
    friendly platform for exploring recommendations and visualizing analytics. On the backend, the system uses a 
    combination of **Python, Pandas, and machine learning models**, along with a lightweight **API service** built 
    with FastAPI to deliver recommendations in real-time. Movie metadata, including enriched information like directors 
    and actors, is incorporated to improve recommendation quality and enhance the overall browsing experience.

    A major feature of this project is the **analytics dashboard**, which allows users to dive deeper into the dataset. 
    Users can view rating distributions for specific movies, compare rating counts and averages across the most popular 
    titles, and analyze top-rated films by genre. These analytics not only provide transparency into how movies are 
    rated by different users but also highlight broader trends in audience preferences. For instance, users can easily 
    identify genres with consistently higher average ratings or discover hidden gems that may not be mainstream but 
    have high audience appreciation.

    The project also emphasizes **data preprocessing and enrichment**. Since the MovieLens dataset primarily provides 
    movie IDs, titles, genres, and user ratings, additional steps were taken to normalize titles, clean genres, and 
    incorporate supplementary metadata. This enrichment makes the system more practical and informative for end users, 
    allowing them to see not just ratings but also contextual details like directors and actors that may influence 
    viewing choices.

    From a practical perspective, this system demonstrates the value of recommendation engines in real-world 
    applications. Streaming platforms like Netflix, Amazon Prime, and Disney+ rely heavily on similar techniques to 
    keep users engaged and provide a seamless content discovery experience. By replicating these ideas on a smaller 
    scale with open datasets, this project provides both a learning opportunity and a proof-of-concept that can be 
    extended further.

    In summary, the MovieLens Movie Recommender System is more than just a tool for suggesting movies. It is a complete 
    project that combines data engineering, machine learning, and interactive visualization to create an engaging and 
    informative application. Whether you are a casual movie enthusiast looking for your next watch, or a data science 
    learner interested in understanding recommender systems, this project offers a practical and insightful experience 
    into the world of personalized recommendations.
    """

    st.markdown(overview_text)

    # Dataset Snapshot
    st.subheader("📊 Dataset Snapshot")
    total_movies = movies_csv['movieId'].nunique()
    total_users = ratings['userId'].nunique()
    total_ratings = len(ratings)
    avg_ratings_per_movie = ratings.groupby('movieId').size().mean()

    st.metric("Number of Movies", total_movies)
    st.metric("Number of Users", total_users)
    st.metric("Total Ratings", total_ratings)
    st.metric("Avg Ratings per Movie", f"{avg_ratings_per_movie:.2f}")

    # Timeline of recommender systems
    st.subheader("🖼 Evolution of Recommender Systems (Timeline)")
    timeline = pd.DataFrame({
        "Year": [1990, 2000, 2010, 2020],
        "Method": [
            "Collaborative Filtering – Early methods based on user-item matrices",
            "Content-Based Filtering – Leveraging metadata like genres & actors",
            "Hybrid Models – Combining collaborative + content approaches",
            "Deep Learning – Neural networks for embeddings & sequence models"
        ]
    })
    timeline_chart = alt.Chart(timeline).mark_line(point=True).encode(
        x="Year:O",
        y=alt.Y("Year:O", axis=None),
        tooltip=["Year", "Method"]
    ).properties(title="Timeline of Recommender System Evolution")
    st.altair_chart(timeline_chart, use_container_width=True)

# ---------------- Recommendations ----------------
elif page == "Recommendations":
    st.title("Movie Recommender")
    selected_movie = st.selectbox("Choose a Movie:", movie_titles)
    if st.button("Get Recommendations", key='recs'):
        params = {"title": selected_movie, "n": 10}
        try:
            response = requests.get("http://127.0.0.1:8000/recommend/", params=params, timeout=10)
            data = response.json()
            if "recommendations" in data and data["recommendations"]:
                for rec in data["recommendations"]:
                    st.markdown(f"### {rec['title']}")
                    genres = rec.get('genres', '')
                    if not genres or genres.strip().lower() == "(no genres listed)":
                        st.write("Genres: N/A")
                    else:
                        st.write(f"Genres: {genres}")
                    if rec.get('director') and rec['director'].strip().lower() not in ['', 'unknown']:
                        st.write(f"Director: {rec['director']}")
                    if rec.get('actors') and rec['actors'].strip().lower() not in ['', 'unknown']:
                        st.write(f"Actors: {rec['actors']}")
                    st.write(f"Similarity Score: {rec['similarity']}")
                    st.markdown("---")
            else:
                st.info("No recommendations found.")
        except Exception as e:
            st.error(f"Error querying API: {e}")

# ---------------- Analytics ----------------
elif page == "Analytics":
    st.title("Analytics")
    selected_movie_a = st.selectbox("Select a Movie for Analytics:", movie_titles, key='analytics')
    movie_info = movies[movies['title'] == selected_movie_a].iloc[0]
    st.markdown(f"**Title:** {movie_info['title'].title()}")
    st.write(f"Genres: {'|'.join(movie_info['genres']) if isinstance(movie_info['genres'], list) else movie_info['genres']}")

    # Map title to movieId
    movies_csv['title_norm'] = movies_csv['title'].apply(normalize_title)
    selected_movie_norm = normalize_title(selected_movie_a)
    try:
        movie_id = movies_csv.set_index('title_norm').loc[selected_movie_norm, 'movieId']
    except Exception:
        movie_id = None

    if movie_id is not None:
        selected_ratings = ratings[ratings['movieId'] == movie_id]
    else:
        selected_ratings = pd.DataFrame()

    if len(selected_ratings) > 0:
        st.write(f"Number of ratings: {len(selected_ratings)}")
        st.write(f"Average rating: {selected_ratings['rating'].mean():.2f}")
        rating_counts = selected_ratings['rating'].value_counts().sort_index()
        rating_df = pd.DataFrame({'Rating': rating_counts.index, 'Count': rating_counts.values})

        bar_chart = alt.Chart(rating_df).mark_bar().encode(
            x=alt.X('Rating:O', sort='ascending'),
            y='Count:Q',
            tooltip=['Rating', 'Count']
        ).properties(width=500, height=300, title="Rating Distribution for Selected Movie")
        st.altair_chart(bar_chart, use_container_width=True)

    # Aggregate stats
    movie_stats = ratings.groupby('movieId').agg(
        rating_count=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    movie_stats = movie_stats.merge(movies_csv[['movieId', 'title']], on='movieId', how='left')

    # Rating Counts Comparison
    st.subheader("Rating Counts Comparison")
    top_counts = movie_stats.sort_values('rating_count', ascending=False).head(10)
    if movie_id is not None and movie_id not in top_counts['movieId'].values:
        top_counts = pd.concat([top_counts, movie_stats[movie_stats['movieId'] == movie_id]])
    bar_counts = top_counts.sort_values('rating_count', ascending=True)
    bar_chart_counts = alt.Chart(bar_counts).mark_bar().encode(
        x=alt.X('rating_count:Q', title='Rating Count'),
        y=alt.Y('title:N', sort='-x', title='Title'),
        color=alt.condition(alt.datum.movieId == movie_id, alt.value('orange'), alt.value('steelblue')),
        tooltip=['title:N', 'rating_count:Q']
    )
    st.altair_chart(bar_chart_counts, use_container_width=True)

    # Average Ratings Comparison
    st.subheader("Average Ratings Comparison")
    top_avgs = movie_stats[movie_stats['rating_count'] >= 10].sort_values('avg_rating', ascending=False).head(10)
    if movie_id is not None and movie_id not in top_avgs['movieId'].values:
        top_avgs = pd.concat([top_avgs, movie_stats[movie_stats['movieId'] == movie_id]])
    scatter_avgs = top_avgs.sort_values('avg_rating', ascending=True)
    scatter_chart = alt.Chart(scatter_avgs).mark_circle(size=150).encode(
        x=alt.X('avg_rating:Q', title='Average Rating'),
        y=alt.Y('title:N', sort='-x', title='Title'),
        color=alt.condition(alt.datum.movieId == movie_id, alt.value('orange'), alt.value('dodgerblue')),
        tooltip=['title:N', 'avg_rating:Q', 'rating_count:Q']
    )
    st.altair_chart(scatter_chart, use_container_width=True)

    # Top Rated Movies by Genre
    st.subheader("Top Rated Movies by Genre")
    all_genres = sorted(set(
        g for gs in movies['genres'] if isinstance(gs, list) for g in gs if g and g.lower() != "(no genres listed)"
    ))
    selected_genre = st.selectbox("Pick a genre:", all_genres)
    genre_movies = movies[movies['genres'].apply(lambda gs: selected_genre in gs if isinstance(gs, list) else False)]
    if not genre_movies.empty:
        genre_movie_ids = movies_csv[movies_csv['title'].isin(genre_movies['title'])]['movieId']
        ratings_in_genre = ratings[ratings['movieId'].isin(genre_movie_ids)]
        genre_stats = ratings_in_genre.groupby('movieId').agg(
            avg_rating=('rating', 'mean'),
            rating_count=('rating', 'count')
        ).reset_index()
        genre_stats = genre_stats.merge(movies_csv[['movieId', 'title']], on='movieId', how='left')
        genre_stats = genre_stats[genre_stats['rating_count'] >= 10]
        top10 = genre_stats.sort_values('avg_rating', ascending=False).head(10)
        if len(top10) > 0:
            bar = alt.Chart(top10).mark_bar().encode(
                x=alt.X('avg_rating:Q', title='Average Rating'),
                y=alt.Y('title:N', sort='-x', title='Movie'),
                tooltip=['title', 'avg_rating', 'rating_count']
            ).properties(title=f"Top Rated {selected_genre} Movies (min 10 ratings)")
            st.altair_chart(bar, use_container_width=True)
        else:
            st.info("No movies found with 10+ ratings in this genre.")

    # Genre Popularity Over Time
    st.subheader("📈 Genre Popularity Over Time")
    movies_year = movies_csv.copy()
    movies_year['year'] = movies_year['title'].str.extract(r'\((\d{4})\)').astype(float)
    movies_year = movies_year.dropna(subset=['year'])
    movies_year['genres'] = movies_year['genres'].fillna('')
    movies_year = movies_year.assign(genre=movies_year['genres'].str.split('|')).explode('genre')
    genre_year_merge = movies_year.merge(ratings, on='movieId', how='inner')
    genre_trend = genre_year_merge.groupby(['year', 'genre']).size().reset_index(name='count')
    selected_genre_trend = st.selectbox("Select a genre for trend:", genre_trend['genre'].unique())
    trend_data = genre_trend[genre_trend['genre'] == selected_genre_trend]
    line_chart = alt.Chart(trend_data).mark_line().encode(
        x=alt.X('year:O', title="Year"),
        y=alt.Y('count:Q', title="Number of Ratings"),
        tooltip=['year', 'count']
    ).properties(title=f"Trend of {selected_genre_trend} Ratings Over Time")
    st.altair_chart(line_chart, use_container_width=True)

    # Compare Movies Side-by-Side
    st.subheader("🎬 Compare Movies Side-by-Side")
    movie1 = st.selectbox("Select first movie:", movie_titles, key="cmp1")
    movie2 = st.selectbox("Select second movie:", movie_titles, key="cmp2")
    def get_movie_stats(title):
        norm = normalize_title(title)
        try:
            mid = movies_csv.set_index('title_norm').loc[norm, 'movieId']
            r = ratings[ratings['movieId'] == mid]
            return {"title": title, "count": len(r), "avg": r['rating'].mean() if len(r) > 0 else 0}
        except Exception:
            return {"title": title, "count": 0, "avg": 0}
    stats1, stats2 = get_movie_stats(movie1), get_movie_stats(movie2)
    cmp_df = pd.DataFrame([stats1, stats2])
    cmp_chart = alt.Chart(cmp_df).mark_bar().encode(
        x='title:N', y='avg:Q', color='title:N',
        tooltip=['title', 'avg', 'count']
    ).properties(title="Average Rating Comparison")
    st.altair_chart(cmp_chart, use_container_width=True)
