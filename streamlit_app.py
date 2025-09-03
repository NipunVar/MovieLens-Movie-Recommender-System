import streamlit as st
import pandas as pd
import requests
import altair as alt

@st.cache_data(ttl=300)
def load_movies():
    return pd.read_pickle('data/processed_movies.pkl')

@st.cache_data(ttl=300)
def load_ratings():
    return pd.read_pickle('data/processed_ratings.pkl')

movies = load_movies()
ratings = load_ratings()
movie_titles = sorted(movies['title'].dropna().unique())

st.set_page_config(page_title="MovieLens Movie Recommender", layout="wide")

tab1, tab2 = st.tabs(["Recommendations", "Analytics"])

# Recommendations Tab
with tab1:
    st.title("Movie Recommender")
    selected_movie = st.selectbox("Choose a Movie:", movie_titles)

    if st.button("Get Recommendations"):
        with st.spinner("Getting recommendations..."):
            try:
                response = requests.get(
                    "http://127.0.0.1:8000/recommend/",
                    params={"title": selected_movie, "n": 10},
                    timeout=10,
                )
                if response.status_code != 200:
                    st.error(f"API error {response.status_code}: {response.text}")
                else:
                    data = response.json()
                    if "error" in data:
                        st.error(data["error"])
                    elif "recommendations" in data:
                        for rec in data["recommendations"]:
                            st.markdown(f"### {rec['title']}")
                            st.write(f"Genres: {rec['genres']}")
                            st.markdown("---")
                    else:
                        st.error("No recommendations found.")
            except Exception as e:
                st.error(f"Error: {e}")

# Analytics Tab
with tab2:
    st.title("Analytics")
    selected_movie_analytics = st.selectbox("Select a Movie for Analytics:", movie_titles, key="analytics")

    try:
        response = requests.get(
            "http://127.0.0.1:8000/recommend/",
            params={"title": selected_movie_analytics, "n": 10},
        )
        if response.status_code != 200:
            st.error(f"API error {response.status_code}: {response.text}")
        else:
            recommendations = response.json()
            if "error" in recommendations:
                st.error(recommendations["error"])
            else:
                rec_titles = [rec["title"] for rec in recommendations.get("recommendations", [])]
                rec_movies = movies[movies['title'].isin(rec_titles)]
                rec_movie_ids = rec_movies['movieId'].tolist()

                selected_movie_row = movies[movies['title'] == selected_movie_analytics]
                if not selected_movie_row.empty:
                    selected_movie_id = selected_movie_row.iloc[0]['movieId']
                    selected_count = ratings[ratings['movieId'] == selected_movie_id].shape[0]
                    selected_avg = ratings[ratings['movieId'] == selected_movie_id]['rating'].mean()
                else:
                    selected_movie_id = None
                    selected_count = 0
                    selected_avg = 0.0

                rec_counts = ratings[ratings['movieId'].isin(rec_movie_ids)]['movieId'].value_counts()
                rec_avg = ratings[ratings['movieId'].isin(rec_movie_ids)].groupby('movieId')['rating'].mean()

                counts_df = pd.DataFrame({
                    'Title': [selected_movie_analytics] + rec_movies['title'].tolist(),
                    'Rating Count': [selected_count] + [rec_counts.get(mid, 0) for mid in rec_movie_ids]
                })

                count_bar = alt.Chart(counts_df).mark_bar().encode(
                    x=alt.X('Rating Count:Q'),
                    y=alt.Y('Title:N', sort='-x'),
                    tooltip=['Title', 'Rating Count']
                ).properties(width=700, height=400)
                st.subheader("Rating Counts Comparison")
                st.altair_chart(count_bar)

                avg_df = pd.DataFrame({
                    'Title': [selected_movie_analytics] + rec_movies['title'].tolist(),
                    'Average Rating': [selected_avg] + [rec_avg.get(mid, 0) for mid in rec_movie_ids]
                })
                avg_chart = alt.Chart(avg_df).mark_circle(size=150).encode(
                    x=alt.X('Average Rating:Q', scale=alt.Scale(domain=[0, 5])),
                    y=alt.Y('Title:N', sort='-x'),
                    tooltip=['Title', 'Average Rating']
                ).properties(width=700, height=400)
                st.subheader("Average Ratings Comparison")
                st.altair_chart(avg_chart)
    except Exception as e:
        st.error(f"Error fetching analytics data: {e}")
