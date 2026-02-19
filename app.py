import streamlit as st
import pickle
import pandas as pd
import requests

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔑 OMDb API key
API_KEY = "36377c72"

# ---------------- FETCH MOVIE POSTER ---------------- #
def fetch_poster(title):
    try:
        clean_title = title.split("(")[0].strip()
        url = f"http://www.omdbapi.com/?t={clean_title}&apikey={API_KEY}"
        data = requests.get(url, timeout=5).json()

        if data.get('Poster') and data['Poster'] != "N/A":
            return data['Poster']
    except:
        pass

    return "https://via.placeholder.com/300x450?text=No+Image"

# ---------------- LOAD DATA ---------------- #
movies = pickle.load(open('movies.pkl','rb'))

# ---------------- CREATE SIMILARITY ---------------- #
@st.cache_data
def create_similarity():
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    return cosine_similarity(vectors)

similarity = create_similarity()

# ---------------- RECOMMEND FUNCTION ---------------- #
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movies_list:
        title = movies.iloc[i[0]].title
        recommended_movies.append(title)
        recommended_posters.append(fetch_poster(title))

    return recommended_movies, recommended_posters

# ---------------- UI ---------------- #
st.title('🎬 Movie Recommender System')

selected_movie = st.selectbox(
    "Choose a movie",
    movies['title'].values
)

if st.button('Recommend'):
    with st.spinner("Finding best movies for you... 🎥"):
        names, posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    for i, col in enumerate([col1, col2, col3, col4, col5]):
        with col:
            st.image(posters[i])
            st.caption(names[i])
