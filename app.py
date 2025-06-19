# app.py

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
movies = pd.read_csv("D:/movie_recommender/movies.csv/movies.csv")

# Handle missing genres
movies['genres'] = movies['genres'].fillna('')

# TF-IDF Vectorization of genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Build index mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Streamlit UI
st.title("🎬 Movie Recommendation System")
user_input = st.text_input("Enter a movie title (case-sensitive)", "Toy Story (1995)")

if st.button("Recommend"):
    recommendations = get_recommendations(user_input)
    if not recommendations.empty:
        st.write("Top 5 Recommendations:")
        for movie in recommendations:
            st.write(f"- {movie}")
    else:
        st.write("Movie not found in the dataset.")
