
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os

# URL del CSV en Google Drive
poster_features_id = '1RGzGutC4W721li3EsI2Tn9sltWeRkpb2'
poster_features_url = f'https://drive.google.com/uc?id={poster_features_id}'
features = pd.read_csv(poster_features_url)

# Cargar MovieGenre
movie_genres = pd.read_csv('MovieGenre.csv', encoding='ISO-8859-1')

# Unir por tmdbId e imdbId
features_with_genre = pd.merge(features, movie_genres, left_on='tmdbId', right_on='imdbId')
features_with_genre['year'] = features_with_genre['Title'].str.extract(r'\((\d{4})\)')
features_with_genre['Genre'] = features_with_genre['Genre'].str.split('|')
features_with_genre = features_with_genre.explode('Genre')

# Reducción dimensional
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_with_genre[features.columns[1:]])
kmeans = KMeans(n_clusters=5)
kmeans.fit(features_2d)
features_2d = pd.DataFrame(features_2d, columns=['x', 'y'])
features_2d['cluster'] = kmeans.labels_

st.title("Buscador Visual de Películas")
st.subheader("Visualización PCA + KMeans")
st.scatter_chart(features_2d[['x', 'y']])

def extract_features(image_path):
    image = Image.open(image_path)
    image = image.resize((128, 128))
    image = np.array(image)
    red_hist = np.histogram(image[:, :, 0], bins=256, range=(0, 255))[0]
    green_hist = np.histogram(image[:, :, 1], bins=256, range=(0, 255))[0]
    blue_hist = np.histogram(image[:, :, 2], bins=256, range=(0, 255))[0]
    return np.concatenate([red_hist, green_hist, blue_hist])

def find_similar_movies(image_path, feature_data, kmeans_model):
    image_features = extract_features(image_path)
    similarities = cosine_similarity([image_features], feature_data)
    return np.argsort(similarities[0])[:5]

def is_valid_image_path(image_path):
    if isinstance(image_path, str):
        return os.path.isfile(image_path) or image_path.startswith('http')
    return False

# === Subir imagen ===
st.header("Buscar películas por similitud visual")
uploaded_image = st.file_uploader("Sube un póster", type=["jpg", "png", "jpeg"])
if uploaded_image:
    st.image(uploaded_image, caption="Póster cargado", use_container_width=True)
    similar_movies = find_similar_movies(uploaded_image, features_with_genre, kmeans)
    st.subheader("Películas similares:")
    cols = st.columns(5)
    for i, idx in enumerate(similar_movies):
        with cols[i % 5]:
            movie = features_with_genre.iloc[idx]
            st.image(movie.Poster, use_container_width=True)
            st.markdown(f"**{movie.Title} ({movie.year})**")
            st.markdown(f"*Géneros:* {movie.Genre}")

# === Filtro por género ===
genres = features_with_genre['Genre'].dropna().unique()
selected_genre = st.selectbox("Selecciona género", sorted(genres))
filtered_movies = features_with_genre[features_with_genre['Genre'] == selected_genre]

st.subheader("Películas filtradas por género:")
cols = st.columns(5)
for i, movie in enumerate(filtered_movies.itertuples()):
    with cols[i % 5]:
        if is_valid_image_path(movie.Poster):
            st.image(movie.Poster, use_container_width=True)
        st.markdown(f"**{movie.Title} ({movie.year})**")
        st.markdown(f"*Géneros:* {movie.Genre}")

# === Filtro por año ===
years = features_with_genre['year'].dropna().unique()
selected_year = st.selectbox("Selecciona año", sorted(years))
filtered_by_year = features_with_genre[features_with_genre['year'] == selected_year]

st.subheader("Películas filtradas por año:")
cols = st.columns(5)
for i, movie in enumerate(filtered_by_year.itertuples()):
    with cols[i % 5]:
        if is_valid_image_path(movie.Poster):
            st.image(movie.Poster, use_container_width=True)
        st.markdown(f"**{movie.Title} ({movie.year})**")
        st.markdown(f"*Géneros:* {movie.Genre}")

