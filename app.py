
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os

# Validar si una URL de imagen es válida
def is_valid_image_url(url):
    return isinstance(url, str) and url.startswith('http') and url.endswith(('.jpg', '.jpeg', '.png'))

# Cargar datos desde Google Drive
poster_features_url = 'https://drive.google.com/uc?id=1RGzGutC4W721li3EsI2Tn9sltWeRkpb2'
features = pd.read_csv(poster_features_url)

# Cargar MovieGenre.csv desde el repositorio (si es local, ajusta esto)
movie_genres = pd.read_csv('MovieGenre.csv', encoding='ISO-8859-1')

# Unir features con géneros
features_with_genre = pd.merge(features, movie_genres, left_on='tmdbId', right_on='imdbId')
features_with_genre['year'] = features_with_genre['Title'].str.extract(r'\((\d{4})\)')
features_with_genre['Genre'] = features_with_genre['Genre'].str.split('|')
features_with_genre = features_with_genre.explode('Genre')

# Mostrar algunos datos
st.write("Características cargadas:")
st.write(features_with_genre.head())

# Reducción PCA y clustering
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features[features.columns[1:]])
kmeans = KMeans(n_clusters=5)
kmeans.fit(features_2d)
features_2d = pd.DataFrame(features_2d, columns=['x', 'y'])
features_2d['cluster'] = kmeans.labels_

st.write("Distribución de películas en 2D:")
st.scatter_chart(features_2d[['x', 'y']])

# Función de extracción de features desde una imagen subida
def extract_features(image_path):
    image = Image.open(image_path).resize((128, 128))
    image = np.array(image)
    red_hist = np.histogram(image[:, :, 0], bins=256, range=(0, 255))[0]
    green_hist = np.histogram(image[:, :, 1], bins=256, range=(0, 255))[0]
    blue_hist = np.histogram(image[:, :, 2], bins=256, range=(0, 255))[0]
    return np.concatenate([red_hist, green_hist, blue_hist])

# Buscar películas similares por póster
def find_similar_movies(image_path, feature_data, kmeans_model):
    image_features = extract_features(image_path)
    similarities = cosine_similarity([image_features], feature_data)
    return np.argsort(similarities[0])[:5]

# Upload para búsqueda visual
st.header("Buscar películas por similitud visual")
uploaded_image = st.file_uploader("Sube un póster de película", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    st.image(uploaded_image, caption="Póster cargado", use_column_width=True)
    similar_movies = find_similar_movies(uploaded_image, features_with_genre[features.columns[1:]], kmeans)
    st.write("Películas similares:")
    for idx in similar_movies:
        movie = features_with_genre.iloc[idx]
        st.write(f"{movie.Title} ({movie.year})")
        if is_valid_image_url(movie.Poster):
            st.image(movie.Poster, width=200)
        else:
            st.write("❌ Imagen no disponible")

# Filtro por género
genres = features_with_genre['Genre'].dropna().unique()
selected_genre = st.selectbox('Selecciona género', sorted(genres))
filtered_movies = features_with_genre[features_with_genre['Genre'] == selected_genre]
filtered_movies = filtered_movies.drop_duplicates(subset='tmdbId')
st.write("Películas filtradas por género:")
cols = st.columns(4)
for i, movie in enumerate(filtered_movies.itertuples()):
    with cols[i % 4]:
        st.markdown(f"**{movie.Title} ({movie.year})**")
        if is_valid_image_url(movie.Poster):
            st.image(movie.Poster, width=180)
        else:
            st.write("📷 Imagen no disponible")

# Filtro por año
years = sorted(features_with_genre['year'].dropna().unique())
selected_year = st.selectbox('Selecciona año', years)
filtered_by_year = features_with_genre[features_with_genre['year'] == selected_year]
filtered_by_year = filtered_by_year.drop_duplicates(subset='tmdbId')
st.write("Películas filtradas por año:")
cols = st.columns(4)
for i, movie in enumerate(filtered_by_year.itertuples()):
    with cols[i % 4]:
        st.markdown(f"**{movie.Title} ({movie.year})**")
        if is_valid_image_url(movie.Poster):
            st.image(movie.Poster, width=180)
        else:
            st.write("📷 Imagen no disponible")
