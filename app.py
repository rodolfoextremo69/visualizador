import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os
import gdown  # para descargar archivos grandes desde Google Drive

# Crear carpeta para descargas temporales
if not os.path.exists("data"):
    os.makedirs("data")

# Descargar archivo desde Google Drive solo si no existe
poster_features_id = '1RGzGutC4W721li3EsI2Tn9sltWeRkpb2'
poster_features_url = f'https://drive.google.com/uc?export=download&id={poster_features_id}'
poster_features_path = 'data/poster_features.csv'
if not os.path.exists(poster_features_path):
    gdown.download(poster_features_url, poster_features_path, quiet=False)

# Leer el archivo descargado
features = pd.read_csv(poster_features_path)

# Cargar MovieGenre localmente (puedes hacer lo mismo con Drive si es necesario)
movie_genres = pd.read_csv('MovieGenre.csv', encoding='ISO-8859-1')

# Unir por tmdbId ↔ imdbId
features_with_genre = pd.merge(features, movie_genres, left_on='tmdbId', right_on='imdbId')

# Extraer año de 'Title'
features_with_genre['year'] = features_with_genre['Title'].str.extract(r'\((\d{4})\)')

# Procesar géneros
features_with_genre['Genre'] = features_with_genre['Genre'].str.split('|')
features_with_genre = features_with_genre.explode('Genre')

# Mostrar datos iniciales
st.write("Características cargadas:")
st.write(features_with_genre.head())

# PCA y Clustering
pca = PCA(n_components=2)
features_data_only = features_with_genre[features.columns[1:]]  # Excluir 'tmdbId'
features_2d = pca.fit_transform(features_data_only)

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(features_2d)

features_2d_df = pd.DataFrame(features_2d, columns=['x', 'y'])
features_2d_df['cluster'] = kmeans.labels_
st.write("Distribución de películas en 2D:")
st.scatter_chart(features_2d_df[['x', 'y']])

# Función: extraer características
def extract_features(image_path):
    image = Image.open(image_path).resize((128, 128))
    image = np.array(image)
    red_hist = np.histogram(image[:, :, 0], bins=256, range=(0, 255))[0]
    green_hist = np.histogram(image[:, :, 1], bins=256, range=(0, 255))[0]
    blue_hist = np.histogram(image[:, :, 2], bins=256, range=(0, 255))[0]
    return np.concatenate([red_hist, green_hist, blue_hist])

# Función: encontrar películas similares
def find_similar_movies(image_path, feature_data):
    image_features = extract_features(image_path)
    similarities = cosine_similarity([image_features], feature_data)[0]
    return np.argsort(similarities)[-5:][::-1]

# Función: validación de imagen
def is_valid_image_path(image_path):
    return isinstance(image_path, str) and (os.path.isfile(image_path) or image_path.startswith('http'))

# Interfaz
st.header("Buscar películas por similitud visual")
uploaded_image = st.file_uploader("Sube un póster de película", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Póster cargado", use_column_width=True)
    similar_indices = find_similar_movies(uploaded_image, features_data_only.values)

    st.write("Películas similares:")
    for idx in similar_indices:
        movie = features_with_genre.iloc[idx]
        st.write(f"{movie.Title} ({movie.Genre})")
        if 'Poster' in movie and is_valid_image_path(movie.Poster):
            st.image(movie.Poster, width=300)
        else:
            st.write("Imagen no disponible.")

# Filtro por género
st.subheader("Filtrado por género")
genres = sorted(features_with_genre['Genre'].dropna().unique())
selected_genre = st.selectbox('Selecciona género', genres)

filtered_movies = features_with_genre[features_with_genre['Genre'] == selected_genre]
for movie in filtered_movies.itertuples():
    st.write(f"{movie.Title} ({movie.Genre})")
    if is_valid_image_path(movie.Poster):
        st.image(movie.Poster, width=300)

# Filtro por año
st.subheader("Filtrado por año")
years = sorted(features_with_genre['year'].dropna().unique())
selected_year = st.selectbox('Selecciona año', years)

filtered_movies_year = features_with_genre[features_with_genre['year'] == selected_year]
for movie in filtered_movies_year.itertuples():
    st.write(f"{movie.Title} ({movie.year})")
    if is_valid_image_path(movie.Poster):
        st.image(movie.Poster, width=300)

