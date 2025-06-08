import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os

# =================== FUNCIONES =====================

def extract_features(image_path):
    image = Image.open(image_path).resize((128, 128))
    image = np.array(image)
    r = np.histogram(image[:, :, 0], bins=256, range=(0, 255))[0]
    g = np.histogram(image[:, :, 1], bins=256, range=(0, 255))[0]
    b = np.histogram(image[:, :, 2], bins=256, range=(0, 255))[0]
    return np.concatenate([r, g, b])

def find_similar_movies(image_path, features_data):
    query_features = extract_features(image_path)
    similarity = cosine_similarity([query_features], features_data)[0]
    return np.argsort(similarity)[-5:][::-1]

def is_valid_image_path(path):
    return isinstance(path, str) and path.startswith("http")

# =================== CARGA DE DATOS =====================

st.title("🎬 Buscador Visual de Películas")

# Cargar desde Google Drive
url = "https://drive.google.com/uc?id=1RGzGutC4W721li3EsI2Tn9sltWeRkpb2"
features = pd.read_csv(url)

# Cargar metadata local
metadata = pd.read_csv("MovieGenre.csv", encoding='ISO-8859-1')
features = pd.merge(features, metadata, left_on='tmdbId', right_on='imdbId')
features['year'] = features['Title'].str.extract(r'\((\d{4})\)')
features['Genre'] = features['Genre'].str.split('|')
features = features.explode('Genre')

# =================== PCA + Clustering =====================

pca = PCA(n_components=2)
X = pca.fit_transform(features.iloc[:, 1:1+384])  # Evita columnas extra
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# =================== SUBIR PÓSTER =====================

uploaded_image = st.file_uploader("📤 Sube un póster de película", type=["jpg", "png", "jpeg"])

if uploaded_image:
    st.image(uploaded_image, caption="Póster subido", width=250)
    idxs = find_similar_movies(uploaded_image, features.iloc[:, 1:1+768])
    st.subheader("🔍 Películas similares")
    cols = st.columns(5)
    for i, idx in enumerate(idxs):
        movie = features.iloc[idx]
        if is_valid_image_path(movie.Poster):
            with cols[i % 5]:
                st.image(movie.Poster, caption=f"{movie.Title} ({movie.year})", width=160)

# =================== FILTRO POR GÉNERO =====================

genres = sorted(features['Genre'].dropna().unique())
selected_genre = st.selectbox("🎞️ Selecciona género", genres)

if selected_genre:
    filtered = features[features['Genre'] == selected_genre].drop_duplicates(subset="tmdbId")
    st.subheader("Películas filtradas por género:")
    cols = st.columns(5)
    for i, row in filtered.iterrows():
        if is_valid_image_path(row['Poster']):
            with cols[i % 5]:
                st.image(row['Poster'], caption=f"{row['Title']} ({row['year']})", width=160)

# =================== FILTRO POR AÑO =====================

years = sorted(features['year'].dropna().unique())
selected_year = st.selectbox("📅 Selecciona año", years)

if selected_year:
    filtered = features[features['year'] == selected_year].drop_duplicates(subset="tmdbId")
    st.subheader("Películas filtradas por año:")
    cols = st.columns(5)
    for i, row in filtered.iterrows():
        if is_valid_image_path(row['Poster']):
            with cols[i % 5]:
                st.image(row['Poster'], caption=f"{row['Title']} ({row['year']})", width=160)

