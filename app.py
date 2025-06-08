import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(page_title="Buscador Visual de Películas", layout="wide")
st.title("🎬 Buscador Visual de Películas")

# ========= FUNCIONES =========
def extract_features(image_path):
    image = Image.open(image_path).resize((128, 128))
    image = np.array(image)
    r = np.histogram(image[:, :, 0], bins=256, range=(0, 255))[0]
    g = np.histogram(image[:, :, 1], bins=256, range=(0, 255))[0]
    b = np.histogram(image[:, :, 2], bins=256, range=(0, 255))[0]
    return np.concatenate([r, g, b])

def find_similar_movies(image_path, feature_data):
    query_features = extract_features(image_path)
    similarity = cosine_similarity([query_features], feature_data)[0]
    return np.argsort(similarity)[-5:][::-1]

def is_valid_image_path(path):
    return isinstance(path, str) and path.startswith("http")

# ========= CARGA DE DATOS =========
try:
    features = pd.read_csv("poster_features.csv")
    metadata = pd.read_csv("MovieGenre.csv", encoding="ISO-8859-1")
except Exception as e:
    st.error(f"Error cargando los archivos: {e}")
    st.stop()

# Unir metadatos
features = pd.merge(features, metadata, left_on="tmdbId", right_on="imdbId", how="inner")

# Extraer año y género
features["year"] = features["Title"].str.extract(r"\((\d{4})\)")
features["Genre"] = features["Genre"].str.split("|")
features = features.explode("Genre")

# Seleccionar solo columnas numéricas para PCA
numeric_cols = features.select_dtypes(include=["number"]).columns.tolist()
numeric_features = features[numeric_cols].fillna(0)

# PCA y Clustering
pca = PCA(n_components=2)
try:
    X = pca.fit_transform(numeric_features)
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)
except Exception as e:
    st.error(f"Error al aplicar PCA o clustering: {e}")
    st.stop()

# ========= SIMILITUD VISUAL =========
st.subheader("📤 Sube un póster para buscar películas similares")
uploaded_image = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if uploaded_image:
    st.image(uploaded_image, caption="Póster subido", width=220)
    try:
        idxs = find_similar_movies(uploaded_image, numeric_features)
        st.subheader("🎯 Películas similares:")
        cols = st.columns(5)
        for i, idx in enumerate(idxs):
            movie = features.iloc[idx]
            if is_valid_image_path(movie.Poster):
                with cols[i % 5]:
                    st.image(movie.Poster, caption=f"{movie.Title} ({movie.year})", width=140)
    except Exception as e:
        st.warning(f"No se pudo calcular similitud: {e}")

# ========= FILTRO POR GÉNERO =========
st.subheader("🎬 Películas filtradas por género:")
genres = sorted(features["Genre"].dropna().unique())
selected_genre = st.selectbox("Selecciona género", genres)

if selected_genre:
    filtered = features[features["Genre"] == selected_genre].drop_duplicates("tmdbId")
    cols = st.columns(5)
    for i, row in filtered.iterrows():
        if is_valid_image_path(row.Poster):
            with cols[i % 5]:
                st.image(row.Poster, caption=f"{row.Title} ({row.year})", width=140)

# ========= FILTRO POR AÑO =========
st.subheader("📅 Películas filtradas por año:")
years = sorted(features["year"].dropna().unique())
selected_year = st.selectbox("Selecciona año", years)

if selected_year:
    filtered = features[features["year"] == selected_year].drop_duplicates("tmdbId")
    cols = st.columns(5)
    for i, row in filtered.iterrows():
        if is_valid_image_path(row.Poster):
            with cols[i % 5]:
                st.image(row.Poster, caption=f"{row.Title} ({row.year})", width=140)



