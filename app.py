import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests
from io import BytesIO

# ========== FUNCIONES AUXILIARES ==========
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

# ========== CARGA DE DATOS ==========
st.title("🎬 Buscador Visual de Películas")

# Cargar features desde Google Drive
poster_features_url = "https://drive.google.com/uc?id=1RGzGutC4W721li3EsI2Tn9sltWeRkpb2"
features = pd.read_csv(poster_features_url)

# Cargar metadata desde archivo local o drive
metadata = pd.read_csv("MovieGenre.csv", encoding="ISO-8859-1")
features = pd.merge(features, metadata, left_on="tmdbId", right_on="imdbId")

# Limpieza de datos
to_exclude = ["tmdbId", "imdbId", "Title", "Genre", "Poster", "year"]
features["year"] = features["Title"].str.extract(r"\((\d{4})\)")
features["Genre"] = features["Genre"].str.split("|")
features = features.explode("Genre")

# Convertir columnas de características a float (para PCA)
numeric_features = features.drop(columns=[col for col in features.columns if col in to_exclude], errors="ignore")
numeric_features = numeric_features.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="any")

# ========== PCA y KMEANS ==========
pca = PCA(n_components=2)
X = pca.fit_transform(numeric_features)

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
features["cluster"] = kmeans.labels_

# ========== BUSCADOR VISUAL ==========
st.subheader("📤 Sube un póster para buscar películas similares")
uploaded_image = st.file_uploader("Sube un póster", type=["jpg", "png", "jpeg"])

if uploaded_image:
    st.image(uploaded_image, caption="Póster subido", width=200)
    idxs = find_similar_movies(uploaded_image, numeric_features.values)
    st.subheader("🔍 Películas similares")
    cols = st.columns(5)
    for i, idx in enumerate(idxs):
        movie = features.iloc[idx]
        if is_valid_image_path(movie.Poster):
            with cols[i % 5]:
                st.image(movie.Poster, caption=f"{movie.Title} ({movie.year})", width=160)

# ========== FILTRO POR GÉNERO ==========
st.subheader("🎬 Películas filtradas por género:")
genres = sorted(features["Genre"].dropna().unique())
selected_genre = st.selectbox("Selecciona género", genres)

if selected_genre:
    filtered = features[features["Genre"] == selected_genre].drop_duplicates(subset="tmdbId")
    cols = st.columns(5)
    for i, row in filtered.iterrows():
        if is_valid_image_path(row["Poster"]):
            with cols[i % 5]:
                st.image(row["Poster"], caption=f"{row['Title']} ({row['year']})", width=160)

# ========== FILTRO POR AÑO ==========
st.subheader("📅 Películas filtradas por año:")
years = sorted(features["year"].dropna().unique())
selected_year = st.selectbox("Selecciona año", years)

if selected_year:
    filtered = features[features["year"] == selected_year].drop_duplicates(subset="tmdbId")
    cols = st.columns(5)
    for i, row in filtered.iterrows():
        if is_valid_image_path(row["Poster"]):
            with cols[i % 5]:
                st.image(row["Poster"], caption=f"{row['Title']} ({row['year']})", width=160)



