import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os

# === 1. CARGA DE DATOS ===
# Google Drive raw link
poster_features_url = "https://drive.google.com/uc?id=1RGzGutC4W721li3EsI2Tn9sltWeRkpb2"
features = pd.read_csv(poster_features_url)

# CSV local o del repositorio para MovieGenre
movie_genres = pd.read_csv('MovieGenre.csv', encoding='ISO-8859-1')
posters_clean = pd.read_csv('posters_clean.csv')

# === 2. UNIÓN Y PROCESAMIENTO ===
df = pd.merge(features, movie_genres, left_on='tmdbId', right_on='imdbId')
df = pd.merge(df, posters_clean[['tmdbId', 'Poster']], on='tmdbId', how='left')
df['year'] = df['Title'].str.extract(r'\((\d{4})\)')
df['Genre'] = df['Genre'].str.split('|')
df = df.explode('Genre')

# === 3. CLUSTERING PARA VISUALIZACIÓN 2D ===
pca = PCA(n_components=2)
features_2d = pca.fit_transform(df[features.columns[1:]])
kmeans = KMeans(n_clusters=5)
df['x'], df['y'] = features_2d[:, 0], features_2d[:, 1]
df['cluster'] = kmeans.fit_predict(features_2d)

# === 4. FUNCIONES AUXILIARES ===
def extract_features(image_path):
    image = Image.open(image_path).resize((128, 128))
    image = np.array(image)
    red = np.histogram(image[:, :, 0], bins=256, range=(0, 255))[0]
    green = np.histogram(image[:, :, 1], bins=256, range=(0, 255))[0]
    blue = np.histogram(image[:, :, 2], bins=256, range=(0, 255))[0]
    return np.concatenate([red, green, blue])

def find_similar_movies(image_path, feature_data):
    query = extract_features(image_path)
    similarities = cosine_similarity([query], feature_data)[0]
    return np.argsort(similarities)[-5:][::-1]

def is_valid_image_path(path):
    return isinstance(path, str) and (path.startswith("http") or os.path.isfile(path))

# === 5. INTERFAZ STREAMLIT ===
st.title("🎬 Explorador de Películas por Similitud Visual")
uploaded_image = st.file_uploader("Sube un póster para buscar películas similares", type=["jpg", "png", "jpeg"])

if uploaded_image:
    st.image(uploaded_image, caption="Póster cargado", width=150)
    top_similares = find_similar_movies(uploaded_image, df[features.columns[1:]].values)

    st.subheader("Películas similares:")
    cols = st.columns(5)
    for i, idx in enumerate(top_similares):
        movie = df.iloc[idx]
        with cols[i % 5]:
            st.image(movie.Poster, caption=f"{movie.Title} ({movie.year})", width=150)

# === 6. FILTRO POR GÉNERO ===
st.subheader("🎞️ Películas filtradas por género:")
genre = st.selectbox("Selecciona género", sorted(df['Genre'].dropna().unique()))
filtered = df[df['Genre'] == genre].drop_duplicates('tmdbId')

cols = st.columns(5)
for i, movie in enumerate(filtered.itertuples()):
    if is_valid_image_path(movie.Poster):
        with cols[i % 5]:
            st.image(movie.Poster, caption=f"{movie.Title} ({movie.year})", width=150)

