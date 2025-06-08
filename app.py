import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

st.set_page_config(layout="wide")

st.markdown("""
<style>
    body {
        background-color: #173759;
        color: #f0f0f0;
    }
    .stApp {
        background-color: #173759;
    }
    .css-1v0mbdj, .css-10trblm, .css-1d391kg, .stTextInput, .stSelectbox, .stMultiSelect, .stFileUploader {
        color: #f0f0f0 !important;
    }
    .stSelectbox div[role=combobox], .stTextInput input {
        color: #f0f0f0 !important;
        background-color: #222 !important;
    }
    .st-bx, .st-c6, .st-cg {
        background-color: #173759;
    }
</style>
""", unsafe_allow_html=True)

# ========== Carga de datos ==========
@st.cache_data
def load_data():
    features = pd.read_csv("final_movies_with_posters.csv", encoding='utf-8')
    clean_posters = pd.read_csv("posters_clean.csv")
    features.columns = features.columns.str.strip().str.lower()  # Normalizar nombres
    clean_posters.columns = clean_posters.columns.str.strip().str.lower()
    return features, clean_posters

features, clean_posters = load_data()

if 'title' not in features.columns or 'poster' not in features.columns:
    st.error("❌ El archivo no contiene columnas 'title' o 'poster'")
    st.stop()

# Eliminar columnas duplicadas
features = features.loc[:, ~features.columns.duplicated()]

# Añadir columnas derivadas
features['year'] = features['title'].str.extract(r'\((\d{4})\)')
features['genre'] = features['genre'].str.split('|')
features = features.explode('genre')

# Arreglo de imágenes que faltan
def resolve_poster_url(title):
    row = clean_posters[clean_posters['title'] == title]
    if not row.empty:
        return row.iloc[0]['poster']
    return "https://via.placeholder.com/150x220?text=No+Image"

features['poster'] = features.apply(
    lambda row: row['poster'] if isinstance(row['poster'], str) and row['poster'].startswith("http")
    else resolve_poster_url(row['title']), axis=1)

# ========== Extracción de características ==========
numeric_features = features.filter(regex=r'^feat_', axis=1).dropna()

# Clustering PCA + KMeans
pca = PCA(n_components=10)
X = pca.fit_transform(numeric_features)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# ========== Funciones ==========
def extract_features_from_image(image):
    img = Image.open(image).resize((128, 128))
    img = np.array(img)
    r = np.histogram(img[:, :, 0], bins=256, range=(0, 255))[0]
    g = np.histogram(img[:, :, 1], bins=256, range=(0, 255))[0]
    b = np.histogram(img[:, :, 2], bins=256, range=(0, 255))[0]
    return np.concatenate([r, g, b])

def find_similar_movies(image_path, features_data):
    query_features = extract_features_from_image(image_path)
    similarity = cosine_similarity([query_features], features_data)[0]
    return np.argsort(similarity)[-5:][::-1]

def show_movies(df):
    cols = st.columns(5)
    for i, row in df.iterrows():
        with cols[i % 5]:
            st.image(row['poster'], caption=f"{row['title']}\nGéneros: {row['genre']}\n({row['year']})", use_column_width=True)

# ========== Interfaz ==========
st.sidebar.header("🎬 Filtros")
genres = sorted(features['genre'].dropna().unique())
years = sorted(features['year'].dropna().unique())

selected_genres = st.sidebar.multiselect("Selecciona géneros", genres)
selected_years = st.sidebar.multiselect("Selecciona años", years)
search_title = st.sidebar.text_input("🔎 Buscar película por título")

st.title("🎥 Buscador Visual de Películas")
st.markdown("### 📤 Sube un póster para buscar películas similares")
uploaded_image = st.file_uploader("Sube un póster", type=["jpg", "png", "jpeg"])

if uploaded_image:
    st.image(uploaded_image, caption="Póster subido", width=200)
    idxs = find_similar_movies(uploaded_image, numeric_features)
    st.subheader("Películas similares encontradas")
    show_movies(features.iloc[idxs])

if search_title:
    st.subheader("Resultados por búsqueda textual")
    filtered = features[features['title'].str.contains(search_title, case=False)]
    if not filtered.empty:
        show_movies(filtered.head(10))
    else:
        st.info("No se encontró ninguna película con ese título")

st.markdown("---")
st.subheader("🎬 Películas por filtros")

filtered = features.copy()
if selected_genres:
    filtered = filtered[filtered['genre'].isin(selected_genres)]
if selected_years:
    filtered = filtered[filtered['year'].isin(selected_years)]

filtered = filtered.drop_duplicates(subset='tmdbid')
if not filtered.empty:
    show_movies(filtered.head(20))
else:
    st.info("Ninguna película coincide con los filtros seleccionados.")



