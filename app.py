import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests

# =================== CONFIGURACIÓN =====================
st.set_page_config(layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .stTextInput, .stSelectbox, .stFileUploader {
            background-color: #333 !important;
            color: white !important;
        }
        .css-1d391kg, .css-18e3th9 {
            background-color: #111;
        }
        .st-bx, .st-c6, .st-cg {
            background-color: #111;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        body, .css-1v0mbdj, .css-10trblm, .css-1d391kg {
            color: white !important;
        }
        .stTextInput > div > input, .stSelectbox > div > div, .stMultiSelect > div {
            background-color: #222 !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# ===================== FUNCIONES =========================
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

def get_backup_poster(title):
    title_clean = title.lower().strip()
    matches = df_posters_clean[df_posters_clean["title"].str.lower().str.strip() == title_clean]
    if not matches.empty:
        return matches.iloc[0]["poster_path"]
    return None

def display_posters(df, cols_per_row=5):
    cols = st.columns(cols_per_row)
    i = 0
    for _, row in df.iterrows():
        poster_url = row['Poster'] if is_valid_image_path(row['Poster']) else get_backup_poster(row['Title'])
        caption = f"{row['Title']} ({row['year']})"
        img_url = poster_url if poster_url else "https://via.placeholder.com/150x220?text=No+Image"

        with cols[i % cols_per_row]:
            st.markdown(
                f"""<div style='text-align: center; color: white;'>
                    <img src='{img_url}' width='150'><br>
                    <span style='font-size: 13px;'>{caption}</span>
                </div>""", unsafe_allow_html=True
            )
        i += 1

# ===================== CARGA DE DATOS =========================
st.sidebar.title("🎬 Filtros")

# Desde Drive CSV reducido
url = "https://drive.google.com/uc?id=1RGzGutC4W721li3EsI2Tn9sltWeRkpb2"
features = pd.read_csv(url)

metadata = pd.read_csv("MovieGenre.csv", encoding='ISO-8859-1')
df_posters_clean = pd.read_csv("posters_clean.csv")

features = pd.merge(features, metadata, left_on='tmdbId', right_on='imdbId')
features['year'] = features['Title'].str.extract(r'\((\d{4})\)')
features['Genre'] = features['Genre'].str.split('|')
features = features.explode('Genre')

# Validar solo las columnas de features
numeric_features = features.filter(regex=r'^feat_', axis=1).dropna()

# PCA para clustering
pca = PCA(n_components=10)
X = pca.fit_transform(numeric_features)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# ===================== HEADER =========================
st.markdown("# 🎥 Buscador Visual de Películas")
st.markdown("## 📤 Sube un póster para buscar películas similares")
uploaded_image = st.file_uploader("Sube un póster", type=["jpg", "png", "jpeg"])

# ===================== BÚSQUEDA POR IMAGEN =========================
if uploaded_image:
    st.image(uploaded_image, caption="Póster subido", width=200)
    idxs = find_similar_movies(uploaded_image, numeric_features)
    st.markdown("### 🔍 Películas similares encontradas:")
    display_posters(features.iloc[idxs])

# ===================== BÚSQUEDA POR FILTROS =========================
st.markdown("---")
st.markdown("## 🎞️ Películas filtradas por género y/o año")

genres = sorted(features['Genre'].dropna().unique())
years = sorted(features['year'].dropna().unique())

selected_genres = st.sidebar.multiselect("Selecciona géneros", genres)
selected_years = st.sidebar.multiselect("Selecciona años", years)

filtered = features.copy()
if selected_genres:
    filtered = filtered[filtered['Genre'].isin(selected_genres)]
if selected_years:
    filtered = filtered[filtered['year'].isin(selected_years)]

filtered = filtered.drop_duplicates(subset='tmdbId')
if not filtered.empty:
    display_posters(filtered)
else:
    st.warning("No se encontraron películas con esos filtros.")



