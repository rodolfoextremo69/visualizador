import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
from PIL import Image, UnidentifiedImageError
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ========== CONFIGURACIÓN ==========
st.set_page_config(layout="wide")
st.markdown("""
<style>
body { background-color: #121212; color: #E0E0E0; }
.stTextInput input, .stSelectbox div, .stMultiSelect div {
    background-color:#333 !important; color:#E0E0E0 !important;
}
.stButton>button, .stFileUploader>div {
    background-color:#222; color:#E0E0E0;
}
</style>
""", unsafe_allow_html=True)

# ========== FUNCIONES ==========

def extract_features(image_file):
    try:
        image = Image.open(image_file).convert("RGB").resize((128, 128))
        image = np.array(image)
        r = np.histogram(image[:, :, 0], bins=256, range=(0, 255))[0]
        g = np.histogram(image[:, :, 1], bins=256, range=(0, 255))[0]
        b = np.histogram(image[:, :, 2], bins=256, range=(0, 255))[0]
        return np.concatenate([r, g, b])
    except UnidentifiedImageError:
        st.error("❌ Imagen inválida.")
        return None

def is_valid_image_path(path):
    return isinstance(path, str) and path.strip() != "" and path != "0"

def get_backup_poster(title, posters_df):
    try:
        match = posters_df[posters_df["title"].str.lower().str.strip() == title.lower().strip()]
        if not match.empty:
            return match["Poster"].values[0]
    except:
        return None
    return None

def display_posters_with_buttons(df_view, posters_df, features_data, top_n=8, cols_per_row=5):
    df_view = df_view.reset_index(drop=True)
    cols = st.columns(cols_per_row)
    for i, (_, row) in enumerate(df_view.iterrows()):
        poster_url = row.get("Poster")
        if not is_valid_image_path(poster_url):
            poster_url = get_backup_poster(row["Title"], posters_df)
        if not is_valid_image_path(poster_url):
            poster_url = "https://via.placeholder.com/150x220?text=Sin+imagen"
        caption = f"{row['Title']} ({row.get('year', '')})"
        with cols[i % cols_per_row]:
            st.image(poster_url, width=150, caption=caption)
            if st.button(f"🔁 Ver similares {i}", key=f"sim_{i}"):
                imdb_id = row["imdbId"]
                base_row = df[df["imdbId"] == imdb_id].iloc[0]
                query_vector = features_data.loc[base_row.name].values.reshape(1, -1)
                similarity = cosine_similarity(normalize(query_vector), normalize(features_data.values))[0]
                idxs = np.argsort(similarity)[-top_n:][::-1]
                resultados = df.iloc[idxs].drop_duplicates('tmdbId')
                st.subheader(f"🎯 Películas similares a: {row['Title']}")
                display_posters_with_buttons(resultados, posters_df, features_data)

# ========== CARGA DE ARCHIVOS ==========
st.sidebar.title("🎬 Filtros")

if not os.path.exists("poster_features.csv"):
    if os.path.exists("poster_features.zip"):
        with zipfile.ZipFile("poster_features.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
    else:
        st.error("❌ No se encontró 'poster_features.csv' ni 'poster_features.zip'.")
        st.stop()

try:
    df_features = pd.read_csv("poster_features.csv")
    df_links = pd.read_csv("links.csv")
    df_metadata = pd.read_csv("MovieGenre.csv", encoding='ISO-8859-1')
    df_posters_clean = pd.read_csv("posters_clean.csv")
except Exception as e:
    st.error(f"❌ Error cargando archivos CSV: {e}")
    st.stop()

# ========== MERGE Y PROCESAMIENTO ==========
try:
    df = pd.merge(df_features, df_links, on="tmdbId")
    df = pd.merge(df, df_metadata, on="imdbId")
    df['year'] = df['Title'].str.extract(r'\((\d{4})\)')
    df['Genre'] = df['Genre'].str.split('|')
    df = df.explode('Genre')
    numeric = df.filter(regex='^feat_', axis=1).dropna()

    pca = PCA(n_components=10)
    X = pca.fit_transform(numeric)
    kmeans = KMeans(n_clusters=5, random_state=42).fit(X)

    genres = sorted(df['Genre'].dropna().unique())
    years = sorted(df['year'].dropna().unique())
except Exception as e:
    st.error(f"❌ Error procesando datos: {e}")
    st.stop()

# ========== INTERFAZ PRINCIPAL ==========
st.markdown("# 🎥 Buscador Visual de Películas")
st.markdown("### 📤 Sube un póster o escribe un nombre para buscar")

search_title = st.text_input("🔎 Buscar por nombre")
uploaded_image = st.file_uploader("O sube un póster", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="📌 Póster subido", width=250)
    try:
        query_vec = extract_features(uploaded_image)
        if query_vec is None:
            st.stop()
        if query_vec.shape[0] != numeric.shape[1]:
            st.error(f"❌ Dimensión incompatible. Imagen: {query_vec.shape[0]}, Base: {numeric.shape[1]}")
            st.stop()
        similarity = cosine_similarity(normalize([query_vec]), normalize(numeric.values))[0]
        idxs = np.argsort(similarity)[-8:][::-1]
        resultados = df.iloc[idxs].drop_duplicates('tmdbId')
        st.subheader("🎯 Recomendaciones basadas en el póster")
        display_posters_with_buttons(resultados, df_posters_clean, numeric)
    except Exception as e:
        st.error(f"❌ Error procesando la imagen: {e}")
        st.stop()

elif search_title.strip():
    result = df[df['Title'].str.lower().str.contains(search_title.lower())]
    if not result.empty:
        st.subheader(f"🔍 Resultados para: {search_title}")
        display_posters_with_buttons(result.drop_duplicates('tmdbId'), df_posters_clean, numeric)
    else:
        st.warning("No se encontraron coincidencias.")

# ========== FILTROS ==========
st.markdown("---")
st.subheader("🎞️ Películas por género y año")

sel_genres = st.sidebar.multiselect("Géneros", genres)
sel_years = st.sidebar.multiselect("Años", years)

filtered = df.copy()
if sel_genres:
    filtered = filtered[filtered['Genre'].isin(sel_genres)]
if sel_years:
    filtered = filtered[filtered['year'].isin(sel_years)]

filtered = filtered.drop_duplicates('tmdbId')
if not filtered.empty:
    display_posters_with_buttons(filtered, df_posters_clean, numeric)
else:
    st.warning("No hay resultados para esos filtros.")

