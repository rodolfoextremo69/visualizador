import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

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

def get_backup_poster(title, posters_df):
    match = posters_df[posters_df["title"].str.lower().str.strip() == title.lower().strip()]
    return match["Poster"].values[0] if not match.empty else None

def display_posters(df, posters_df, cols_per_row=5):
    cols = st.columns(cols_per_row)
    for i, (_, row) in enumerate(df.iterrows()):
        poster_url = row.get("Poster") or get_backup_poster(row["Title"], posters_df)
        if not is_valid_image_path(poster_url):
            poster_url = "https://via.placeholder.com/150x220?text=No+Image"
        caption = f"{row['Title']} ({row.get('year', '')})"
        with cols[i % cols_per_row]:
            st.image(poster_url, width=150, caption=caption)

# ========== CARGA DE ARCHIVOS ==========
st.sidebar.title("🎬 Filtros")

try:
    df_features = pd.read_csv("poster_features.csv")
    df_links = pd.read_csv("links.csv")
    df_metadata = pd.read_csv("MovieGenre.csv", encoding='ISO-8859-1')
    df_posters_clean = pd.read_csv("posters_clean.csv")
except Exception as e:
    st.error(f"❌ Error cargando archivos CSV: {e}")
    st.stop()

# ========== MERGE ==========
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
uploaded_image = st.file_uploader("O sube un póster", type=["jpg", "png", "jpeg"])

if uploaded_image:
    st.image(uploaded_image, caption="Póster subido", width=200)
    idxs = find_similar_movies(uploaded_image, numeric)
    st.subheader("Películas similares:")
    display_posters(df.iloc[idxs], df_posters_clean)
elif search_title.strip():
    result = df[df['Title'].str.lower().str.contains(search_title.lower())]
    if not result.empty:
        st.subheader(f"Resultados para: {search_title}")
        display_posters(result.drop_duplicates('tmdbId'), df_posters_clean)
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
    display_posters(filtered, df_posters_clean)
else:
    st.warning("No hay resultados para esos filtros.")




