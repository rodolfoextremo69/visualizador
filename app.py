import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ========== CONFIG ==========
st.set_page_config(layout="wide")
st.title("🎬 Buscador Visual de Películas")
st.write("Busca por nombre, filtra por género o sube un póster para obtener recomendaciones.")

# ========== FUNCIONES ==========

def extract_image_features(image_file):
    try:
        image = Image.open(image_file).convert("RGB").resize((128, 128))
        image = np.array(image)
        r = np.histogram(image[:, :, 0], bins=35, range=(0, 255))[0]
        g = np.histogram(image[:, :, 1], bins=35, range=(0, 255))[0]
        b = np.histogram(image[:, :, 2], bins=35, range=(0, 255))[0]
        return np.concatenate([r, g, b])
    except UnidentifiedImageError:
        st.error("❌ Imagen inválida.")
        return None

def is_valid_url(url):
    return isinstance(url, str) and url.startswith("http")

def display_posters(indices, df_data, features_data, top_n=8):
    cols = st.columns(4)
    for i, idx in enumerate(indices):
        row = df_data.iloc[idx]
        title = row["title"]
        poster = row["Poster"]
        genres = row.get("genres", "")
        year = row.get("year", "")
        score = row.get("IMDB Score", "")
        with cols[i % 4]:
            if is_valid_url(poster):
                st.image(poster, width=150, caption=title)
            else:
                st.image("https://via.placeholder.com/150x220?text=Sin+imagen", width=150)
            st.caption(f"Géneros: {genres}")
            st.caption(f"Año: {year} | IMDB ⭐ {score}")
            if st.button(f"🔁 Ver similares {i}", key=f"sim_{i}_{title}"):
                query_vector = features_data[idx].reshape(1, -1)
                similarity = cosine_similarity(normalize(query_vector), normalize(features_data))[0]
                similar_idx = np.argsort(similarity)[-top_n:][::-1]
                st.subheader(f"🎯 Películas similares a: {title}")
                display_posters(similar_idx, df_data, features_data)

# ========== CARGA DE DATOS ==========
try:
    df_features = pd.read_csv("poster_features.csv")
    df_movies = pd.read_csv("movies_posters.csv")
except Exception as e:
    st.error(f"❌ Error cargando archivos CSV: {e}")
    st.stop()

# Unir por tmdbId
try:
    df = pd.merge(df_features, df_movies, on="tmdbId")
    df["year"] = df["title"].str.extract(r"\((\d{4})\)")
    df["genres_list"] = df["genres"].fillna("").apply(lambda x: x.split("|"))
    feature_cols = [col for col in df.columns if col.startswith("feat_")]
    features_array = df[feature_cols].values
except Exception as e:
    st.error(f"❌ Error combinando datos: {e}")
    st.stop()

# ========== INTERFAZ ==========
st.sidebar.title("🎛️ Filtros")
all_genres = sorted({g.strip() for sublist in df["genres_list"] for g in sublist if g})
all_years = sorted(df["year"].dropna().unique())

selected_genres = st.sidebar.multiselect("Filtrar por género", all_genres)
selected_years = st.sidebar.multiselect("Filtrar por año", all_years)

search_text = st.text_input("🔎 Buscar por nombre")
uploaded_file = st.file_uploader("📤 O sube un póster", type=["jpg", "jpeg", "png"])

# ========== FILTRADO / BÚSQUEDA ==========
filtered_df = df.copy()

if selected_genres:
    filtered_df = filtered_df[filtered_df["genres_list"].apply(lambda g: any(genre in g for genre in selected_genres))]

if selected_years:
    filtered_df = filtered_df[filtered_df["year"].isin(selected_years)]

if search_text.strip():
    filtered_df = filtered_df[filtered_df["title"].str.lower().str.contains(search_text.strip().lower())]

# ========== RESULTADO POR IMAGEN ==========
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📌 Póster subido", width=250)
    query_vec = extract_image_features(uploaded_file)

    if query_vec is not None:
        if query_vec.shape[0] != features_array.shape[1]:
            st.error(f"❌ Dimensión incompatible. Imagen: {query_vec.shape[0]}, Base: {features_array.shape[1]}")
            st.stop()

        similarity = cosine_similarity(normalize([query_vec]), normalize(features_array))[0]
        top_idxs = np.argsort(similarity)[-8:][::-1]
        st.subheader("🎯 Recomendaciones basadas en el póster:")
        display_posters(top_idxs, df, features_array)

# ========== RESULTADOS FILTRADOS / TEXTO ==========
elif not filtered_df.empty:
    st.subheader("📋 Resultados filtrados:")
    filtered_idxs = filtered_df.index.tolist()
    display_posters(filtered_idxs[:8], df, features_array)
else:
    st.info("🛈 No hay resultados. Usa filtros, búsqueda o sube una imagen.")
