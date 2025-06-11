import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_similarity
import clip
import os

# Configuración
st.set_page_config(layout="wide")
st.title("🎬 Búsqueda Visual de Películas con CLIP")
st.write("Sube un póster y encuentra películas visualmente similares usando OpenAI CLIP.")

# Cargar modelo CLIP
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

# Cargar embeddings
@st.cache_data
def load_embeddings():
    with open("clip_posters.pkl", "rb") as f:
        data = pickle.load(f)
    return data["titles"], data["paths"], np.array(data["embeddings"])

titles, paths, embeddings = load_embeddings()

# Extraer vector de la imagen subida
def extract_clip_embedding(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_tensor).cpu().numpy()
        return embedding
    except UnidentifiedImageError:
        st.error("❌ Imagen no válida.")
        return None
    except Exception as e:
        st.error(f"❌ Error procesando la imagen: {e}")
        return None

# Subida de imagen
uploaded_image = st.file_uploader("📤 Sube un póster de película", type=["jpg", "jpeg", "png"])

if uploaded_image:
    st.image(uploaded_image, caption="📌 Imagen subida", width=300)
    query_embedding = extract_clip_embedding(uploaded_image)

    if query_embedding is not None:
        similarity = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(similarity)[-8:][::-1]

        st.subheader("🔍 Películas visualmente similares")
        cols = st.columns(4)
        for i, idx in enumerate(top_indices):
            if os.path.exists(paths[idx]):
                with cols[i % 4]:
                    st.image(paths[idx], caption=titles[idx], width=150)
                    st.caption(f"Similitud: {similarity[idx]:.2f}")
            else:
                st.warning(f"⚠️ Imagen no encontrada: {paths[idx]}")
else:
    st.info("⬆️ Por favor, sube una imagen para comenzar.")
