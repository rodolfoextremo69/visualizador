import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os
# Cargar poster_features.csv desde Google Drive
poster_features_id = '1RGzGutC4W721li3EsI2Tn9sltWeRkpb2'
poster_features_url = f'https://drive.google.com/uc?id={poster_features_id}'
features = pd.read_csv(poster_features_url)

# Cargar las características visuales de las imágenes (previamente calculadas)
features = pd.read_csv('poster_features.csv')  # Ajusta la ruta a tus características

# Cargar los géneros de las películas desde MovieGenre.csv
movie_genres = pd.read_csv('MovieGenre.csv', encoding='ISO-8859-1')

# Unir los archivos usando 'tmdbId' e 'imdbId'
features_with_genre = pd.merge(features, movie_genres, left_on='tmdbId', right_on='imdbId')

# Extraer el año de la columna 'Title'
features_with_genre['year'] = features_with_genre['Title'].str.extract(r'\((\d{4})\)')

# Separar los géneros por el delimitador '|'
features_with_genre['Genre'] = features_with_genre['Genre'].str.split('|')

# Explode para tener una fila por género
features_with_genre = features_with_genre.explode('Genre')

# Verifica las primeras filas para asegurarte que los datos están bien cargados
st.write("Características cargadas:")
st.write(features_with_genre.head())

# Reducir la dimensionalidad con PCA (2D para visualización)
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_with_genre[features.columns[1:]])

# Aplicar K-means para agrupar las películas
kmeans = KMeans(n_clusters=5)  # Por ejemplo, 5 clusters
kmeans.fit(features_2d)  # Asegúrate de ajustar el modelo antes de acceder a labels_

features_2d = pd.DataFrame(features_2d, columns=['x', 'y'])
features_2d['cluster'] = kmeans.labels_  # Ahora puedes acceder a labels_

# Mostrar los clusters en el espacio bidimensional
st.write("Distribución de películas en 2D:")
st.scatter_chart(features_2d[['x', 'y']])

# Función para extraer características de la imagen cargada
def extract_features(image_path):
    image = Image.open(image_path)
    image = image.resize((128, 128))  # Redimensionar la imagen para estandarizarla
    image = np.array(image)  # Convertir la imagen a un array de numpy
    
    # Extraer características simples: histograma de colores en RGB
    red_hist = np.histogram(image[:, :, 0], bins=256, range=(0, 255))[0]
    green_hist = np.histogram(image[:, :, 1], bins=256, range=(0, 255))[0]
    blue_hist = np.histogram(image[:, :, 2], bins=256, range=(0, 255))[0]
    
    # Concatenar los histogramas para obtener un vector de características
    features = np.concatenate([red_hist, green_hist, blue_hist])
    return features

# Función para encontrar las películas más similares
def find_similar_movies(image_path, feature_data, kmeans_model):
    image_features = extract_features(image_path)  # Extraer características de la imagen subida
    similarities = cosine_similarity([image_features], feature_data)  # Comparar con los datos existentes
    similar_movies = np.argsort(similarities[0])[:5]  # Obtener las 5 películas más similares
    return similar_movies

# Función para verificar si la ruta de la imagen es válida
def is_valid_image_path(image_path):
    if isinstance(image_path, str):
        return os.path.isfile(image_path) or image_path.startswith('http')
    return False  # Si no es un string válido, retornamos False

# Mostrar el visualizador para cargar una imagen
st.header("Buscar películas por similitud visual")
uploaded_image = st.file_uploader("Sube un póster de película", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Póster cargado", use_column_width=True)
    
    # Llamar a la función para encontrar películas similares
    similar_movies = find_similar_movies(uploaded_image, features_with_genre, kmeans)
    
    # Mostrar las películas similares
    st.write("Películas similares:")
    for movie in similar_movies:
        st.write(f"Película {movie}")
        # Verificar si la ruta de la imagen es válida antes de mostrarla
        poster_path = features_with_genre['Poster'][movie]  # Ajusta esta columna si es necesario
        if is_valid_image_path(poster_path):
            st.image(poster_path, use_column_width=True)  # Mostrar la imagen del póster
        else:
            st.write(f"No se pudo cargar la imagen para la película {movie}")

# Filtro por género
genres = features_with_genre['Genre'].unique()  # Obtener géneros únicos (ajustado si la columna se llama 'genres')
selected_genre = st.selectbox('Selecciona género', genres)

# Filtrar películas por género
filtered_movies = features_with_genre[features_with_genre['Genre'] == selected_genre]  # Ajusta el nombre de la columna si es necesario
st.write("Películas filtradas por género:")
for movie in filtered_movies.itertuples():
    st.write(f"{movie.Title} ({movie.Genre})")
    # Verificar si la ruta de la imagen es válida antes de mostrarla
    if 'Poster' in features_with_genre.columns:
        poster_path = movie.Poster  # Ajusta esta columna si es necesario
        if is_valid_image_path(poster_path):
            st.image(poster_path, width=300, use_column_width=False)  # Mostrar la imagen del póster con tamaño ajustado
        else:
            st.write(f"No se pudo cargar la imagen para la película {movie.Title}")

# Filtro por año
years = sorted(features_with_genre['year'].unique())  # Cambiar 'year' por la columna correcta si es necesario
selected_year = st.selectbox('Selecciona año', years)

# Filtrar películas por año
filtered_movies_year = features_with_genre[features_with_genre['year'] == selected_year]
st.write("Películas filtradas por año:")
for movie in filtered_movies_year.itertuples():
    st.write(f"{movie.Title} ({movie.year})")
    # Verificar si la ruta de la imagen es válida antes de mostrarla
    if 'Poster' in features_with_genre.columns:
        poster_path = movie.Poster  # Ajusta esta columna si es necesario
        if is_valid_image_path(poster_path):
            st.image(poster_path, width=300, use_column_width=False)  # Mostrar la imagen del póster con tamaño ajustado
        else:
            st.write(f"No se pudo cargar la imagen para la película {movie.Title}")
