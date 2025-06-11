import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# Carpeta donde están tus imágenes
POSTER_FOLDER = posters
OUTPUT_CSV = poster_features.csv

# Función para extraer histograma normalizado
def extract_image_features(path)
    try
        image = Image.open(path).convert(RGB).resize((224, 224))
        image = np.array(image)
        r = np.histogram(image[, , 0], bins=64, range=(0, 255), density=True)[0]
        g = np.histogram(image[, , 1], bins=64, range=(0, 255), density=True)[0]
        b = np.histogram(image[, , 2], bins=64, range=(0, 255), density=True)[0]
        return np.concatenate([r, g, b]).astype(np.float32)
    except
        return None

# Cargar imágenes y generar características
data = []
for fname in tqdm(sorted(os.listdir(POSTER_FOLDER)))
    if fname.endswith(.jpg) or fname.endswith(.png) or fname.endswith(.jpeg)
        movie_id = fname.split(.)[0]
        path = os.path.join(POSTER_FOLDER, fname)
        features = extract_image_features(path)
        if features is not None
            row = {tmdbId int(movie_id)}
            for i, val in enumerate(features)
                row[ffeat_{i}] = val
            data.append(row)

# Guardar a CSV
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)
print(f✅ Archivo generado {OUTPUT_CSV} con {df.shape[0]} pósters y {df.shape[1] - 1} características)
