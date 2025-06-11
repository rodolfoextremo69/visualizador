import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
import clip
import os
import pickle

# Cargar modelo CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Cargar metadata
df = pd.read_csv("clip_metadata.csv")

embeddings = []
titles = []
paths = []

# Procesar imágenes una por una
for idx, row in tqdm(df.iterrows(), total=len(df)):
    path = row["image_path"]
    title = row["title"]
    if not os.path.exists(path):
        continue
    try:
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image).squeeze(0).cpu().numpy()
        embeddings.append(embedding)
        titles.append(title)
        paths.append(path)
    except Exception as e:
        print(f"Error con {path}: {e}")

# Guardar vectores
with open("clip_posters.pkl", "wb") as f:
    pickle.dump({"titles": titles, "paths": paths, "embeddings": embeddings}, f)

print("✅ Embeddings guardados en clip_posters.pkl")