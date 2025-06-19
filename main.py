from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import pickle
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import re

app = FastAPI()

# Load models
with open("eco_model.pkl", "rb") as f:
    eco_model = pickle.load(f)

with open("eco_vectorizer.pkl", "rb") as f:
    eco_vectorizer = pickle.load(f)

with open("sentence_transformer_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("product_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Load product data
df = pd.read_csv("webdata.csv")

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

# Input schema for /predict
class Product(BaseModel):
    title: str
    description: str
    brand: str
    category: str

# -------------------------
#        /predict
# -------------------------
@app.post("/predict")
async def predict_score(product: Product):
    combined_text = f"{product.title} {product.description} {product.brand} {product.category}"
    cleaned = clean_text(combined_text)
    X = eco_vectorizer.transform([cleaned])
    y_pred = eco_model.predict(X)[0]

    labels = ["Biodegradable", "Recyclable", "Waste", "Packaging", "Carbon", "Renewable"]
    breakdown = dict(zip(labels, y_pred))
    eco_score = int((sum(y_pred) / len(y_pred)) * 100)

    return {
        "eco_score": eco_score,
        "breakdown": breakdown
    }

# -------------------------
#        /related
# -------------------------
@app.get("/related")
def related_products(query: str, top_k: int = 5):
    cleaned = clean_text(query)
    emb = model.encode([cleaned], convert_to_tensor=True)
    scores = cosine_similarity(emb.cpu().numpy(), embeddings[:])[0]
    top_indices = scores.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        item = {
            "title": str(df.loc[idx, "title"]),
            "brand": str(df.loc[idx, "brand"]),
            "description": str(df.loc[idx, "description"]),
            "score": float(scores[idx])
        }
        results.append(item)

    return {
        "query": query,
        "top_k": top_k,
        "results": results
    }
