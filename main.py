# main.py

from fastapi import FastAPI, Query
from pydantic import BaseModel
import joblib
import re
import pandas as pd
import numpy as np
import torch
import pickle
from sentence_transformers import util

# -------- Load Models & Files --------

# Load eco-rating model and vectorizer
eco_model = joblib.load("eco_model.pkl")
eco_vectorizer = joblib.load("eco_vectorizer.pkl")

# Load sentence transformer model
with open("sentence_transformer_model.pkl", "rb") as f:
    rec_model = pickle.load(f)
rec_model = rec_model.to("cuda" if torch.cuda.is_available() else "cpu")

# Load product embeddings
with open("product_embeddings.pkl", "rb") as f:
    product_embeddings = pickle.load(f)

# Load product dataset
df = pd.read_csv("webdata.csv")
text_columns = ["title", "name", "category", "material", "description"]
df[text_columns] = df[text_columns].fillna('')
df["combined_text"] = df[text_columns].agg(" ".join, axis=1)

# -------- Init FastAPI --------
app = FastAPI()

# -------- Shared Cleaner --------
def clean_text(text):
    text = str(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\w\s&'%\-]", "", text)
    text = re.sub(r"([!?.])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

# -------- Eco Rating Endpoint --------

class ProductData(BaseModel):
    title: str
    description: str
    material: str = ""
    category: str = ""

def prepare_input(data: ProductData):
    parts = [data.title, data.description, data.material, data.category]
    combined = " | ".join([p for p in parts if p.strip()])
    cleaned = clean_text(combined)
    return cleaned

@app.post("/predict")
def predict_eco_score(product: ProductData):
    cleaned_text = prepare_input(product)
    vector = eco_vectorizer.transform([cleaned_text])
    prediction = eco_model.predict(vector)[0]

    labels = ["biodegradable", "recyclable", "low_waste", "eco_packaging", "low_carbon", "renewable"]
    result = dict(zip(labels, prediction.tolist()))
    eco_score = int(sum(prediction))

    return {
        "eco_parameters": result,
        "eco_score": eco_score
    }

@app.get("/related")
def recommend_products(query: str = Query(...), top_k: int = 5):
    try:
        if len(df) == 0 or len(product_embeddings) == 0:
            return {"error": "No data available for recommendations."}

        query_embedding = rec_model.encode([clean_text(query)], convert_to_tensor=True)[0].cpu().numpy()

        if isinstance(product_embeddings, torch.Tensor):
            product_embeddings_np = product_embeddings.cpu().numpy()
        else:
            product_embeddings_np = product_embeddings

        similarities = util.cos_sim(query_embedding, product_embeddings_np).flatten()

        if np.any(np.isnan(similarities)):
            return {"error": "Invalid similarity scores found."}

        top_k_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_k_indices:
            results.append({
                "title": df.iloc[idx]["title"],
                "brand": df.iloc[idx].get("brand", ""),
                "description": df.iloc[idx]["description"],
                "score": float(similarities[idx])
            })

        return {
            "query": query,
            "top_k": top_k,
            "results": results
        }

    except Exception as e:
        return {"error": str(e)}

# -------- Root --------
@app.get("/")
def root():
    return {"message": "Eco + Recommendation API is live!"}
