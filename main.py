from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

# -------- Load Models --------
eco_model = joblib.load("eco_model.pkl")
eco_vectorizer = joblib.load("eco_vectorizer.pkl")

# -------- Init FastAPI --------
app = FastAPI()

# -------- Text Cleaner --------
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

# -------- Input Model --------
class ProductData(BaseModel):
    title: str
    description: str
    material: str = ""
    category: str = ""

def prepare_input(data: ProductData):
    parts = [data.title, data.description, data.material, data.category]
    combined = " | ".join([p for p in parts if p.strip()])
    return clean_text(combined)

# -------- Eco Rating Endpoint --------
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

# -------- Health Check --------
@app.get("/")
def root():
    return {"message": "Eco Rating API is live!"}
