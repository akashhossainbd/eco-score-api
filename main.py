from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

# Load your saved model and vectorizer
model = joblib.load("eco_model.pkl")
vectorizer = joblib.load("eco_vectorizer.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define the input format expected by the API
class ProductData(BaseModel):
    title: str
    description: str
    material: str = ""
    category: str = ""

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Combine and clean all product fields into a single text string
def prepare_input(data: ProductData):
    parts = [data.title, data.description, data.material, data.category]
    combined = " | ".join([p for p in parts if p.strip()])
    cleaned = clean_text(combined)
    return cleaned

# Define the prediction endpoint
@app.post("/predict")
def predict_eco_score(product: ProductData):
    cleaned_text = prepare_input(product)
    vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(vector)[0]

    # Map prediction to labels
    labels = ["biodegradable", "recyclable", "low_waste", "eco_packaging", "low_carbon", "renewable"]
    result = dict(zip(labels, prediction.tolist()))
    eco_score = sum(prediction)

    return {
        "eco_parameters": result,
        "eco_score": int(eco_score)
    }
@app.get("/")
def read_root():
    return {"message": "Eco Rating API is live!"}