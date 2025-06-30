from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
from pathlib import Path
import boto3
import yaml
from io import BytesIO
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import start_http_server

# --- Constants ---
MODEL_NOT_LOADED_DETAIL = "Model not loaded. Please ensure the training pipeline has run successfully."

# --- Global variables for model and encoder ---
model = None
label_encoder = None

def load_model_from_s3():
    """Loads the model and encoder from S3."""
    global model, label_encoder
    
    print("Loading model from S3...")
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    bucket_name = config["s3"]["bucket"]
    model_key = config["s3"]["model_key"]
    encoder_key = config["s3"]["encoder_key"]

    try:
        s3_client = boto3.client("s3")
        
        # Load model
        model_obj = s3_client.get_object(Bucket=bucket_name, Key=model_key)
        model_bytes = model_obj['Body'].read()
        model = joblib.load(BytesIO(model_bytes))
        
        # Load encoder
        encoder_obj = s3_client.get_object(Bucket=bucket_name, Key=encoder_key)
        encoder_bytes = encoder_obj['Body'].read()
        label_encoder = joblib.load(BytesIO(encoder_bytes))
        
        print("Model and label encoder loaded successfully from S3.")
    except Exception as e:
        print(f"ERROR: Failed to load model from S3: {e}")
        # Keep them as None if loading fails
        model = None
        label_encoder = None

app = FastAPI(
    title="EPL Score Prediction API",
    description="API to predict English Premier League match outcomes.",
    version="1.0.0"
)

# --- Pydantic Models for Input Validation ---
class MatchFeatures(BaseModel):
    avg_GoalsScored_home: float
    avg_GoalsConceded_home: float
    avg_Shots_home: float
    avg_ShotsOnTarget_home: float
    avg_GoalsScored_away: float
    avg_GoalsConceded_away: float
    avg_Shots_away: float
    avg_ShotsOnTarget_away: float

class BatchRequest(BaseModel):
    matches: List[MatchFeatures]

# --- API Endpoints ---
@app.on_event("startup")
def startup_event():
    """Load the model during API startup."""
    load_model_from_s3()

@app.get("/health", summary="Check API Health")
def health():
    """Check if the API is running and the model is loaded."""
    if model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail=MODEL_NOT_LOADED_DETAIL)
    return {"status": "ok", "model_loaded": True}

@app.post("/predict", summary="Predict a single match outcome")
def predict(features: MatchFeatures):
    """
    Predicts the outcome of a single EPL match.
    - **Input**: Rolling average stats for home and away teams.
    - **Output**: Predicted outcome ('H' for Home Win, 'D' for Draw, 'A' for Away Win).
    """
    if model is None:
        raise HTTPException(status_code=503, detail=MODEL_NOT_LOADED_DETAIL)

    # Convert Pydantic model to DataFrame
    input_df = pd.DataFrame([features.dict()])
    
    # Predict
    prediction_encoded = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    # Decode prediction and format probabilities
    prediction_decoded = label_encoder.inverse_transform([prediction_encoded])[0]
    probabilities = dict(zip(label_encoder.classes_, prediction_proba))

    return {
        "predicted_outcome": prediction_decoded,
        "probabilities": probabilities
    }

@app.post("/batch_predict", summary="Predict multiple match outcomes")
def batch_predict(batch: BatchRequest):
    """
    Predicts outcomes for a batch of EPL matches.
    - **Input**: A list of match features.
    - **Output**: A list of predicted outcomes and their probabilities.
    """
    if model is None:
        raise HTTPException(status_code=503, detail=MODEL_NOT_LOADED_DETAIL)
        
    # Convert list of Pydantic models to DataFrame
    input_df = pd.DataFrame([m.dict() for m in batch.matches])
    
    # Predict
    predictions_encoded = model.predict(input_df)
    predictions_proba = model.predict_proba(input_df)

    # Decode predictions
    predictions_decoded = label_encoder.inverse_transform(predictions_encoded)

    results = []
    for i, outcome in enumerate(predictions_decoded):
        probabilities = dict(zip(label_encoder.classes_, predictions_proba[i]))
        results.append({
            "predicted_outcome": outcome,
            "probabilities": probabilities
        })

    return {"predictions": results}

Instrumentator().instrument(app).expose(app)

start_http_server(8002) 