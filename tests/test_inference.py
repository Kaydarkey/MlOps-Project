from fastapi.testclient import TestClient
from inference.inference_api import app
import pytest

# This test depends on the training fixture to ensure model artifacts exist
from tests.test_train import run_training

client = TestClient(app)

# Example valid payload for testing
# In a real scenario, this might come from a fixture or a shared utility
VALID_PAYLOAD = {
    "avg_GoalsScored_home": 1.5,
    "avg_GoalsConceded_home": 1.0,
    "avg_Shots_home": 12.5,
    "avg_ShotsOnTarget_home": 4.5,
    "avg_GoalsScored_away": 1.2,
    "avg_GoalsConceded_away": 1.3,
    "avg_Shots_away": 10.0,
    "avg_ShotsOnTarget_away": 3.8,
}

def test_health_check():
    """Tests the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}

def test_single_prediction():
    """Tests the /predict endpoint with a valid payload."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    json_response = response.json()
    assert "predicted_outcome" in json_response
    assert "probabilities" in json_response
    assert json_response["predicted_outcome"] in ["H", "D", "A"]

def test_batch_prediction():
    """Tests the /batch_predict endpoint with a valid payload."""
    batch_payload = {"matches": [VALID_PAYLOAD, VALID_PAYLOAD]}
    response = client.post("/batch_predict", json=batch_payload)
    assert response.status_code == 200
    json_response = response.json()
    assert "predictions" in json_response
    assert isinstance(json_response["predictions"], list)
    assert len(json_response["predictions"]) == 2
    for prediction in json_response["predictions"]:
        assert "predicted_outcome" in prediction
        assert "probabilities" in prediction

def test_prediction_with_invalid_payload():
    """Tests the API's response to a payload with missing fields."""
    invalid_payload = VALID_PAYLOAD.copy()
    del invalid_payload["avg_GoalsScored_home"]
    response = client.post("/predict", json=invalid_payload)
    # FastAPI should return a 422 Unprocessable Entity for Pydantic validation errors
    assert response.status_code == 422 