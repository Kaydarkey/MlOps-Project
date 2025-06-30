import subprocess
import sys
from pathlib import Path
import joblib
import pytest

# This test depends on the data pipeline fixture in test_data_pipeline
from tests.test_data_pipeline import run_data_pipeline

MODEL_PATH = Path("models/epl_model.pkl")
ENCODER_PATH = Path("models/epl_label_encoder.pkl")
TRAIN_SCRIPT = "scripts/train.py"

@pytest.fixture(scope="module", autouse=True)
def run_training():
    """Fixture to run the training script once before tests in this module."""
    # This fixture relies on the run_data_pipeline fixture having already run
    print("\nRunning training script...")
    subprocess.run([sys.executable, TRAIN_SCRIPT], check=True)
    yield
    print("\nCleaning up model files...")
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    if ENCODER_PATH.exists():
        ENCODER_PATH.unlink()

def test_training_creates_model_artifacts():
    """Tests if training creates both model and encoder files."""
    assert MODEL_PATH.exists(), "Model file was not created."
    assert ENCODER_PATH.exists(), "Label encoder file was not created."

def test_model_artifacts_are_loadable():
    """Tests if the created model and encoder can be loaded."""
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        assert model is not None, "Loaded model is None."
        assert encoder is not None, "Loaded encoder is None."
    except Exception as e:
        pytest.fail(f"Failed to load model artifacts: {e}") 