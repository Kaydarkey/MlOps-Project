import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest

# Define paths to the scripts
DATA_COLLECTION_SCRIPT = "scripts/data_collection.py"
PREPROCESS_SCRIPT = "scripts/preprocess.py"
RAW_DATA_PATH = Path("data/raw_epl_data.csv")
PROCESSED_DATA_PATH = Path("data/processed_epl_data.csv")

@pytest.fixture(scope="module", autouse=True)
def run_data_pipeline():
    """Fixture to run the data pipeline once before tests in this module."""
    # Clean up old files before running
    if RAW_DATA_PATH.exists():
        RAW_DATA_PATH.unlink()
    if PROCESSED_DATA_PATH.exists():
        PROCESSED_DATA_PATH.unlink()
        
    print("\nRunning data collection script...")
    subprocess.run([sys.executable, DATA_COLLECTION_SCRIPT], check=True)
    
    print("Running preprocessing script...")
    subprocess.run([sys.executable, PREPROCESS_SCRIPT], check=True)
    
    yield # This is where the tests will run
    
    # Teardown: clean up created files
    print("\nCleaning up data files...")
    if RAW_DATA_PATH.exists():
        RAW_DATA_PATH.unlink()
    if PROCESSED_DATA_PATH.exists():
        PROCESSED_DATA_PATH.unlink()

def test_data_collection_creates_raw_file():
    """Tests if the data collection script creates the raw CSV file."""
    assert RAW_DATA_PATH.exists(), f"{RAW_DATA_PATH} was not created."
    df = pd.read_csv(RAW_DATA_PATH)
    assert not df.empty, "Raw data file is empty."

def test_preprocess_creates_processed_file():
    """Tests if the preprocessing script creates the processed CSV file."""
    assert PROCESSED_DATA_PATH.exists(), f"{PROCESSED_DATA_PATH} was not created."
    df = pd.read_csv(PROCESSED_DATA_PATH)
    assert not df.empty, "Processed data file is empty."

def test_processed_data_has_engineered_features():
    """Tests if the processed data contains the expected engineered feature columns."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    expected_feature_cols = [
        'avg_GoalsScored_home', 'avg_GoalsConceded_home', 'avg_Shots_home', 'avg_ShotsOnTarget_home',
        'avg_GoalsScored_away', 'avg_GoalsConceded_away', 'avg_Shots_away', 'avg_ShotsOnTarget_away'
    ]
    for col in expected_feature_cols:
        assert col in df.columns, f"Expected feature column '{col}' not in processed data." 