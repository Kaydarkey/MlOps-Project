import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from prometheus_client import start_http_server, Gauge
import time
import yaml
import boto3
from io import BytesIO
import logging

def get_latest_run_id(experiment_name: str) -> str:
    """Gets the ID of the most recent run from a given MLflow experiment."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if runs.empty:
        raise ValueError(f"No runs found in experiment '{experiment_name}'.")
        
    return runs.iloc[0].run_id

def evaluate_model():
    """
    Continuously evaluates the latest model from MLflow and exposes metrics.
    """
    # Load config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    s3_processed_path = f"s3://{config['s3']['bucket']}/{config['s3']['processed_data_key']}"
    mlflow_experiment_name = config["mlflow"]["experiment_name"]

    # Set up Prometheus metrics
    accuracy_gauge = Gauge('model_accuracy', 'Current model accuracy')
    f1_gauge = Gauge('model_f1_score', 'Current model F1 score (weighted)')
    
    # Start Prometheus server
    start_http_server(8002)
    print("Prometheus server started on port 8002")

    while True:
        try:
            # Get the latest MLflow run
            print(f"Fetching latest run from MLflow experiment: '{mlflow_experiment_name}'")
            run_id = get_latest_run_id(mlflow_experiment_name)
            print(f"Found latest run with ID: {run_id}")

            # Define artifact URIs
            model_uri = f"runs:/{run_id}/model"
            encoder_uri = f"runs:/{run_id}/encoder/label_encoder.pkl"

            # Load artifacts
            try:
                print(f"Loading model from: {model_uri}")
                model = mlflow.sklearn.load_model(model_uri)
                
                print(f"Loading encoder from: {encoder_uri}")
                # For non-model files, download_artifacts is the way to go
                local_encoder_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="encoder/label_encoder.pkl")
                le = joblib.load(local_encoder_path)

            except Exception as e:
                print(f"Failed to load artifacts: {e}")
                time.sleep(60) # Wait before retrying
                continue
            
            # Load the test data
            print(f"Reading processed data from {s3_processed_path}...")
            df = pd.read_csv(s3_processed_path)

            if df.empty:
                print("Processed data is empty, skipping evaluation.")
                time.sleep(60)
                continue
            
            # Re-create the same train/test split to get the test set
            features = [col for col in df.columns if 'avg_' in col]
            x = df[features]
            y = df['FTR']
            
            # Ensure the label encoder is fitted on all possible labels from the dataset
            le.fit(y)
            y_encoded = le.transform(y)

            _, x_test, _, y_test = train_test_split(
                x, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            # Make predictions
            y_pred = model.predict(x_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print("--- Evaluation Metrics ---")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"F1 Score (Weighted): {f1:.3f}")
            print(classification_report(y_test, y_pred, target_names=le.classes_))
            
            # Update Prometheus gauges
            accuracy_gauge.set(accuracy)
            f1_gauge.set(f1)

        except Exception as e:
            print(f"An error occurred during evaluation: {e}")
        
        # Wait for a defined interval before re-evaluating
        print("\nSleeping for 60 seconds before next evaluation...")
        time.sleep(60)

if __name__ == '__main__':
    evaluate_model() 