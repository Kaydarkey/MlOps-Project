import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import yaml
import boto3
from io import BytesIO
import mlflow
import mlflow.sklearn

def train_model(s3_processed_path: str):
    """
    Loads processed data from a given S3 path, trains a model, 
    and logs the experiment to MLflow, saving artifacts to S3 via MLflow.
    """
    # Load config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    s3_config = config["s3"]
    mlflow_config = config["mlflow"]

    # Set up MLflow
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])
    
    # The artifact location will be s3://<bucket>/<experiment_id>/<run_id>/artifacts
    # It's good practice to set this at the experiment level
    try:
        experiment = mlflow.get_experiment_by_name(mlflow_config["experiment_name"])
        if experiment is None:
            mlflow.create_experiment(
                mlflow_config["experiment_name"],
                artifact_location=f"s3://{s3_config['bucket']}/mlflow_artifacts"
            )
    except Exception as e:
        print(f"Could not configure MLflow experiment: {e}")


    # Start an MLflow run
    with mlflow.start_run() as run:
        print(f"Starting MLflow run: {run.info.run_name}")
        mlflow.log_param("model_version", config["model"]["version"])

        # Load data from S3
        try:
            print(f"Reading processed data from {s3_processed_path}...")
            df = pd.read_csv(s3_processed_path)
        except Exception as e:
            print(f"Failed to read from S3: {e}")
            raise

        if df.empty:
            print("ERROR: The processed dataframe is empty. No data to train on.")
            print("This can happen if preprocessing removes all rows, e.g., due to insufficient data for rolling averages.")
            # Exit gracefully without raising an exception to not fail the whole flow if desired
            return

        # Define features and target
        features = [col for col in df.columns if 'avg_' in col]
        x = df[features]
        y = df['FTR']

        # Encode target variable
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Split data
        x_train, x_test, y_train, y_test = train_test_split(
            x, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Train a RandomForestClassifier
        print("Training RandomForestClassifier...")
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            min_samples_leaf=1,
            max_features='sqrt'
        )
        model.fit(x_train, y_train)

        # Log parameters
        mlflow.log_params(model.get_params())

        # Evaluate model
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy on test set: {accuracy:.3f}")
        mlflow.log_metric("accuracy", accuracy)

        # Log the model to MLflow (which saves it to S3)
        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="epl-prediction-model"
        )
        
        # Save and log the label encoder as an artifact
        encoder_path = "/tmp/label_encoder.pkl"
        joblib.dump(le, encoder_path)
        mlflow.log_artifact(encoder_path, artifact_path="encoder")

        print("MLflow run completed successfully.")


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    s3_path = f"s3://{config['s3']['bucket']}/{config['s3']['processed_data_key']}"
    train_model(s3_path) 