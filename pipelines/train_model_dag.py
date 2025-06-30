from prefect import flow, task, get_run_logger
import sys
import subprocess

# Add scripts directory to path to allow direct imports
sys.path.append('scripts')

from data_collection import upload_raw_data
from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model

# Download latest data from S3 before anything else
subprocess.run(['python', 'scripts/download_latest_data.py'], check=True)

@task
def collect_data_task():
    logger = get_run_logger()
    logger.info("--- Running Data Collection Task ---")
    try:
        s3_path = upload_raw_data()
        logger.info(f"Raw data uploaded to {s3_path}")
        return s3_path
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise

@task
def preprocess_data_task(raw_data_path: str):
    logger = get_run_logger()
    logger.info("--- Running Preprocessing Task ---")
    try:
        processed_path = preprocess_data(raw_data_path)
        logger.info(f"Processed data saved to {processed_path}")
        return processed_path
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

@task
def train_model_task(processed_data_path: str):
    logger = get_run_logger()
    logger.info("--- Running Training Task ---")
    try:
        train_model(processed_data_path)
        logger.info("Model training and logging completed.")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

@task
def evaluate_model_task():
    logger = get_run_logger()
    logger.info("--- Running Evaluation Task ---")
    try:
        # This task runs indefinitely, so we don't expect it to "finish" in the same way
        evaluate_model()
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise

@flow(name="EPL Model Training Pipeline", log_prints=True)
def train_model_flow():
    """
    Orchestrates the full EPL model training pipeline with data dependencies.
    """
    print("Starting EPL Model Training Pipeline...")
    
    raw_path = collect_data_task()
    processed_path = preprocess_data_task(raw_path)
    train_model_task(processed_path)
    evaluate_model_task()
    
    print("Pipeline execution finished.")

if __name__ == "__main__":
    train_model_flow() 