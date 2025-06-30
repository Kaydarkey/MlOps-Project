import boto3
import yaml
from pathlib import Path
from combine_local_data import combine_local_data

def upload_raw_data():
    """
    Combines local data files and uploads the result to the S3 bucket.
    Returns the S3 path of the uploaded file.
    """
    # First, combine the local data files
    print("--- Combining local data files ---")
    combine_local_data()
    print("--- Data combination complete ---")
    
    # Load config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    bucket_name = config["s3"]["bucket"]
    raw_data_key = config["s3"]["raw_data_key"]
    local_raw_path = Path("data/raw_epl_data.csv")

    if not local_raw_path.exists():
        print(f"ERROR: Local data file not found at {local_raw_path}")
        print("Please ensure you have a local copy of the raw data.")
        exit(1)

    try:
        print(f"Uploading {local_raw_path} to s3://{bucket_name}/{raw_data_key}...")
        s3_client = boto3.client("s3")
        s3_client.upload_file(str(local_raw_path), bucket_name, raw_data_key)
        print("Upload complete.")
        s3_raw_path = f"s3://{bucket_name}/{raw_data_key}"
        return s3_raw_path
    except Exception as e:
        print(f"Failed to upload to S3: {e}")
        raise


if __name__ == "__main__":
    upload_raw_data() 