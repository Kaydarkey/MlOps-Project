import boto3
import os

S3_BUCKET = 'eplprediction-mlops'
FILES = [
    ('data/enhanced_data.csv', 'data/enhanced_data.csv'),
    ('data/premier_league_2024_2025_table.csv', 'data/premier_league_2024_2025_table.csv'),
]

def download_from_s3():
    s3 = boto3.client('s3')
    os.makedirs('data', exist_ok=True)
    for s3_key, local_path in FILES:
        print(f"Downloading {s3_key} from S3 to {local_path}...")
        s3.download_file(S3_BUCKET, s3_key, local_path)
        print(f"Downloaded {local_path}")

if __name__ == '__main__':
    download_from_s3() 