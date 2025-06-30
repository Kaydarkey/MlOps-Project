import re
import pandas as pd
import boto3
import os

RAW_FILE = 'raw_results.txt'
RESULTS_CSV = 'premier_league_2024_2025_results.csv'
S3_BUCKET = 'eplprediction-mlops'
S3_KEY = 'data/enhanced_data.csv'
EXISTING_S3_KEY = 'data/raw_epl_data.csv'

def parse_results():
    with open(RAW_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    date = None
    date_pattern = re.compile(r'^\d{1,2} \w+ 2025$')
    result_pattern = re.compile(
        r'^FT\s+([^\t]+)\t([^\t]+)\t(\d+)\tv\t(\d+)\t([^\t]+)\t([^\t]+)'
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if date_pattern.match(line):
            date = line
            continue
        m = result_pattern.match(line)
        if m:
            home_short, home_full, home_score, away_score, away_full, away_short = m.groups()
            data.append({
                'Date': date,
                'Home Team': home_short,
                'Home Name': home_full,
                'Home Score': int(home_score),
                'Away Score': int(away_score),
                'Away Name': away_full,
                'Away Team': away_short
            })
    df = pd.DataFrame(data)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved parsed results to {RESULTS_CSV}")
    return df

def merge_with_existing(df):
    s3 = boto3.client('s3')
    try:
        s3.download_file(S3_BUCKET, EXISTING_S3_KEY, 'raw_epl_data.csv')
        existing_df = pd.read_csv('raw_epl_data.csv')
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        print(f"Merged with existing data from {EXISTING_S3_KEY}")
    except Exception as e:
        print(f"No existing data found or error: {e}. Using only new data.")
        combined_df = df
    combined_df.to_csv('enhanced_data.csv', index=False)
    s3.upload_file('enhanced_data.csv', S3_BUCKET, S3_KEY)
    print(f"Uploaded merged data to s3://{S3_BUCKET}/{S3_KEY}")

if __name__ == '__main__':
    df = parse_results()
    merge_with_existing(df) 