import re
import pandas as pd
import boto3

RAW_FILE = 'league_table.txt'
CSV_FILE = 'premier_league_2024_2025_table.csv'
S3_BUCKET = 'eplprediction-mlops'
S3_KEY = 'data/premier_league_2024_2025_table.csv'

# Example line: 1   LIV  38  84  W W L D L
row_pattern = re.compile(r'^(\d+)\s+(\w+)\s+(\d+)\s+(\d+)\s+([WLDR ]+)$')

def parse_table():
    with open(RAW_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('Pos.'):
            continue
        m = row_pattern.match(line)
        if m:
            pos, team, played, pts, form = m.groups()
            form_list = form.strip().split()
            data.append({
                'Position': int(pos),
                'Team': team,
                'Played': int(played),
                'Points': int(pts),
                'Form': ' '.join(form_list),
                'Form_Wins': form_list.count('W'),
                'Form_Draws': form_list.count('D'),
                'Form_Losses': form_list.count('L')
            })
    df = pd.DataFrame(data)
    df.to_csv(CSV_FILE, index=False)
    print(f"Saved league table to {CSV_FILE}")
    return df

def upload_to_s3():
    s3 = boto3.client('s3')
    s3.upload_file(CSV_FILE, S3_BUCKET, S3_KEY)
    print(f"Uploaded league table to s3://{S3_BUCKET}/{S3_KEY}")

if __name__ == '__main__':
    df = parse_table()
    upload_to_s3() 