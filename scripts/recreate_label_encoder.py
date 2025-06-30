import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
csv_path = 'data/enhanced_data.csv'
df = pd.read_csv(csv_path)

# Get unique team names
teams = {t for t in df['HomeTeam'].unique() if isinstance(t, str)} | {t for t in df['AwayTeam'].unique() if isinstance(t, str)}
teams = sorted(teams)

# Fit label encoder
encoder = LabelEncoder()
encoder.fit(teams)

# Save encoder
joblib.dump(encoder, 'epl_label_encoder.pkl')
print('Label encoder saved as epl_label_encoder.pkl') 