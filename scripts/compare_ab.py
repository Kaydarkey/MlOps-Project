import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

# Use the modern NumPy random generator with a seed for reproducibility
rng = np.random.default_rng(42)

# Placeholder: load dummy validation data
data = pd.DataFrame({
    'home_team_stat': rng.random(30),
    'away_team_stat': rng.random(30),
    'home_goals': rng.integers(0, 5, size=30),
    'away_goals': rng.integers(0, 5, size=30)
})

X_val = data[['home_team_stat', 'away_team_stat']]
y_true = (data['home_goals'] > data['away_goals']).astype(int)

model_a = joblib.load('models/epl_model_A.pkl')
model_b = joblib.load('models/epl_model_B.pkl')

y_pred_a = model_a.predict(X_val)
y_pred_b = model_b.predict(X_val)

acc_a = accuracy_score(y_true, y_pred_a)
acc_b = accuracy_score(y_true, y_pred_b)

print(f'Model A Accuracy: {acc_a:.3f}')
print(f'Model B Accuracy: {acc_b:.3f}')

# Log results (placeholder: print, but could log to S3, Prometheus, etc.) 