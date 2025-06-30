import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load processed data
csv_path = 'data/processed_epl_data.csv'
df = pd.read_csv(csv_path)

# Features and targets
features = [
    'avg_GoalsScored_home', 'avg_GoalsConceded_home', 'avg_Shots_home', 'avg_ShotsOnTarget_home',
    'avg_GoalsScored_away', 'avg_GoalsConceded_away', 'avg_Shots_away', 'avg_ShotsOnTarget_away'
]
X = df[features]
y_home = df['FTHG']  # Full Time Home Goals
y_away = df['FTAG']  # Full Time Away Goals

# Split data
X_train, X_test, y_home_train, y_home_test = train_test_split(X, y_home, test_size=0.2, random_state=42)
_, _, y_away_train, y_away_test = train_test_split(X, y_away, test_size=0.2, random_state=42)

# Train regressors
home_model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=1, max_features='auto')
home_model.fit(X_train, y_home_train)
away_model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=1, max_features='auto')
away_model.fit(X_train, y_away_train)

# Evaluate
home_pred = home_model.predict(X_test)
away_pred = away_model.predict(X_test)
print(f"Home Goals MAE: {mean_absolute_error(y_home_test, home_pred):.2f}")
print(f"Away Goals MAE: {mean_absolute_error(y_away_test, away_pred):.2f}")

# Save models
os.makedirs('models', exist_ok=True)
joblib.dump(home_model, 'models/home_goals_model.pkl')
joblib.dump(away_model, 'models/away_goals_model.pkl')
print('Models saved as models/home_goals_model.pkl and models/away_goals_model.pkl') 