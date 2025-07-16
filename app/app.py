import os
import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load regression models for score prediction
import joblib
import boto3
import yaml
from io import BytesIO
home_goals_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'home_goals_model.pkl')
away_goals_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'away_goals_model.pkl')

def load_model_from_s3(model_key):
    with open(os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    bucket_name = config['s3']['bucket']
    s3_client = boto3.client('s3')
    try:
        obj = s3_client.get_object(Bucket=bucket_name, Key=model_key)
        model_bytes = obj['Body'].read()
        return joblib.load(BytesIO(model_bytes))
    except Exception as e:
        print(f"[WARNING] Could not load {model_key} from S3: {e}")
        return None

home_goals_model = load_model_from_s3('models/home_goals_model.pkl')
if home_goals_model is None:
    home_goals_model = joblib.load(home_goals_model_path) if os.path.exists(home_goals_model_path) else None

away_goals_model = load_model_from_s3('models/away_goals_model.pkl')
if away_goals_model is None:
    away_goals_model = joblib.load(away_goals_model_path) if os.path.exists(away_goals_model_path) else None

# Extract unique teams from data
csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_epl_data.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    home_teams = {t for t in df['HomeTeam'].unique() if isinstance(t, str)}
    away_teams = {t for t in df['AwayTeam'].unique() if isinstance(t, str)}
    teams = sorted(home_teams | away_teams)
else:
    teams = []

def get_team_rows(df, team_a, team_b):
    home_row = df[df['HomeTeam'] == team_a].sort_values('Date', ascending=False).head(1)
    away_row = df[df['AwayTeam'] == team_b].sort_values('Date', ascending=False).head(1)
    return home_row, away_row

def prepare_input_data(home_row, away_row):
    return pd.DataFrame({
        'avg_GoalsScored_home': home_row['avg_GoalsScored_home'].values,
        'avg_GoalsConceded_home': home_row['avg_GoalsConceded_home'].values,
        'avg_Shots_home': home_row['avg_Shots_home'].values,
        'avg_ShotsOnTarget_home': home_row['avg_ShotsOnTarget_home'].values,
        'avg_GoalsScored_away': away_row['avg_GoalsScored_away'].values,
        'avg_GoalsConceded_away': away_row['avg_GoalsConceded_away'].values,
        'avg_Shots_away': away_row['avg_Shots_away'].values,
        'avg_ShotsOnTarget_away': away_row['avg_ShotsOnTarget_away'].values,
    })

def predict_score(home_goals_model, away_goals_model, input_data):
    pred_home_goals = home_goals_model.predict(input_data)[0]
    pred_away_goals = away_goals_model.predict(input_data)[0]
    return int(round(pred_home_goals)), int(round(pred_away_goals))

def determine_outcome(team_a, team_b, pred_home_goals_rounded, pred_away_goals_rounded):
    if pred_home_goals_rounded > pred_away_goals_rounded:
        return f"Winner: {team_a}"
    elif pred_home_goals_rounded < pred_away_goals_rounded:
        return f"Winner: {team_b}"
    else:
        return "Draw"

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = None
    team_a = None
    team_b = None
    scoreline = None
    outcome = None
    if request.method == 'POST':
        team_a = request.form.get('team_a')
        team_b = request.form.get('team_b')
        if home_goals_model is not None and away_goals_model is not None:
            try:
                print(f"[DEBUG] Selected teams: {team_a} (home), {team_b} (away)")
                df = pd.read_csv(csv_path)
                home_row, away_row = get_team_rows(df, team_a, team_b)
                print(f"[DEBUG] home_row empty: {home_row.empty}, away_row empty: {away_row.empty}")
                if home_row.empty or away_row.empty:
                    error = "Stats not found for one or both teams in the data."
                else:
                    input_data = prepare_input_data(home_row, away_row)
                    print(f"[DEBUG] Input data for prediction:\n{input_data}")
                    pred_home_goals_rounded, pred_away_goals_rounded = predict_score(home_goals_model, away_goals_model, input_data)
                    scoreline = f"{pred_home_goals_rounded} - {pred_away_goals_rounded}"
                    outcome = determine_outcome(team_a, team_b, pred_home_goals_rounded, pred_away_goals_rounded)
                    prediction = f"{team_a} vs {team_b}: {scoreline} ({outcome})"
            except Exception as e:
                error = f"Error: {e}"
        else:
            error = "Score prediction models not loaded."
        return render_template('index.html', team_a=team_a, team_b=team_b, prediction=prediction, error=error, teams=teams)
    return render_template('index.html', prediction=None, error=None, teams=teams)

if __name__ == '__main__':
    app.run(debug=True) 