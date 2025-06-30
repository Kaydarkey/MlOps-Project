import pandas as pd
from pathlib import Path
import yaml
import boto3

def get_rolling_averages(group, cols, new_cols):
    """Calculate rolling averages for specified columns."""
    group = group.sort_values("Date")
    rolling_stats = group[cols].rolling(5, closed='left', min_periods=1).mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

def preprocess_data(s3_raw_path: str) -> str:
    """
    Loads raw data from a given S3 path, cleans it, engineers features, 
    and saves the processed data back to S3.
    Returns the S3 path of the processed file.
    """
    # Load config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    bucket_name = config["s3"]["bucket"]
    processed_data_key = config["s3"]["processed_data_key"]
    s3_processed_path = f"s3://{bucket_name}/{processed_data_key}"
    
    try:
        print(f"Reading raw data from {s3_raw_path}...")
        df = pd.read_csv(s3_raw_path, parse_dates=['Date'])
    except Exception as e:
        print(f"Failed to read from S3: {e}")
        raise

    # Basic cleaning
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'])
    
    # Feature Engineering: Rolling Averages
    # We calculate stats for home and away teams separately
    df_home = df[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'HS', 'HST']].rename(
        columns={'HomeTeam': 'Team', 'FTHG': 'GoalsScored', 'FTAG': 'GoalsConceded', 'HS': 'Shots', 'HST': 'ShotsOnTarget'}
    )
    df_away = df[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'AS', 'AST']].rename(
        columns={'AwayTeam': 'Team', 'FTAG': 'GoalsScored', 'FTHG': 'GoalsConceded', 'AS': 'Shots', 'AST': 'ShotsOnTarget'}
    )
    
    df_team_stats = pd.concat([df_home, df_away]).sort_values('Date')

    cols = ['GoalsScored', 'GoalsConceded', 'Shots', 'ShotsOnTarget']
    new_cols = [f"avg_{c}" for c in cols]

    df_rolling = df_team_stats.groupby("Team").apply(lambda x: get_rolling_averages(x, cols, new_cols))
    
    # After groupby, 'Team' is in the index. Let's reset it to be a column.
    df_rolling = df_rolling.droplevel(0).reset_index()

    # Prepare stats for home teams
    home_stats = df_rolling.rename(columns={
        'avg_GoalsScored': 'avg_GoalsScored_home',
        'avg_GoalsConceded': 'avg_GoalsConceded_home',
        'avg_Shots': 'avg_Shots_home',
        'avg_ShotsOnTarget': 'avg_ShotsOnTarget_home',
    })
    home_stats = home_stats[['Date', 'Team', 'avg_GoalsScored_home', 'avg_GoalsConceded_home', 'avg_Shots_home', 'avg_ShotsOnTarget_home']]

    # Prepare stats for away teams
    away_stats = df_rolling.rename(columns={
        'avg_GoalsScored': 'avg_GoalsScored_away',
        'avg_GoalsConceded': 'avg_GoalsConceded_away',
        'avg_Shots': 'avg_Shots_away',
        'avg_ShotsOnTarget': 'avg_ShotsOnTarget_away',
    })
    away_stats = away_stats[['Date', 'Team', 'avg_GoalsScored_away', 'avg_GoalsConceded_away', 'avg_Shots_away', 'avg_ShotsOnTarget_away']]

    # Merge home team stats
    df = df.merge(
        home_stats, 
        left_on=['Date', 'HomeTeam'], 
        right_on=['Date', 'Team'],
        how='left'
    ).drop('Team', axis=1)

    # Merge away team stats
    df = df.merge(
        away_stats, 
        left_on=['Date', 'AwayTeam'], 
        right_on=['Date', 'Team'],
        how='left'
    ).drop('Team', axis=1)

    # Final feature selection and cleaning
    final_cols = [
        'Date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG',
        'avg_GoalsScored_home', 'avg_GoalsConceded_home', 'avg_Shots_home', 'avg_ShotsOnTarget_home',
        'avg_GoalsScored_away', 'avg_GoalsConceded_away', 'avg_Shots_away', 'avg_ShotsOnTarget_away'
    ]
    # Drop rows where stats couldn't be calculated (first few games of a season for a team)
    processed_df = df[final_cols].dropna()

    # Save processed data to S3
    try:
        print(f"Writing processed data to {s3_processed_path}...")
        processed_df.to_csv(s3_processed_path, index=False)
        print("Processed data saved successfully.")
        return s3_processed_path
    except Exception as e:
        print(f"Failed to write to S3: {e}")
        raise

if __name__ == "__main__":
    # This part is for standalone execution and testing
    # In a real run, the s3_raw_path would be passed from the orchestration tool
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    s3_path = f"s3://{config['s3']['bucket']}/{config['s3']['raw_data_key']}"
    preprocess_data(s3_path) 