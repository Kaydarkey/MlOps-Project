import pandas as pd
from pathlib import Path

def combine_local_data():
    """
    Combines local EPL and Championship data for the 2023-2024 season
    and saves the result to the raw_epl_data.csv file.
    """
    # Paths to the local data files
    epl_path = Path("data/E0.csv")
    champ_path = Path("data/E1.csv")
    output_path = Path("data/raw_epl_data.csv")

    # Check if the source files exist
    if not epl_path.exists() or not champ_path.exists():
        print("="*80)
        print("ERROR: Local data files not found.")
        print(f"Please download the following files and place them in the '{epl_path.parent}' directory:")
        print(f"1. Premier League Data: https://www.football-data.co.uk/mmz4281/2324/E0.csv  (save as E0.csv)")
        print(f"2. Championship Data:  https://www.football-data.co.uk/mmz4281/2324/E1.csv  (save as E1.csv)")
        print("="*80)
        raise FileNotFoundError("Required data files are missing. Please download them.")

    print("Reading local Premier League data...")
    try:
        epl_data = pd.read_csv(epl_path)
    except Exception as e:
        print(f"Error reading {epl_path}: {e}")
        return

    print("Reading local Championship data...")
    try:
        champ_data = pd.read_csv(champ_path)
    except Exception as e:
        print(f"Error reading {champ_path}: {e}")
        return

    # Combine the two dataframes
    combined_data = pd.concat([epl_data, champ_data], ignore_index=True)

    # Save the combined data to the output file
    print(f"Saving combined data to {output_path}...")
    try:
        combined_data.to_csv(output_path, index=False)
        print("Data combined and saved successfully.")
    except Exception as e:
        print(f"Failed to save data to {output_path}: {e}")

if __name__ == "__main__":
    combine_local_data() 