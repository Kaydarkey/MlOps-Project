# EPL Score Prediction MLOps Pipeline

This project implements a production-grade machine learning pipeline to predict the outcomes of English Premier League (EPL) matches. It is built with a focus on MLOps best practices, including automated pipelines, experiment tracking, model management, and real-time monitoring.

## Core Features
- **Automated Training Pipeline:** Orchestrated with Prefect to automate data collection, preprocessing, model training, and evaluation.
- **Experiment Tracking:** Uses MLflow to log model parameters, metrics, and artifacts for every run.
- **Centralized Artifact Storage:** Leverages AWS S3 for storing datasets, trained models, and MLflow artifacts.
- **Live Model Evaluation:** The final pipeline step serves model performance metrics via a Prometheus endpoint for real-time monitoring.
- **Monitoring & Visualization:** Prometheus scrapes metrics from the evaluation service, and Grafana visualizes them in real time.
- **Containerized Inference API:** FastAPI-based inference service, containerized and managed via Docker Compose, exposes a `/metrics` endpoint for Prometheus.
- **CI/CD Ready:** Includes a `Jenkinsfile` for setting up a continuous integration and deployment pipeline using Docker Compose.

## Tech Stack
- **Orchestration:** Prefect
- **Experiment Tracking:** MLflow
- **Cloud Storage:** AWS S3
- **ML & Data:** Pandas, Scikit-learn
- **API & Inference:** FastAPI (defined in `inference/`)
- **Monitoring:** Prometheus, Grafana
- **Containerization:** Docker, Docker Compose
- **CI/CD:** Jenkins

## Project Structure
```
MlOps-Project/
├── app/
│   ├── app.py                # Web UI (if used)
│   ├── static/
│   └── templates/
├── configs/
│   └── config.yaml           # Paths, S3 bucket, MLflow URI
├── data/                     # Local source data (E0.csv, E1.csv, enhanced_data.csv, etc.)
├── inference/
│   ├── inference_api.py      # FastAPI serving logic
│   └── Dockerfile            # Dockerfile for the inference service
├── pipelines/
│   ├── train_model_dag.py    # Prefect flow for the main pipeline
│   └── rollback.py           # Model rollback logic
├── scripts/
│   ├── combine_local_data.py # Merges local CSVs
│   ├── preprocess.py         # Cleans data and engineers features
│   ├── train.py              # Trains model and logs to MLflow
│   ├── evaluate.py           # Evaluates the latest model from MLflow and exposes Prometheus metrics
│   ├── download_latest_data.py # Downloads latest data from S3
│   ├── parse_results_to_csv.py # Parses raw match results text to CSV
│   ├── parse_league_table_to_csv.py # Parses league table text to CSV
│   └── ...                   # Other utility scripts
├── tests/                    # Unit and integration tests
├── Jenkinsfile               # CI/CD pipeline definition (uses Docker Compose)
├── docker-compose.yml        # Orchestrates inference, Prometheus, and Grafana
├── grafana/                  # Grafana provisioning configs
├── mlruns/                   # Local MLflow tracking data
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup and Installation

### 1. Prerequisites
- Python 3.10+
- Docker & Docker Compose
- An AWS account with an S3 bucket and configured credentials (`~/.aws/credentials`).

### 2. Clone the Repository
```bash
git clone <repository_url>
cd MlOps-Project
```

### 3. Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure the Project
Update `configs/config.yaml` with your specific settings:
```yaml
s3:
  bucket: "your-s3-bucket-name" # e.g., eplprediction-mlops
  # ... other keys
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "epl_score_prediction"
# ... other configs
```

### 6. Download the Data
The pipeline requires two local CSV files. Download them and place them in the `data/` directory:
1.  **Premier League Data (2023/24):** [Download Link](https://www.football-data.co.uk/mmz4281/2324/E0.csv)
    -   Save as `data/E0.csv`.
2.  **Championship Data (2023/24):** [Download Link](https://www.football-data.co.uk/mmz4281/2324/E1.csv)
    -   Save as `data/E1.csv`.

## How to Run the Pipeline

### 1. Start the MLflow UI
In a separate terminal, start the MLflow tracking server. This will also store run data locally in the `mlruns` directory.
```bash
mlflow ui
```
You can access the UI at `http://localhost:5000`.

### 2. Start Monitoring Stack (Prometheus, Grafana, Inference API)
Run all services using Docker Compose:
```bash
docker-compose up --build
```
- This will start the inference API (FastAPI), Prometheus, and Grafana.
- Prometheus scrapes metrics from the inference API's `/metrics` endpoint.
- Grafana is pre-configured to use Prometheus as a data source.
- Access Grafana at [http://localhost:3000](http://localhost:3000) (default login: `admin`/`admin`).
- Visualize model evaluation metrics in Grafana dashboards.

### 3. Execute the Prefect Flow
Run the main training pipeline:
```bash
python pipelines/train_model_dag.py
```
- The pipeline will automatically download the latest enhanced data and league table from S3 before each run.

### Pipeline Stages
1.  **Collect Data:** The `combine_local_data` script merges `E0.csv` and `E1.csv` into `raw_epl_data.csv`, which is then uploaded to S3.
2.  **Preprocess Data:** Raw data is read from S3, cleaned, and features (rolling averages for team performance) are engineered. The processed data is saved back to S3.
3.  **Train Model:** The processed data is used to train a `RandomForestClassifier`. The model, label encoder, parameters, and accuracy metric are logged to MLflow.
4.  **Evaluate Model:** The latest model version is loaded from the MLflow Model Registry. It then runs in a loop, exposing performance metrics (accuracy, F1 score) on a Prometheus endpoint at `http://localhost:8000/metrics`.

### 4. CI/CD Pipeline
- The `Jenkinsfile` is set up to use Docker Compose for building, running, and stopping services.
- Automated tests are run inside the inference API container using `pytest`.
- Integrate with Jenkins for a full CI/CD loop.

## Prometheus & Grafana Integration
- **Prometheus** scrapes metrics from the inference API's `/metrics` endpoint (exposed by `prometheus_fastapi_instrumentator`).
- **Grafana** is pre-provisioned to use Prometheus as a data source and can be used to visualize real-time model evaluation metrics.
- All services are orchestrated via `docker-compose.yml`.

## Automated Data Pipeline
- The pipeline automatically downloads the latest enhanced data and league table from S3 before each run, ensuring the most up-to-date data is used for training and evaluation.
- Scripts for parsing raw match results and league tables to CSV, merging with S3 data, and uploading results are included in the `scripts/` directory.

## Troubleshooting
- **Port Conflicts:** If you encounter `OSError: [Errno 98] Address already in use`, change the port used in the evaluation script or stop the process using the port (e.g., `lsof -i :8000` and `kill <pid>`).
- **Missing /metrics Endpoint:** Ensure `prometheus_fastapi_instrumentator` is installed and properly integrated in the FastAPI app.
- **Docker Build Issues:** Make sure the build context includes all necessary files (e.g., `requirements.txt`, `configs/`).

## Future Work
- Fully implement the `inference_api.py` to serve predictions from the best model.
- Implement A/B testing and automated model rollback logic.
- Switch from Prefect's default SQLite database to a more robust backend like PostgreSQL for production use.

## Authors
- SKKD

## License
- MIT 