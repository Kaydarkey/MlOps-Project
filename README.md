# EPL Score Prediction MLOps Pipeline

This project predicts English Premier League (EPL) match outcomes using a robust, production-grade MLOps pipeline. It automates everything from data collection to model deployment, with real-time monitoring and CI/CD, following best practices for reliability, reproducibility, and scalability.

## Key Features
- **End-to-End Automation:** Prefect orchestrates data collection, preprocessing, model training, evaluation, and deployment.
- **Experiment Tracking:** MLflow logs every model run, parameter, and metric, with artifacts stored in AWS S3.
- **Centralized Storage:** All datasets, models, and experiment logs are managed in S3 for easy access and reproducibility.
- **Model Serving:** Predictions are available via both a user-friendly web UI (Flask) and a programmatic FastAPI endpoint.
- **Live Monitoring:** Model performance metrics are exposed via Prometheus and visualized in Grafana dashboards.
- **Containerized Deployment:** Docker Compose manages the inference API, Prometheus, and Grafana for consistent, portable deployment.
- **CI/CD Pipeline:** Jenkins automates testing, building, and deployment, ensuring code quality and reliability.

## Tech Stack
- **Orchestration:** Prefect
- **Experiment Tracking:** MLflow
- **Cloud Storage:** AWS S3
- **ML & Data:** Pandas, Scikit-learn
- **Web UI:** Flask (app/app.py)
- **API & Inference:** FastAPI (inference/inference_api.py)
- **Monitoring:** Prometheus, Grafana
- **Containerization:** Docker, Docker Compose
- **CI/CD:** Jenkins

## Project Structure
```
MlOps-Project/
├── app/
│   ├── app.py                # Flask web UI for predictions
│   ├── static/
│   └── templates/
├── configs/
│   └── config.yaml           # Central config (S3, MLflow, etc.)
├── data/                     # Local data files (raw, processed, etc.)
├── inference/
│   ├── inference_api.py      # FastAPI for programmatic predictions & metrics
│   └── Dockerfile            # Inference service container
├── pipelines/
│   ├── train_model_dag.py    # Prefect pipeline (end-to-end automation)
│   └── rollback.py           # (Planned) Model rollback logic
├── scripts/
│   ├── combine_local_data.py # Merges EPL & Championship CSVs
│   ├── preprocess.py         # Cleans and engineers features
│   ├── train.py              # Trains classifier, logs to MLflow
│   ├── train_regression.py   # Trains regression models for score prediction
│   ├── evaluate.py           # Evaluates model, exposes Prometheus metrics
│   ├── download_latest_data.py # Downloads latest data from S3
│   ├── parse_results_to_csv.py # Parses match results to CSV
│   ├── parse_league_table_to_csv.py # Parses league table to CSV
│   └── ...                   # Other utility scripts
├── tests/                    # Unit and integration tests
├── Jenkinsfile               # CI/CD pipeline (Docker Compose)
├── docker-compose.yml        # Orchestrates inference, Prometheus, Grafana
├── grafana/                  # Grafana provisioning/configs
├── mlruns/                   # Local MLflow tracking data
├── requirements.txt          # Python dependencies
└── README.md
```

## How the Pipeline Works

1. **Data Collection & Preparation**
   - EPL and Championship CSVs are merged (`combine_local_data.py`).
   - Combined data is uploaded to S3 (`data_collection.py`).
   - Data is cleaned and features (rolling averages, etc.) are engineered (`preprocess.py`).

2. **Model Training & Tracking**
   - A RandomForestClassifier predicts match outcomes (`train.py`).
   - Separate regressors predict exact home/away goals (`train_regression.py`).
   - All runs, parameters, and metrics are logged to MLflow, with artifacts in S3.

3. **Model Evaluation & Monitoring**
   - The latest model is evaluated, and metrics (accuracy, F1, etc.) are exposed via a Prometheus endpoint (`evaluate.py`).
   - Prometheus scrapes these metrics; Grafana visualizes them in real time.

4. **Model Serving**
   - **Web UI:** Flask app (`app/app.py`) lets users select teams and get score predictions.
   - **API:** FastAPI service (`inference/inference_api.py`) exposes REST endpoints for predictions and metrics.

5. **Automation & CI/CD**
   - **Pipeline Orchestration:** Prefect automates the full workflow (`pipelines/train_model_dag.py`).
   - **CI/CD:** Jenkinsfile defines steps for testing, building, and deploying with Docker Compose.

## Setup and Installation

### 1. Prerequisites
- Python 3.10+
- Docker & Docker Compose
- AWS account with S3 bucket and credentials (`~/.aws/credentials`)

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
Edit `configs/config.yaml` with your S3, MLflow, and other settings:
```yaml
s3:
  bucket: "your-s3-bucket-name"
  # ... other keys
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "epl_score_prediction"
# ... other configs
```

### 6. Download the Data
Download and place these files in the `data/` directory:
- [EPL Data 2023/24](https://www.football-data.co.uk/mmz4281/2324/E0.csv) → `data/E0.csv`
- [Championship Data 2023/24](https://www.football-data.co.uk/mmz4281/2324/E1.csv) → `data/E1.csv`

## Running the System

### 1. Start MLflow UI
```bash
mlflow ui
```
Access at [http://localhost:5000](http://localhost:5000)

### 2. Start Monitoring & Inference Stack
```bash
docker-compose up --build
```
- Starts inference API (FastAPI), Prometheus, and Grafana.
- Prometheus scrapes metrics from `/metrics` endpoint.
- Grafana dashboards at [http://localhost:3000](http://localhost:3000) (default: admin/admin).

### 3. Run the Training Pipeline
```bash
python pipelines/train_model_dag.py
```
- Downloads latest data from S3, processes, trains, evaluates, and logs everything.

### 4. Use the Web UI
- Go to the Flask app (usually at [http://localhost:5000] or as configured) to select teams and get predictions.

### 5. Use the API
- FastAPI endpoints for predictions and metrics (see `inference/inference_api.py`).

## Pipeline Stages (Detailed)
1. **Collect Data:** Merge and upload raw data to S3.
2. **Preprocess Data:** Clean and engineer features, save processed data to S3.
3. **Train Model:** Train classifier and regressors, log to MLflow.
4. **Evaluate Model:** Expose metrics for monitoring.
5. **Serve Predictions:** Web UI and API for user/programmatic access.
6. **Monitor & Visualize:** Prometheus and Grafana for real-time model health.
7. **CI/CD:** Jenkins automates testing, building, and deployment.

## Monitoring & Visualization
- **Prometheus:** Collects real-time model metrics from the inference API.
- **Grafana:** Visualizes metrics with pre-configured dashboards.

## CI/CD
- **Jenkins:** Runs tests, builds Docker images, starts/stops services, and ensures everything works before deployment.

## Troubleshooting
- **Port Conflicts:** If you see `OSError: [Errno 98] Address already in use`, change the port or stop the process using it (`lsof -i :8000` and `kill <pid>`).
- **Missing /metrics Endpoint:** Ensure `prometheus_fastapi_instrumentator` is installed and integrated in FastAPI.
- **Docker Build Issues:** Make sure all necessary files (e.g., `requirements.txt`, `configs/`) are present.

## Future Work
- Fully implement the `inference_api.py` for production-grade predictions.
- Add A/B testing and automated model rollback.
- Use a more robust Prefect backend (e.g., PostgreSQL) for production.

## Authors
- SKKD

## License
- MIT 