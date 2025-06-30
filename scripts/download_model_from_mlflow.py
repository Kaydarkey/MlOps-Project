import mlflow
import joblib
import os

# Set MLflow tracking URI to the running server
mlflow.set_tracking_uri("http://localhost:5001")

model_name = "epl-prediction-model"
model_version = "8"  # Change if you want a different version

# Download the model as a sklearn object
model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")

# Save it as a pickle file for your Flask app
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/epl_model.pkl")
print("Model downloaded and saved as models/epl_model.pkl") 