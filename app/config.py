import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
CSV_PATH = Path(os.getenv("CSV_PATH", BASE_DIR / "diabetes.csv"))
MODEL_PATH = MODEL_DIR / "diabetes_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
HYPERPARAMETERS_PATH = MODEL_DIR / "hyperparameters.json"


def _database_url() -> str:
    configured = os.getenv("DATABASE_URL")
    if configured:
        return configured
    database_file = (DATA_DIR / "predictions.db").resolve()
    return f"sqlite:///{database_file.as_posix()}"


DATABASE_URL = _database_url()
