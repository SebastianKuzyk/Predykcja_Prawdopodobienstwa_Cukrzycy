"""Trening i predykcja ryzyka cukrzycy.

Potok:
1. W kolumnach, w których 0 oznacza brak pomiaru, zera zamieniane są na braki.
2. Braki uzupełnia mediana policzona na danych treningowych.
3. Cechy są standaryzowane.
4. Regresja logistyczna zwraca prawdopodobieństwo klasy „cukrzyca”.

Liczba ciąż równa 0 zostaje zerem. Metryki w metrics.json pochodzą z podziału
80/20. Plik joblib zawiera model nauczony ponownie na całym zbiorze.
"""

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import CSV_PATH, METRICS_PATH, MODEL_DIR, MODEL_PATH
from app.schemas import PatientInput

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

_model: Pipeline | None = None


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def _mark_missing_zeros(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in ZERO_AS_MISSING:
        prepared[column] = prepared[column].mask(prepared[column] == 0)
    return prepared


def _load_training_frame() -> tuple[pd.DataFrame, pd.Series]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Brak pliku z danymi: {CSV_PATH}")

    dataframe = pd.read_csv(CSV_PATH)
    required = FEATURES + ["Outcome"]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Brak kolumn w pliku CSV: {', '.join(missing)}")

    dataframe = dataframe.dropna(subset=["Outcome"])
    features = _mark_missing_zeros(dataframe.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce"))
    target = dataframe["Outcome"].astype(int)
    if target.nunique() < 2:
        raise ValueError("Zbiór treningowy musi zawierać obie klasy wyniku.")
    return features, target


def train_and_save() -> dict:
    global _model

    features, target = _load_training_frame()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    holdout = build_pipeline()
    holdout.fit(x_train, y_train)
    probabilities = holdout.predict_proba(x_test)[:, 1]
    predicted = holdout.predict(x_test)

    metrics = {
        "algorithm": "LogisticRegression",
        "accuracy": accuracy_score(y_test, predicted),
        "precision": precision_score(y_test, predicted, zero_division=0),
        "recall": recall_score(y_test, predicted, zero_division=0),
        "f1": f1_score(y_test, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "rows": int(len(features)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "random_state": 42,
        "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }

    deployed = build_pipeline()
    deployed.fit(features, target)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(deployed, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _model = deployed
    return metrics


def ensure_model() -> Pipeline:
    global _model
    if _model is None:
        if MODEL_PATH.exists():
            _model = joblib.load(MODEL_PATH)
        else:
            print("Brak zapisanego modelu — rozpoczynam trening.")
            train_and_save()
    return _model


def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def features_frame(patient: PatientInput) -> pd.DataFrame:
    frame = pd.DataFrame([patient.to_features()], columns=FEATURES)
    return _mark_missing_zeros(frame)


def predict_patient(patient: PatientInput) -> tuple[int, float]:
    model = ensure_model()
    frame = features_frame(patient)
    classes = list(model.named_steps["model"].classes_)
    positive_index = classes.index(1)
    probability = float(model.predict_proba(frame)[0, positive_index])
    label = int(model.predict(frame)[0])
    return label, probability


def main() -> None:
    metrics = train_and_save()
    print(f"Zapisano model: {MODEL_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
