"""Trening, porównanie modeli i wyjaśnienie predykcji.

Potok dla obu algorytmów:
1. Zera oznaczające brak pomiaru zamieniane są na braki.
2. Braki uzupełnia mediana.
3. Cechy są standaryzowane.
4. Regresja logistyczna albo las losowy zwraca prawdopodobieństwo klasy „cukrzyca”.

Modele są porównywane na podziale 80/20 i w 5-krotnej walidacji krzyżowej.
Do predykcji służy algorytm z wyższym ROC AUC z walidacji krzyżowej,
nauczony ponownie na całym zbiorze. Wynik pojedynczej predykcji
tłumaczą wartości SHAP.
"""

import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import CSV_PATH, HYPERPARAMETERS_PATH, METRICS_PATH, MODEL_DIR, MODEL_PATH
from app.schemas import PatientInput

logger = logging.getLogger(__name__)

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

FEATURE_LABELS = {
    "Pregnancies": "Liczba ciąż",
    "Glucose": "Glukoza",
    "BloodPressure": "Ciśnienie krwi",
    "SkinThickness": "Grubość fałdu skórnego",
    "Insulin": "Insulina",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Wskaźnik rodowodu cukrzycy",
    "Age": "Wiek",
}

ALGORITHM_LABELS = {
    "LogisticRegression": "Regresja logistyczna",
    "RandomForest": "Las losowy",
}

ALGORITHMS = ("LogisticRegression", "RandomForest")

ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

DEFAULT_HYPERPARAMETERS = {
    "logreg_c": 1.0,
    "forest_n_estimators": 200,
    "forest_max_depth": 6,
}

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

_model: Pipeline | None = None
_background: np.ndarray | None = None


def build_pipeline(algorithm: str, hyperparameters: dict) -> Pipeline:
    if algorithm == "LogisticRegression":
        estimator = LogisticRegression(
            C=float(hyperparameters["logreg_c"]),
            max_iter=1000,
            random_state=42,
        )
    elif algorithm == "RandomForest":
        estimator = RandomForestClassifier(
            n_estimators=int(hyperparameters["forest_n_estimators"]),
            max_depth=int(hyperparameters["forest_max_depth"]),
            random_state=42,
        )
    else:
        raise ValueError(f"Nieznany algorytm: {algorithm}")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def load_hyperparameters() -> dict:
    if not HYPERPARAMETERS_PATH.exists():
        return dict(DEFAULT_HYPERPARAMETERS)
    try:
        stored = json.loads(HYPERPARAMETERS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(DEFAULT_HYPERPARAMETERS)
    values = dict(DEFAULT_HYPERPARAMETERS)
    values.update({key: stored[key] for key in DEFAULT_HYPERPARAMETERS if key in stored})
    return values


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


def _holdout_metrics(pipeline: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict:
    probabilities = pipeline.predict_proba(features)[:, 1]
    predicted = pipeline.predict(features)
    return {
        "accuracy": float(accuracy_score(target, predicted)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
    }


def _cv_metrics(pipeline: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(pipeline, features, target, cv=folds, scoring=SCORING)
    return {name: float(scores[f"test_{name}"].mean()) for name in SCORING}


def train_and_save(hyperparameters: dict | None = None) -> dict:
    global _model, _background

    settings = dict(DEFAULT_HYPERPARAMETERS)
    if hyperparameters:
        settings.update(hyperparameters)
    features, target = _load_training_frame()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    logger.info("Porównuję modele na %s wierszach.", len(features))

    compared = []
    for algorithm in ALGORITHMS:
        pipeline = build_pipeline(algorithm, settings)
        pipeline.fit(x_train, y_train)
        compared.append(
            {
                "algorithm": algorithm,
                "label": ALGORITHM_LABELS[algorithm],
                "holdout": _holdout_metrics(pipeline, x_test, y_test),
                "cv": _cv_metrics(build_pipeline(algorithm, settings), x_train, y_train),
            }
        )

    selected = max(compared, key=lambda item: (item["cv"]["roc_auc"], item["holdout"]["roc_auc"]))
    deployed = build_pipeline(selected["algorithm"], settings)
    deployed.fit(features, target)

    metrics = {
        "selected": selected["algorithm"],
        "selected_label": selected["label"],
        "selection_metric": "roc_auc_cv",
        "rows": int(len(features)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "cv_folds": 5,
        "random_state": 42,
        "hyperparameters": settings,
        "models": compared,
        "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(deployed, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    HYPERPARAMETERS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    _model = deployed
    _background = None
    logger.info("Wybrano model %s.", selected["algorithm"])
    return metrics


def _metrics_are_current(metrics: dict | None) -> bool:
    return bool(metrics and isinstance(metrics.get("models"), list) and len(metrics["models"]) >= 2)


def ensure_model() -> Pipeline:
    global _model
    if _model is not None:
        return _model
    if MODEL_PATH.exists() and _metrics_are_current(load_metrics()):
        _model = joblib.load(MODEL_PATH)
        return _model
    logger.info("Brak aktualnego modelu — rozpoczynam porównanie i trening.")
    train_and_save(load_hyperparameters())
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


def predict_frame(frame: pd.DataFrame) -> tuple[int, float]:
    model = ensure_model()
    classes = list(model.named_steps["model"].classes_)
    positive_index = classes.index(1)
    probability = float(model.predict_proba(frame)[0, positive_index])
    label = int(model.predict(frame)[0])
    return label, probability


def predict_patient(patient: PatientInput) -> tuple[int, float]:
    return predict_frame(features_frame(patient))


def _background_matrix(model: Pipeline) -> np.ndarray:
    global _background
    if _background is None:
        features, _target = _load_training_frame()
        sample = features.sample(n=min(80, len(features)), random_state=42)
        _background = np.asarray(model[:-1].transform(sample))
    return _background


def _positive_shap(raw: object, estimator) -> np.ndarray:
    if hasattr(raw, "values"):
        raw = raw.values
    classes = list(getattr(estimator, "classes_", [0, 1]))
    positive = classes.index(1) if 1 in classes else -1
    if isinstance(raw, list):
        return np.asarray(raw[positive])
    array = np.asarray(raw)
    if array.ndim == 3:
        return array[:, :, positive]
    return array


def explain_frame(frame: pd.DataFrame) -> list[dict]:
    model = ensure_model()
    estimator = model.named_steps["model"]
    transformed = np.asarray(model[:-1].transform(frame))
    background = _background_matrix(model)
    if hasattr(estimator, "coef_"):
        explainer = shap.LinearExplainer(estimator, background)
        raw = explainer.shap_values(transformed)
    else:
        explainer = shap.TreeExplainer(estimator)
        raw = explainer.shap_values(transformed)
    values = _positive_shap(raw, estimator).reshape(-1, len(FEATURES))[0]
    maximum = float(np.max(np.abs(values))) or 1.0
    contributions = [
        {
            "feature": feature,
            "label": FEATURE_LABELS[feature],
            "value": float(value),
            "width": abs(float(value)) / maximum * 100,
            "direction": "up" if value > 0 else "down",
        }
        for feature, value in zip(FEATURES, values, strict=True)
    ]
    contributions.sort(key=lambda item: abs(item["value"]), reverse=True)
    return contributions


def explain_patient(patient: PatientInput) -> list[dict]:
    return explain_frame(features_frame(patient))


def main() -> None:
    metrics = train_and_save()
    print(f"Zapisano model: {MODEL_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
