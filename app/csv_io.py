"""Wczytywanie nowych przypadków z CSV i zapis wyników do CSV."""

import io

import pandas as pd

from app.forms import PEDIGREE_LEVELS, FormError, parse_form
from app.schemas import PatientInput, PredictionRead

COLUMN_ALIASES = {
    "pregnancies": "pregnancies",
    "liczba_ciaz": "pregnancies",
    "glucose": "glucose",
    "glukoza": "glucose",
    "blood_pressure": "blood_pressure",
    "bloodpressure": "blood_pressure",
    "cisnienie": "blood_pressure",
    "cisnienie_krwi": "blood_pressure",
    "skin_thickness": "skin_thickness",
    "skinthickness": "skin_thickness",
    "grubosc_faldu_skornego": "skin_thickness",
    "insulin": "insulin",
    "insulina": "insulin",
    "bmi": "bmi",
    "diabetes_pedigree": "diabetes_pedigree",
    "diabetespedigreefunction": "diabetes_pedigree",
    "wskaznik_rodowodu_cukrzycy": "diabetes_pedigree",
    "age": "age",
    "wiek": "age",
}

REQUIRED_COLUMNS = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
]


def _normalize_header(name: object) -> str:
    folded = str(name).strip().lower().translate(str.maketrans("ąćęłńóśźż", "acelnoszz"))
    key = folded.replace(" ", "_")
    return COLUMN_ALIASES.get(key, key)


def _pedigree_cell(value: str) -> str:
    number = float(value.strip().replace(",", "."))
    if number.is_integer() and int(number) in {0, 1, 2, 3}:
        weight = next(item["weight"] for item in PEDIGREE_LEVELS if item["level"] == int(number))
        return str(weight)
    return value


def parse_patients_csv(text: str) -> tuple[list[PatientInput], list[str]]:
    try:
        frame = pd.read_csv(io.StringIO(text))
    except pd.errors.ParserError:
        return [], ["Nie udało się odczytać pliku CSV."]
    frame = frame.rename(columns={column: _normalize_header(column) for column in frame.columns})
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        return [], [f"Brak kolumn w pliku CSV: {', '.join(missing)}."]
    if len(frame) > 300:
        return [], ["Plik CSV może mieć najwyżej 300 wierszy."]

    patients: list[PatientInput] = []
    messages: list[str] = []
    for position, record in enumerate(frame.to_dict(orient="records"), start=2):
        raw = {}
        for column in REQUIRED_COLUMNS:
            cell = record.get(column, "")
            raw[column] = "" if pd.isna(cell) else str(cell)
        if raw["diabetes_pedigree"].strip():
            try:
                raw["diabetes_pedigree"] = _pedigree_cell(raw["diabetes_pedigree"])
            except ValueError:
                pass
        try:
            patients.append(parse_form(raw, pedigree_as_level=False))
        except FormError as exc:
            messages.append(f"Wiersz {position}: {exc.message}")
    return patients, messages


def predictions_to_csv(rows: list[PredictionRead], high_risk_only: bool = False) -> str:
    selected = [row for row in rows if row.probability >= 0.5] if high_risk_only else rows
    columns = [
        "data",
        "pregnancies",
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
        "diabetes_pedigree",
        "age",
        "probability",
        "prediction",
        "risk_level",
    ]
    records = [
        {
            "data": row.created_at.strftime("%Y-%m-%d %H:%M"),
            "pregnancies": row.pregnancies,
            "glucose": row.glucose,
            "blood_pressure": row.blood_pressure,
            "skin_thickness": row.skin_thickness,
            "insulin": row.insulin,
            "bmi": row.bmi,
            "diabetes_pedigree": row.diabetes_pedigree,
            "age": row.age,
            "probability": row.probability,
            "prediction": row.prediction,
            "risk_level": row.risk_level,
        }
        for row in selected
    ]
    return pd.DataFrame(records, columns=columns).to_csv(index=False)
