import pytest

from app.forms import EXAMPLES, FORM_FIELDS, FormError, parse_form
from app.ml import FEATURES
from app.risk import describe_risk
from app.schemas import PatientInput


def test_risk_bands():
    assert describe_risk(0.329).level == "niskie"
    assert describe_risk(0.33).level == "umiarkowane"
    assert describe_risk(0.659).level == "umiarkowane"
    assert describe_risk(0.66).level == "wysokie"


def test_form_fields_match_model():
    schema = PatientInput.model_json_schema()["properties"]
    assert set(schema) == {field["name"] for field in FORM_FIELDS}
    for field in FORM_FIELDS:
        prop = schema[field["name"]]
        assert prop["minimum"] == field["min"]
        assert prop["maximum"] == field["max"]


def test_feature_names_match_dataset_order():
    patient = PatientInput(**EXAMPLES["niskie"])
    PatientInput(**EXAMPLES["wysokie"])
    assert list(patient.to_features()) == FEATURES


def _sample_form(**overrides: str) -> dict[str, str]:
    values = {
        "pregnancies": "1",
        "glucose": "89",
        "blood_pressure": "66",
        "skin_thickness": "23",
        "insulin": "94",
        "bmi": "28.1",
        "diabetes_pedigree": "1",
        "age": "21",
    }
    values.update(overrides)
    return values


def test_pedigree_level_becomes_genetic_weight():
    patient = parse_form(_sample_form(diabetes_pedigree="3"))
    assert patient.diabetes_pedigree == 0.5

    with pytest.raises(FormError):
        parse_form(_sample_form(diabetes_pedigree="4"))


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Glukoza" in response.text
    assert "Oblicz ryzyko" in response.text
    assert "Wagi genetyczne" in response.text
    assert "Kuzyni pierwszego stopnia" in response.text


def test_form_requires_fields(client):
    response = client.post("/", data={"pregnancies": ""})
    assert response.status_code == 400
    assert "Uzupełnij pole" in response.text


def test_comma_decimal_and_history(client):
    response = client.post(
        "/",
        data={
            "pregnancies": "2",
            "glucose": "120",
            "blood_pressure": "70",
            "skin_thickness": "20",
            "insulin": "80",
            "bmi": "28,5",
            "diabetes_pedigree": "2",
            "age": "33",
        },
    )
    assert response.status_code == 200
    assert "Wynik predykcji" in response.text

    history = client.get("/historia")
    assert history.status_code == 200
    assert "28.5" in history.text
    assert "0.25" in history.text


def test_api_examples_are_ordered(client):
    low = client.post("/api/predict", json=EXAMPLES["niskie"])
    high = client.post("/api/predict", json=EXAMPLES["wysokie"])
    assert low.status_code == 200
    assert high.status_code == 200
    assert 0 <= low.json()["probability"] <= 1
    assert low.json()["probability"] < high.json()["probability"]

    history = client.get("/api/predictions")
    assert history.status_code == 200
    assert len(history.json()) >= 2


def test_api_rejects_out_of_range(client):
    payload = dict(EXAMPLES["niskie"])
    payload["age"] = 0
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
