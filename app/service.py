import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db_models import Prediction
from app.ml import predict_patient
from app.risk import classification_label, describe_risk, probability_percent
from app.schemas import PatientInput, PredictionRead

logger = logging.getLogger(__name__)


def to_read(row: Prediction) -> PredictionRead:
    risk = describe_risk(row.probability)
    return PredictionRead(
        id=row.id,
        pregnancies=row.pregnancies,
        glucose=row.glucose,
        blood_pressure=row.blood_pressure,
        skin_thickness=row.skin_thickness,
        insulin=row.insulin,
        bmi=row.bmi,
        diabetes_pedigree=row.diabetes_pedigree,
        age=row.age,
        probability=row.probability,
        probability_percent=probability_percent(row.probability),
        prediction=row.prediction,
        risk_level=risk.level,
        risk_label=risk.label,
        risk_color=risk.color,
        risk_badge=risk.badge,
        risk_bar=risk.bar,
        classification_label=classification_label(row.prediction),
        created_at=row.created_at,
    )


def create_prediction(db: Session, patient: PatientInput) -> PredictionRead:
    label, probability = predict_patient(patient)
    record = Prediction(
        pregnancies=patient.pregnancies,
        glucose=patient.glucose,
        blood_pressure=patient.blood_pressure,
        skin_thickness=patient.skin_thickness,
        insulin=patient.insulin,
        bmi=patient.bmi,
        diabetes_pedigree=patient.diabetes_pedigree,
        age=patient.age,
        probability=probability,
        prediction=label,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.info("Zapisano predykcję %s, prawdopodobieństwo %.3f.", record.id, probability)
    return to_read(record)


def list_predictions(db: Session, limit: int = 200) -> list[PredictionRead]:
    statement = (
        select(Prediction)
        .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .limit(limit)
    )
    rows = db.scalars(statement).all()
    return [to_read(row) for row in rows]
