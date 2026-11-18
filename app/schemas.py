from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PatientInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "pregnancies": 1,
                    "glucose": 89,
                    "blood_pressure": 66,
                    "skin_thickness": 23,
                    "insulin": 94,
                    "bmi": 28.1,
                    "diabetes_pedigree": 0.167,
                    "age": 21,
                }
            ]
        }
    )

    pregnancies: int = Field(ge=0, le=20, description="Liczba ciąż")
    glucose: float = Field(ge=0, le=400, description="Glukoza (mg/dL)")
    blood_pressure: float = Field(ge=0, le=250, description="Ciśnienie rozkurczowe (mm Hg)")
    skin_thickness: float = Field(ge=0, le=100, description="Grubość fałdu skórnego (mm)")
    insulin: float = Field(ge=0, le=1000, description="Insulina (μU/ml)")
    bmi: float = Field(ge=0, le=80, description="BMI (kg/m²)")
    diabetes_pedigree: float = Field(
        ge=0,
        le=3,
        description="Diabetes Pedigree Function",
    )
    age: int = Field(ge=1, le=120, description="Wiek w latach")

    def to_features(self) -> dict[str, float]:
        return {
            "Pregnancies": self.pregnancies,
            "Glucose": self.glucose,
            "BloodPressure": self.blood_pressure,
            "SkinThickness": self.skin_thickness,
            "Insulin": self.insulin,
            "BMI": self.bmi,
            "DiabetesPedigreeFunction": self.diabetes_pedigree,
            "Age": self.age,
        }


class PredictionRead(BaseModel):
    id: int
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: int
    probability: float
    probability_percent: float
    prediction: int
    risk_level: str
    risk_label: str
    risk_color: str
    risk_badge: str
    risk_bar: str
    classification_label: str
    created_at: datetime
