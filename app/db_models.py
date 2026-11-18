from datetime import datetime

from sqlalchemy import DateTime, Float, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pregnancies: Mapped[int] = mapped_column(Integer)
    glucose: Mapped[float] = mapped_column(Float)
    blood_pressure: Mapped[float] = mapped_column(Float)
    skin_thickness: Mapped[float] = mapped_column(Float)
    insulin: Mapped[float] = mapped_column(Float)
    bmi: Mapped[float] = mapped_column(Float)
    diabetes_pedigree: Mapped[float] = mapped_column(Float)
    age: Mapped[int] = mapped_column(Integer)
    probability: Mapped[float] = mapped_column(Float)
    prediction: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
