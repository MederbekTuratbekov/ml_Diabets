# diabetes/main.py

import uvicorn
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
import joblib

BASE_DIR = Path(__file__).parent

FIELDS_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model  = joblib.load(BASE_DIR / "model_log_Diabetes.pkl")
    app.state.scaler = joblib.load(BASE_DIR / "scaler_Diabetes.pkl")
    yield


app = FastAPI(title="Diabetes Classifier", lifespan=lifespan)


# ── Schema ─────────────────────────────────────────────────────────────────────
class DiabetesSchema(BaseModel):
    Pregnancies:              float
    Glucose:                  float
    BloodPressure:            float
    SkinThickness:            float
    Insulin:                  float
    BMI:                      float
    DiabetesPedigreeFunction: float
    Age:                      float

    @field_validator("Glucose", "BloodPressure", "BMI")
    @classmethod
    def must_be_positive(cls, v: float, info) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} должен быть больше нуля")
        return v


# ── Utils ──────────────────────────────────────────────────────────────────────
def build_features(data: DiabetesSchema) -> np.ndarray:
    return np.array([[getattr(data, f) for f in FIELDS_ORDER]], dtype=float)


# ── Endpoint ───────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(diagnose: DiabetesSchema):
    scaled     = app.state.scaler.transform(build_features(diagnose))
    prediction = int(app.state.model.predict(scaled)[0])
    proba      = app.state.model.predict_proba(scaled)[0].tolist()

    return {
        "prediction":          prediction,
        "diabetes_detected":   bool(prediction),
        "message":             "Высокая вероятность наличия сахарного диабета" if prediction == 1
                               else "Вероятность наличия сахарного диабета низкая",
        "probability_positive": round(proba[1] * 100, 2),
        "probability_negative": round(proba[0] * 100, 2),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)