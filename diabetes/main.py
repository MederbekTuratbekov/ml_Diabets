import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path



BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / 'model_Diabets.pkl')
scaler = joblib.load(BASE_DIR / 'scaler_Diabets.pkl')

app = FastAPI()


class ModelSchema(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post('/predict/')
async def predict(diagnose: ModelSchema):
    diagnose_dict = dict(diagnose)
    features = list(diagnose_dict.values())
    scaled = scaler.transform([features])
    predict = model.predict(scaled)[0]
    return {'approved': bool(predict)}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
