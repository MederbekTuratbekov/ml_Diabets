import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / 'model_Diabetes.pkl')
scaler = joblib.load(BASE_DIR / 'scaler_Diabetes.pkl')

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

    prediction = int(model.predict(scaled)[0])

    if prediction == 1:
        message = "Ответ модели 1 означает, что у вас высокая вероятность наличия сахарного диабета."
    else:
        message = "Ответ модели 0 означает, что вероятность наличия сахарного диабета низкая."

    return {
        'approved': bool(prediction),
        'message': message
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)