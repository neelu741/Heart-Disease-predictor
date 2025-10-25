from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load trained model and scaler
with open("knn_heart.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_heart.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI()

# Input data model
class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int

@app.post("/predict")
def predict(data: PatientData):
    input_data = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope
    ]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return {"prediction": int(prediction[0])}
