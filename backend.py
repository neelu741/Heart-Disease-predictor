from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import sys

# Load trained model, scaler, and columns
try:
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler_heart.pkl")
    columns = joblib.load("columns_heart.pkl")
except FileNotFoundError as e:
    print(f"Model file missing: {e}")
    sys.exit(1)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model matching your trained features
class PatientData(BaseModel):
    Age: float
    RestingBP: float
    Cholesterol: float
    FastingBS: int
    MaxHR: float
    Oldpeak: float
    Sex_M: int
    ChestPainType_ATA: int
    ChestPainType_NAP: int
    ChestPainType_TA: int
    RestingECG_Normal: int
    RestingECG_ST: int
    ExerciseAngina_Y: int
    ST_Slope_Flat: int
    ST_Slope_Up: int

@app.get("/health")
def health():
    return {"status": "ok", "model": "KNN Heart Disease Predictor"}

@app.post("/predict")
def predict(data: PatientData):
    input_data = np.array([[
        data.Age, data.RestingBP, data.Cholesterol,
        data.FastingBS, data.MaxHR, data.Oldpeak,
        data.Sex_M, data.ChestPainType_ATA,
        data.ChestPainType_NAP, data.ChestPainType_TA,
        data.RestingECG_Normal, data.RestingECG_ST,
        data.ExerciseAngina_Y, data.ST_Slope_Flat,
        data.ST_Slope_Up
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    return {
        "prediction": int(prediction),
        "confidence": round(float(max(probability)) * 100, 1),
        "label": "High Risk" if prediction == 1 else "Low Risk"
    }
