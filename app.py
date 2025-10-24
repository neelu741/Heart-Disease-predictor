import streamlit as st
import numpy as np
import joblib

# ------------------ Load Files ------------------
model = joblib.load(open('KNN_heart.pkl', 'rb'))
scaler = joblib.load(open('scaler_heart.pkl', 'rb'))
columns = joblib.load(open('columns_heart.pkl', 'rb'))

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("Fill the following details to check heart disease risk:")

# ------------------ Input Fields ------------------
age = st.number_input("Age", min_value=1, max_value=120, step=1)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0)
cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Max Heart Rate", min_value=0)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("Oldpeak (ST Depression)", format="%.1f")
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ------------------ Encoding ------------------
sex = 1 if sex == "Male" else 0
fasting_bs = 1 if fasting_bs == "Yes" else 0
exercise_angina = 1 if exercise_angina == "Yes" else 0
st_slope = {"Up": 2, "Flat": 1, "Down": 0}[st_slope]
chest_pain = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}[chest_pain]
resting_ecg = {"Normal": 0, "ST": 1, "LVH": 2}[resting_ecg]

# ------------------ Prepare Data ------------------
input_data = np.array([[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                        resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])

# Ensure same feature order
input_df = np.zeros((1, len(columns)))
for i, col in enumerate(columns):
    if col == 'Age': input_df[0, i] = age
    elif col == 'Sex': input_df[0, i] = sex
    elif col == 'ChestPainType': input_df[0, i] = chest_pain
    elif col == 'RestingBP': input_df[0, i] = resting_bp
    elif col == 'Cholesterol': input_df[0, i] = cholesterol
    elif col == 'FastingBS': input_df[0, i] = fasting_bs
    elif col == 'RestingECG': input_df[0, i] = resting_ecg
    elif col == 'MaxHR': input_df[0, i] = max_hr
    elif col == 'ExerciseAngina': input_df[0, i] = exercise_angina
    elif col == 'Oldpeak': input_df[0, i] = oldpeak
    elif col == 'ST_Slope': input_df[0, i] = st_slope

# Scale input
scaled_input = scaler.transform(input_df)

# ------------------ Prediction ------------------
if st.button("🔍 Predict"):
    prediction = model.predict(scaled_input)[0]
    if prediction == 1:
        st.error("🚨 High Risk of Heart Disease Detected!")
    else:
        st.success("✅ No Heart Disease Detected! Stay Healthy 💪")
