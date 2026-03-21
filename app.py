import streamlit as st
import numpy as np
import joblib

# ------------------ Load Files ------------------
try:
    model = joblib.load('KNN_heart.pkl')
    scaler = joblib.load('scaler_heart.pkl')
    columns = joblib.load('columns_heart.pkl')
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Heart Disease Prediction",
                   layout="centered")
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

# ------------------ Predict Button ------------------
if st.button("🔍 Predict"):

    # Input validation
    if resting_bp == 0 or cholesterol == 0 or max_hr == 0:
        st.warning("⚠️ Please enter valid non-zero values "
                   "for BP, Cholesterol and Max HR.")
    else:
        # ---------- One-Hot Encode to Match Trained Features ----------
        input_df = np.zeros((1, len(columns)))

        for i, col in enumerate(columns):
            if col == 'Age':
                input_df[0, i] = age
            elif col == 'RestingBP':
                input_df[0, i] = resting_bp
            elif col == 'Cholesterol':
                input_df[0, i] = cholesterol
            elif col == 'FastingBS':
                input_df[0, i] = 1 if fasting_bs == "Yes" else 0
            elif col == 'MaxHR':
                input_df[0, i] = max_hr
            elif col == 'Oldpeak':
                input_df[0, i] = oldpeak
            elif col == 'Sex_M':
                input_df[0, i] = 1 if sex == "Male" else 0
            elif col == 'ChestPainType_ATA':
                input_df[0, i] = 1 if chest_pain == "ATA" else 0
            elif col == 'ChestPainType_NAP':
                input_df[0, i] = 1 if chest_pain == "NAP" else 0
            elif col == 'ChestPainType_TA':
                input_df[0, i] = 1 if chest_pain == "TA" else 0
            elif col == 'RestingECG_Normal':
                input_df[0, i] = 1 if resting_ecg == "Normal" else 0
            elif col == 'RestingECG_ST':
                input_df[0, i] = 1 if resting_ecg == "ST" else 0
            elif col == 'ExerciseAngina_Y':
                input_df[0, i] = 1 if exercise_angina == "Yes" else 0
            elif col == 'ST_Slope_Flat':
                input_df[0, i] = 1 if st_slope == "Flat" else 0
            elif col == 'ST_Slope_Up':
                input_df[0, i] = 1 if st_slope == "Up" else 0

        # ---------- Scale & Predict ----------
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0]
        confidence = round(float(max(probability)) * 100, 1)

        # ---------- Display Result ----------
        if prediction == 1:
            st.error("🚨 High Risk of Heart Disease Detected!")
        else:
            st.success("✅ No Heart Disease Detected! Stay Healthy 💪")

        st.metric("Confidence", f"{confidence}%")
