# ❤️ Heart Disease Predictor

A machine learning web application that predicts the risk of heart 
disease based on clinical parameters. Built with Python, Scikit-learn, 
and Streamlit.

---

## 🔗 Live Demo
👉 [Click here to try the app](https://your-app-link.streamlit.app)

---

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. 
This project uses a K-Nearest Neighbors (KNN) classifier trained 
on clinical data to predict whether a patient is at high or low 
risk of heart disease based on 11 input features.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Scikit-learn | Machine learning |
| Matplotlib / Seaborn | Data visualization |
| Streamlit | Web app frontend |
| FastAPI | REST API backend |
| Joblib | Model serialization |

---

## 📂 Project Structure
```
heart-disease-predictor/
│
├── app.py                  # Streamlit frontend
├── backend.py              # FastAPI backend
├── database.ipynb          # EDA + model training notebook
│
├── KNN_heart.pkl           # Trained KNN model
├── scaler_heart.pkl        # Fitted StandardScaler
├── columns_heart.pkl       # Feature column names
│
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## 📊 Dataset

- **Source:** [Heart Failure Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Size:** 918 rows × 12 columns
- **Target:** `HeartDisease` (0 = No Disease, 1 = Disease)

### Features Used

| Feature | Description |
|---|---|
| Age | Age of the patient |
| Sex | Gender (M/F) |
| ChestPainType | ATA / NAP / ASY / TA |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| FastingBS | Fasting blood sugar > 120 mg/dl |
| RestingECG | Resting ECG results |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise induced angina (Y/N) |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Slope of peak exercise ST segment |

---

## 🧹 Data Preprocessing

- Replaced zero values in `Cholesterol` and `RestingBP` with 
  column mean
- Applied `pd.get_dummies()` for one-hot encoding of 
  categorical features
- Scaled numerical features using `StandardScaler`

---

## 🤖 Model Training

Five models were trained and compared:

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 87.0% | 0.885 |
| **K-Nearest Neighbors** | **86.4%** | **0.882** |
| Naive Bayes | 84.8% | 0.861 |
| Support Vector Machine | 84.8% | 0.867 |
| Decision Tree | 78.3% | 0.802 |

✅ **KNN was selected** as the final model based on its 
balance of accuracy and F1 score.

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/neelu741/heart-disease-predictor.git
cd heart-disease-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

### 4. Run the FastAPI backend (optional)
```bash
uvicorn backend:app --reload
```
Then open `http://127.0.0.1:8000/docs` to test the API.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check if server is running |
| POST | `/predict` | Get heart disease prediction |

### Example Request
```json
POST /predict
{
  "Age": 55,
  "RestingBP": 140,
  "Cholesterol": 250,
  "FastingBS": 0,
  "MaxHR": 130,
  "Oldpeak": 1.5,
  "Sex_M": 1,
  "ChestPainType_ATA": 0,
  "ChestPainType_NAP": 0,
  "ChestPainType_TA": 0,
  "RestingECG_Normal": 1,
  "RestingECG_ST": 0,
  "ExerciseAngina_Y": 1,
  "ST_Slope_Flat": 1,
  "ST_Slope_Up": 0
}
```

### Example Response
```json
{
  "prediction": 1,
  "confidence": 87.5,
  "label": "High Risk"
}
```

---

## 📸 Screenshots

> Add screenshots of your Streamlit app here

---

## ⚠️ Disclaimer

This application is for **educational purposes only**. It is not 
a substitute for professional medical advice, diagnosis, or 
treatment. Always consult a qualified cardiologist for medical 
decisions.

---

## 👩‍💻 Author

**Neelu Kushwaha**
- 🌐 Portfolio: [neelukushwaha.netlify.app](https://neelukushwaha.netlify.app)
- 💼 LinkedIn: [www.linkedin.com/in/neelu-kushwaha-9745b2301](https://www.linkedin.com/in/neelu-kushwaha-9745b2301/)
- 🐙 GitHub: [github.com/neelu741](https://github.com/neelu741)

---

## 📄 License

This project is open source and available under the 
[MIT License](LICENSE).
