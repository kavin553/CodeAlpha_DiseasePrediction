import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Disease Prediction System", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
body {background: linear-gradient(180deg, #f0f6ff 0%, #ffffff 100%);}
.card {background: linear-gradient(90deg,#ffffff,#f7fbff); padding:16px; border-radius:12px}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("models/best_model.pkl")
PREP_PATH = Path("models/preprocessor.pkl")
LE_PATH = Path("models/label_encoder.pkl")

def load_artifacts():
    if not MODEL_PATH.exists() or not PREP_PATH.exists() or not LE_PATH.exists():
        st.warning("Models not found. Run `python train_model.py` to train and save models.")
        return None, None, None
    model = joblib.load(MODEL_PATH)
    pre = joblib.load(PREP_PATH)
    le = joblib.load(LE_PATH)
    return model, pre, le

model, preprocessor, label_encoder = load_artifacts()

st.title("🩺 Disease Prediction System")
st.write("Predict likelihood of common diseases using patient medical data.")

with st.sidebar:
    st.header("Patient Input")
    age = st.slider("Age", 18, 100, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bp = st.number_input("Blood Pressure", 60, 220, 120)
    hr = st.number_input("Heart Rate", 40, 180, 75)
    cholesterol = st.number_input("Cholesterol Level", 100, 400, 200)
    blood_sugar = st.selectbox("High Blood Sugar (diagnosed)", [0, 1])
    bmi = st.slider("BMI", 10.0, 50.0, 26.0)
    smoking = st.selectbox("Smoking Habit", ["Never", "Former", "Current"])
    alcohol = st.selectbox("Alcohol Consumption", ["None", "Moderate", "High"])
    activity = st.selectbox("Physical Activity", ["Low", "Medium", "High"])
    family = st.selectbox("Family History (1=yes)", [0, 1])
    glucose = st.number_input("Glucose Level", 50.0, 400.0, 110.0)
    insulin = st.number_input("Insulin Level", 0.0, 500.0, 15.0)
    severity = st.slider("Symptom Severity", 0, 10, 2)

    submit = st.button("Predict 🚀")

input_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Blood Pressure": bp,
    "Heart Rate": hr,
    "Cholesterol Level": cholesterol,
    "Blood Sugar": blood_sugar,
    "BMI": bmi,
    "Smoking Habit": smoking,
    "Alcohol Consumption": alcohol,
    "Physical Activity": activity,
    "Family History": family,
    "Chest Pain": 0,
    "Fatigue": 0,
    "Fever": 0,
    "Cough": 0,
    "Breathing Difficulty": 0,
    "Headache": 0,
    "Nausea": 0,
    "Glucose Level": glucose,
    "Insulin Level": insulin,
    "Symptom Severity": severity,
}])

if submit:
    if model is None:
        st.error("Model artifacts missing. Please train the model first.")
    else:
        with st.spinner("Predicting..."):
            Xp = preprocessor.transform(input_df)
            probs = model.predict_proba(Xp)[0]
            idx = np.argmax(probs)
            label = label_encoder.inverse_transform([idx])[0]
            confidence = probs[idx]

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"Prediction: {label}")
            st.metric("Confidence", f"{confidence*100:.1f}%")
            if confidence > 0.7:
                st.success("High confidence — please consult a physician.")
            elif confidence > 0.4:
                st.warning("Moderate confidence — consider further tests.")
            else:
                st.info("Low confidence — likely healthy.")

        with col2:
            fig = px.pie(values=probs, names=label_encoder.inverse_transform(range(len(probs))),
                         title="Prediction Probabilities")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Recommendations")
        if label == "Healthy":
            st.write("Keep a healthy lifestyle: balanced diet, regular exercise, routine check-ups.")
        elif label == "Diabetes":
            st.write("Recommend glucose monitoring, dietary changes, and consulting an endocrinologist.")
        elif label == "Heart Disease":
            st.write("Recommend cardiac evaluation, control blood pressure and cholesterol, and lifestyle changes.")
        elif label == "Breast Cancer":
            st.write("Recommend immediate specialist referral and imaging as appropriate.")

st.markdown("---")
st.write("Built for demo: quick, actionable predictions. Not a substitute for professional medical diagnosis.")
