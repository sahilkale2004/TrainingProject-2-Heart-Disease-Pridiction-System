import streamlit as st
import requests
import json

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Prediction System")
st.markdown("Enter the patient details below to predict the likelihood of heart disease.")

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        chest_pain_type = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
        resting_blood_pressure = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
        cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
        fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
    
    with col2:
        resting_ecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
        max_heart_rate = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
        exercise_induced_angina = st.selectbox("Exercise Induced Angina", options=[0, 1])
        st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0)
        st_slope = st.selectbox("ST Slope", options=[0, 1, 2])
        num_major_vessels = st.selectbox("Number of Major Vessels", options=[0, 1, 2, 3, 4])
    
    thalassemia = st.selectbox("Thalassemia", options=[0, 1, 2, 3])
    
    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    # Prepare data for API
    payload = {
        "age": float(age),
        "sex": float(sex),
        "chest_pain_type": float(chest_pain_type),
        "resting_blood_pressure": float(resting_blood_pressure),
        "cholesterol": float(cholesterol),
        "fasting_blood_sugar": float(fasting_blood_sugar),
        "resting_ecg": float(resting_ecg),
        "max_heart_rate": float(max_heart_rate),
        "exercise_induced_angina": float(exercise_induced_angina),
        "st_depression": float(st_depression),
        "st_slope": float(st_slope),
        "num_major_vessels": float(num_major_vessels),
        "thalassemia": float(thalassemia)
    }
    
    try:
        # Backend URL (Update if deployed)
        backend_url = "http://localhost:8000/predict"
        response = requests.post(backend_url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            prob = result["heart_disease_probability"]
            pred = result["prediction"]
            
            st.divider()
            if pred == 1:
                st.error(f"Prediction: **HEART DISEASE DETECTED** (Probability: {prob:.2f})")
            else:
                st.success(f"Prediction: **NO HEART DISEASE** (Probability: {prob:.2f})")
        else:
            st.warning("Could not reach the backend. Is it running?")
            
    except Exception as e:
        st.error(f"Error: {e}")

