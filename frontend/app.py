import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import os

# Model definition (must match training)
class HeartNet(nn.Module):
    def __init__(self, input_size):
        super(HeartNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

@st.cache_resource
def load_model_and_scaler():
    model_path = "model/heart_model.pth"
    scaler_path = "model/scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        input_size = scaler.n_features_in_
        model = HeartNet(input_size)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model, scaler
    else:
        return None, None

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Prediction System")
st.markdown("Enter the patient details below to predict the likelihood of heart disease.")

model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("Model or scaler files not found. Please ensure `model/heart_model.pth` and `model/scaler.pkl` exist.")
    st.stop()

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
    # Preprocess input
    input_features = np.array([[
        age, sex, chest_pain_type, resting_blood_pressure,
        cholesterol, fasting_blood_sugar, resting_ecg,
        max_heart_rate, exercise_induced_angina, st_depression,
        st_slope, num_major_vessels, thalassemia
    ]])
    
    try:
        input_scaled = scaler.transform(input_features)
        input_tensor = torch.FloatTensor(input_scaled)
        
        # Inference
        with torch.no_grad():
            prediction = model(input_tensor).item()
        
        prob = prediction
        pred = 1 if prob > 0.5 else 0
        
        st.divider()
        if pred == 1:
            st.error(f"Prediction: **HEART DISEASE DETECTED** (Probability: {prob:.2f})")
        else:
            st.success(f"Prediction: **NO HEART DISEASE** (Probability: {prob:.2f})")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")
