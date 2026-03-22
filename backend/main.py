from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np
import os

app = FastAPI()

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

# Global variables for model and scaler
model = None
scaler = None

@app.on_event("startup")
def load_model():
    global model, scaler
    model_path = "model/heart_model.pth"
    scaler_path = "model/scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        input_size = scaler.n_features_in_
        model = HeartNet(input_size)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Model and scaler loaded successfully.")
    else:
        print("Model or scaler not found. Please place them in the 'model/' folder.")

class HeartData(BaseModel):
    age: float
    sex: float
    chest_pain_type: float
    resting_blood_pressure: float
    cholesterol: float
    fasting_blood_sugar: float
    resting_ecg: float
    max_heart_rate: float
    exercise_induced_angina: float
    st_depression: float
    st_slope: float
    num_major_vessels: float
    thalassemia: float

@app.post("/predict")
def predict(data: HeartData):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check model/ folder.")
    
    # Preprocess
    input_features = np.array([[
        data.age, data.sex, data.chest_pain_type, data.resting_blood_pressure,
        data.cholesterol, data.fasting_blood_sugar, data.resting_ecg,
        data.max_heart_rate, data.exercise_induced_angina, data.st_depression,
        data.st_slope, data.num_major_vessels, data.thalassemia
    ]])
    
    input_scaled = scaler.transform(input_features)
    input_tensor = torch.FloatTensor(input_scaled)
    
    # Inference
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    return {
        "heart_disease_probability": prediction,
        "prediction": 1 if prediction > 0.5 else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
