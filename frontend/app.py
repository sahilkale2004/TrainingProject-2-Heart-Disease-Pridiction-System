import streamlit as st
import requests
import json

st.set_page_config(page_title="Heart Disease Predictor", layout="wide", initial_sidebar_state="collapsed")

# Inject Custom CSS for Premium Design
st.markdown("""
<style>
    /* Import modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(30, 30, 30) 0%, rgb(18, 18, 18) 90%);
        color: #f1f1f1;
    }

    /* Typography */
    h1 {
        font-weight: 700 !important;
        font-size: 3rem !important;
        background: -webkit-linear-gradient(45deg, #ff4b4b, #ff8a8a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem !important;
        padding-top: 2rem !important;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #a0a0a0;
        margin-bottom: 3rem;
    }

    /* Form Container Glassmorphism */
    [data-testid="stForm"] {
        background: rgba(40, 40, 40, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        max-width: 900px;
        margin: 0 auto;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    [data-testid="stForm"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
    }

    /* Input Fields styling */
    .stNumberInput > div > div > input, 
    .stSelectbox > div > div > div {
        background-color: rgba(30,30,30, 0.6) !important;
        border: 1px solid rgba(255,255,255, 0.1) !important;
        color: #fff !important;
        border-radius: 10px !important;
        padding: 8px 12px !important;
        transition: all 0.2s ease !important;
    }
    
    .stNumberInput > div > div > input:focus, 
    .stSelectbox > div > div > div:focus-within {
        border: 1px solid #ff4b4b !important;
        box-shadow: 0 0 10px rgba(255, 75, 75, 0.3) !important;
    }

    /* Labels */
    .stNumberInput label, .stSelectbox label {
        color: #e0e0e0 !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
        font-size: 0.95rem !important;
    }

    /* Submit Button */
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(90deg, #ff4b4b 0%, #ff2a2a 100%);
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border-radius: 30px !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        margin-top: 1.5rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }

    [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.6);
        background: linear-gradient(90deg, #ff5c5c 0%, #ff3b3b 100%);
    }
    
    /* Success/Error Alerts */
    [data-testid="stAlert"] {
        border-radius: 15px !important;
        padding: 20px !important;
        font-weight: 600 !important;
        border-left: 5px solid transparent;
    }
</style>
""", unsafe_allow_html=True)

st.title("❤️ Heart Disease Prediction System")
st.markdown("<p class='subtitle'>Advanced AI diagnostic tool for evaluating cardiac risk based on clinical parameters.</p>", unsafe_allow_html=True)

# Main Container Wrapper
with st.container():
    # Input Form
    with st.form("prediction_form"):
        st.markdown("<h3 style='color: #fff; margin-bottom: 20px;'><span style='color: #ff4b4b;'>|</span> Clinical Details</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50, help="Age of the patient in years")
            sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
            chest_pain_type = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic")
            resting_blood_pressure = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120, help="Resting blood pressure in mm Hg")
            cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200, help="Serum cholesterol in mg/dl")
            fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], help="1 = True, 0 = False")
        
        with col2:
            resting_ecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], help="0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy")
            max_heart_rate = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
            exercise_induced_angina = st.selectbox("Exercise Induced Angina", options=[0, 1], help="1 = Yes, 0 = No")
            st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0, step=0.1, help="ST depression induced by exercise relative to rest")
            st_slope = st.selectbox("ST Slope", options=[0, 1, 2], help="0: Upsloping, 1: Flat, 2: Downsloping")
            num_major_vessels = st.selectbox("Number of Major Vessels", options=[0, 1, 2, 3, 4], help="Number of major vessels (0-3) colored by flourosopy")
        
        thalassemia = st.selectbox("Thalassemia", options=[0, 1, 2, 3], help="1: Normal, 2: Fixed Defect, 3: Reversable Defect")
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button(label="Generate Prediction")

if submit_button:
    with st.spinner('Analyzing patient clinical data...'):
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
            # Deployed Render Backend URL
            backend_url = "https://trainingproject-2-heart-disease.onrender.com/predict"
            response = requests.post(backend_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                prob = result["heart_disease_probability"]
                pred = result["prediction"]
                
                # Render results in a nice card
                st.markdown("<br>", unsafe_allow_html=True)
                if pred == 1:
                    st.error(f"⚠️ **CRITICAL FINDING: HIGH RISK OF HEART DISEASE**\n\nThe model predicts a **{prob*100:.1f}% probability** of heart disease based on the provided clinical parameters. Immediate medical consultation is recommended.")
                else:
                    st.success(f"✅ **FINDING: LOW RISK OF HEART DISEASE**\n\nThe model predicts a **{prob*100:.1f}% probability** of heart disease. The clinical parameters appear within a safe range, but regular checkups are advised.")
            else:
                st.warning(f"Could not reach the backend. Status Code: {response.status_code}. Please ensure the Render web service is live.")
                
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")
