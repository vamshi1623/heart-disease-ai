import streamlit as st
import tempfile
import os
from predictor import FusionHeartModel

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = FusionHeartModel(
    os.path.join(BASE_DIR, "models", "best_ecg_model.h5"),
    os.path.join(BASE_DIR, "models", "heart_xgb_model.pkl"),
    os.path.join(BASE_DIR, "models", "scaler.pkl")
)

st.title("AI Heart Disease Risk Predictor")

st.header("Enter Clinical Data")

# --- KEEP SOME OLD UI BUT MAP THEM ---

age = st.slider("Age", 20, 80, 50)

sex = st.selectbox("Sex", ["Female", "Male"])
gender = 1 if sex == "Male" else 2   # ✅ mapped

# Replace irrelevant features with meaningful ones
height = st.number_input("Height (cm)", 140, 210, 170)
weight = st.number_input("Weight (kg)", 40, 150, 70)

trestbps = st.number_input("Systolic BP", 80, 200, 120)
ap_hi = trestbps

ap_lo = st.number_input("Diastolic BP", 50, 130, 80)

chol_option = st.selectbox(
    "Cholesterol Level",
    ["Normal", "Above Normal", "Well Above Normal"]
)

chol_map = {
    "Normal": 1,
    "Above Normal": 2,
    "Well Above Normal": 3
}

chol = chol_map[chol_option]

gluc_option = st.selectbox(
    "Glucose Level",
    ["Normal", "Above Normal", "Well Above Normal"]
)

gluc_map = {
    "Normal": 1,
    "Above Normal": 2,
    "Well Above Normal": 3
}

gluc = gluc_map[gluc_option]
smoke_option = st.selectbox("Do you smoke?", ["No", "Yes"])
smoke = 1 if smoke_option == "Yes" else 0
alco_option = st.selectbox("Alcohol Intake?", ["No", "Yes"])
alco = 1 if alco_option == "Yes" else 0
active_option = st.selectbox("Do you exercise regularly?", ["Yes", "No"])
active = 1 if active_option == "Yes" else 0
# --- REMOVE OLD FEATURES ---
# cp, slope, restecg, thalch, oldpeak ❌

st.header("Upload ECG Image")

uploaded_file = st.file_uploader("Upload ECG", type=["png","jpg","jpeg"])

if st.button("Predict Heart Risk"):

    if uploaded_file is None:
        st.warning("Please upload an ECG image")
    else:
        # Save temp image
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(uploaded_file.read())
        tmp.close()


        # Prepare input (MATCH MODEL)
        raw_features = [
            age, gender, height, weight,
            ap_hi, ap_lo,
            chol, gluc,
            smoke, alco, active
        ]

        # Predict
        result = model.predict(raw_features, tmp.name)

        st.subheader("Prediction Result")

        st.metric("Confidence", f"{result['Confidence']}%")

        st.write("ECG Status:", result["ECG_Status"])
        st.write("Condition:", result["Condition"])
        st.write("Final Disease:", result["Final_Disease"])
        st.write("Risk Level:", result["Risk_Level"])
