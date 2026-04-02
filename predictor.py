import joblib
import numpy as np
import os

import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
class FusionHeartModel:
    def __init__(self, cnn_path, xgb_path, scaler_path):
        # Load models and scaler
        self.cnn_model = load_model(cnn_path)
        self.tabular_model = joblib.load(xgb_path)
        self.scaler = joblib.load(scaler_path)
        self.features = joblib.load(os.path.join(os.path.dirname(__file__), "models", "features.pkl"))

    # CNN prediction
    def predict_cnn(self, img_path, target_size=(128,128)):
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        pred_probs = self.cnn_model.predict(img_array)[0]
        class_names = ['Normal_Person', 'Abnormal_heartbeat', 'History_of_MI']
        return class_names[np.argmax(pred_probs)], pred_probs

    # Tabular prediction


    def predict_tabular(self, raw_features):
            age, gender, height, weight, ap_hi, ap_lo, chol, gluc, smoke, alco, active = raw_features

            # Feature engineering
            BMI = weight / ((height / 100) ** 2)
            MAP = ap_lo + (1 / 3) * (ap_hi - ap_lo)
            risk_score = (
                    int(ap_hi > 140) +
                    int(chol == 3) +
                    int(gluc == 3) +
                    int(smoke == 1) +
                    int(alco == 1) +
                    int(active == 0)
            )

            # Create dataframe
            input_dict = {
                'age': age,
                'gender': gender,
                'height': height,
                'weight': weight,
                'ap_hi': ap_hi,
                'ap_lo': ap_lo,
                'cholesterol': chol,
                'gluc': gluc,
                'smoke': smoke,
                'alco': alco,
                'active': active,
                'BMI': BMI,
                'MAP': MAP,
                'risk_score': risk_score
            }

            input_df = pd.DataFrame([input_dict])

            # Ensure correct feature order
            input_df = input_df[self.features]

            # Scale numeric features
            num_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'BMI', 'MAP']
            input_df[num_cols] = self.scaler.transform(input_df[num_cols])

            # Predict
            pred_probs = self.tabular_model.predict_proba(input_df)[0]
            class_names = ['Normal', 'Hypertension', 'Cholesterol', 'Glucose', 'Lifestyle']

            return class_names[np.argmax(pred_probs)], pred_probs
    # Fusion logic
    def fusion_decision(self, cnn_label, cnn_probs, tab_label, tab_probs):
        ecg_status = 'Normal' if cnn_label=='Normal_Person' else 'Abnormal'
        if ecg_status=='Normal' and tab_label=='Normal':
            final_disease = 'Normal'
        elif tab_label=='Hypertension' and ecg_status=='Abnormal':
            final_disease = 'Hypertensive Heart Disease'
        elif tab_label=='Cholesterol' and ecg_status=='Abnormal':
            final_disease = 'Coronary Artery Disease'
        elif tab_label=='Glucose' and ecg_status=='Abnormal':
            final_disease = 'Diabetic Heart Disease'
        elif tab_label=='Lifestyle' and ecg_status=='Abnormal':
            final_disease = 'Lifestyle-induced Heart Risk'
        elif ecg_status=='Abnormal' and tab_label!='Normal':
            final_disease = 'High Risk Cardiac Condition'
        else:
            final_disease = 'At Risk / Monitor'

        if ecg_status=='Normal' and tab_label=='Normal':
            risk_level = 'Low'
        elif ecg_status=='Abnormal' or tab_label!='Normal':
            risk_level = 'Medium'
        elif ecg_status!='Normal' and sum(tab_probs>0.5)>1:
            risk_level = 'High'
        else:
            risk_level = 'Medium'

        cnn_conf, tab_conf = max(cnn_probs), max(tab_probs)
        final_confidence = 0.6*cnn_conf + 0.4*tab_conf

        return {"ECG_Status": ecg_status,
                "Condition": tab_label,
                "Final_Disease": final_disease,
                "Risk_Level": risk_level,
                "Confidence": round(final_confidence*100,2)}

    # Single function to call everything
    def predict(self, raw_features, ecg_path):
        cnn_label, cnn_probs = self.predict_cnn(ecg_path)
        tab_label, tab_probs = self.predict_tabular(raw_features)
        return self.fusion_decision(cnn_label, cnn_probs, tab_label, tab_probs)