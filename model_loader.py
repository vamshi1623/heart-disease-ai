import tensorflow as tf
import os
import joblib
from tensorflow.keras.models import load_model

# Load models locally
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cnn_model = load_model(os.path.join(BASE_DIR, "models", "best_ecg_model.h5"))
xgb_model = joblib.load(os.path.join(BASE_DIR, "models", "heart_xgb_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "models", "label_encoder.pkl"))
