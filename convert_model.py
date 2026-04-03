import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

h5_path = os.path.join(BASE_DIR, "models", "best_ecg_model.h5")
keras_path = os.path.join(BASE_DIR, "models", "best_ecg_model.keras")

model = tf.keras.models.load_model(h5_path, compile=False)

model.save(keras_path)

print("✅ Converted successfully")