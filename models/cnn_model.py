import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing import image

# =====================
# Dataset path
# =====================
dataset_path = r"C:\Users\vamshi\PycharmProjects\Heart\models\new_ECG"

# =====================
# Image Data Generator with augmentation
# =====================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# =====================
# Base Model (MobileNetV2)
# =====================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)
base_model.trainable = False

# =====================
# Model architecture
# =====================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# =====================
# Compile model
# =====================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =====================
# Callbacks
# =====================
checkpoint = ModelCheckpoint('best_ecg_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)

# =====================
# Train model
# =====================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# =====================
# Load best model and evaluate
# =====================
best_model = load_model('best_ecg_model.h5')
val_loss, val_acc = best_model.evaluate(val_generator)
print(f'Validation Accuracy: {val_acc*100:.2f}%')

# =====================
# Predict on a new image
# =====================
def predict_ecg(img_path, model, target_size=(128, 128)):
    # Load and display image
    img = image.load_img(img_path, target_size=target_size)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Preprocess
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = pred[0][class_idx] * 100

    class_names = list(train_generator.class_indices.keys())
    predicted_class = class_names[class_idx]

    print(f'Predicted Class: {predicted_class} ({confidence:.2f}% confidence)')

# Example usage
image1_path = r"C:\Users\vamshi\PycharmProjects\Heart\models\new_ECG\Abnormal_heartbeat\HB_(236).png"
predict_ecg(image1_path, best_model)

# =====================
# Save model
# =====================
best_model.save(r"C:\Users\vamshi\PycharmProjects\Heart\models\best_ecg_model.keras")
print("Model saved correctly")

# =====================
# Check TensorFlow version
# =====================
import tensorflow as tf
print(tf.__version__)