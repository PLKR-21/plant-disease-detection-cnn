import hashlib
import json
from pathlib import Path
from io import BytesIO

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Resolve paths relative to this app.py file
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_cnn.keras"
LABEL_PATH = BASE_DIR / "models" / "class_names.json"

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered",
)

st.title("🌿 Plant Disease Detection using CNN")
st.write(
    "Upload a tomato leaf image to predict whether it is healthy "
    "or affected by one of three diseases."
)

st.warning(
    "Educational prototype only. Predictions are not a professional "
    "plant-health diagnosis."
)


@st.cache_resource
def load_artifacts(model_path, label_path):
    model = tf.keras.models.load_model(model_path)

    with open(label_path, encoding="utf-8") as file:
        labels = json.load(file)

    return model, labels


if not MODEL_PATH.exists() or not LABEL_PATH.exists():
    st.error("Model files are missing. Check the models folder.")
    st.stop()

model, labels = load_artifacts(str(MODEL_PATH), str(LABEL_PATH))

uploaded = st.file_uploader(
    "Choose a tomato leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    # Read the current upload directly
    image_bytes = uploaded.getvalue()
    image_id = hashlib.sha256(image_bytes).hexdigest()[:12]

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        st.error("Could not read this image. Please upload another file.")
        st.stop()

    st.image(image, caption="Current uploaded image", use_container_width=True)

    # Display an identifier to verify that the upload changed
    st.caption(f"Image: {uploaded.name} | Image ID: {image_id}")

    # Match the preprocessing used by predict.py
    resized = image.resize((128, 128))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    batch = np.expand_dims(array, axis=0)

    probabilities = model.predict(batch, verbose=0)[0]

    if len(probabilities) != len(labels):
        st.error("The model output and class-name mapping do not match.")
        st.stop()

    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index]) * 100

    st.subheader(f"Prediction: {labels[best_index]}")
    st.metric("Model confidence", f"{confidence:.2f}%")

    st.subheader("All class probabilities")

    for index in np.argsort(probabilities)[::-1]:
        probability = float(probabilities[index])
        st.write(f"**{labels[index]}** — {probability * 100:.2f}%")
        st.progress(probability)

    st.caption(
        "Probabilities represent the model's outputs, "
        "not a guarantee of correctness."
    )