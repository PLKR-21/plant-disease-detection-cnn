import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("models/plant_disease_cnn.keras")
LABEL_PATH = Path("models/class_names.json")

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")
st.title("🌿 Plant Disease Detection using CNN")
st.write("Upload a leaf image to classify it among the classes used to train this model.")
st.warning(
    "Educational prototype only. Predictions are not a professional plant-health diagnosis. "
    "Performance may be lower on field images with different lighting or backgrounds."
)

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_PATH, encoding="utf-8") as f:
        labels = json.load(f)
    return model, labels

uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if not MODEL_PATH.exists() or not LABEL_PATH.exists():
        st.error("Model files are missing. Install the dataset and run `python train.py` first.")
    else:
        model, labels = load_artifacts()
        resized = image.resize((128, 128))
        array = np.asarray(resized, dtype=np.float32) / 255.0
        probs = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
        order = np.argsort(probs)[::-1]
        best = int(order[0])
        st.subheader(f"Prediction: {labels[best]}")
        st.metric("Model confidence", f"{probs[best] * 100:.1f}%")
        st.caption("Confidence is the model's probability output, not a guarantee of correctness.")
        st.write("Top predictions")
        for idx in order[:min(3, len(labels))]:
            st.write(f"**{labels[int(idx)]}** — {probs[int(idx)] * 100:.1f}%")
