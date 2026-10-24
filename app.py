
from pathlib import Path

import streamlit as st
from PIL import Image

from src.inference import load_artifacts, predict_image

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
)

st.title("🌿 Plant Disease Detection using CNN")
st.write("Upload a tomato leaf image for classification.")

st.warning(
    "Educational prototype only. Predictions are not a professional diagnosis."
)


@st.cache_resource
def get_model():
    return load_artifacts()


try:
    model, class_names = get_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

uploaded = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded leaf", use_container_width=True)

        # Use the exact shared inference function as predict.py
        probabilities = predict_image(model, class_names, image)

        order = probabilities.argsort()[::-1]
        best = int(order[0])

        st.subheader(f"Prediction: {class_names[best]}")
        st.metric(
            "Model confidence",
            f"{float(probabilities[best]) * 100:.2f}%",
        )

        st.subheader("Class probabilities")

        for index in order:
            score = float(probabilities[index])
            st.write(
                f"**{class_names[int(index)]}**: {score * 100:.2f}%"
            )
            st.progress(score)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
