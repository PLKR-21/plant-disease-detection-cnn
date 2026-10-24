
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_cnn.keras"
LABEL_PATH = BASE_DIR / "models" / "class_names.json"


def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABEL_PATH, encoding="utf-8") as f:
        class_names = json.load(f)

    return model, class_names


def predict_image(model, class_names, image):
    image = image.convert("RGB").resize((128, 128))

    array = np.asarray(image, dtype=np.float32) / 255.0
    batch = np.expand_dims(array, axis=0)

    probabilities = model.predict(batch, verbose=0)[0]

    if len(probabilities) != len(class_names):
        raise ValueError("Model output does not match class names.")

    return probabilities
