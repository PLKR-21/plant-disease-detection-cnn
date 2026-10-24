import json
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/plant_disease_cnn.keras"
LABEL_PATH = "models/class_names.json"

def predict(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    image = Image.open(image_path).convert("RGB").resize((128, 128))
    array = np.asarray(image, dtype=np.float32) / 255.0
    probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
    ranked = np.argsort(probabilities)[::-1][:3]
    for idx in ranked:
        print(f"{class_names[idx]}: {probabilities[idx] * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/leaf_image.jpg")
        raise SystemExit(2)
    predict(sys.argv[1])
