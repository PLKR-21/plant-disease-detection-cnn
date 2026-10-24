from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.data_utils import make_datasets

MODEL_PATH = "models/plant_disease_cnn.keras"

def main():
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run: python train.py")
    _, _, test_ds, class_names = make_datasets()
    model = tf.keras.models.load_model(MODEL_PATH)
    y_true, y_pred = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())

    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0
    )
    print(classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0
    ))
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(max(7, len(class_names)), max(6, len(class_names) * 0.7)))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    ax.set_title("Plant Disease CNN - Confusion Matrix")
    fig.tight_layout()
    fig.savefig("outputs/confusion_matrix.png", dpi=160)
    plt.close(fig)
    print("Saved outputs/classification_report.json and outputs/confusion_matrix.png")

if __name__ == "__main__":
    main()
