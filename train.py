from pathlib import Path
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from src.data_utils import make_datasets
from src.model import build_cnn

EPOCHS = 15
BATCH_SIZE = 32
DATA_DIR = "data/raw"
MODEL_PATH = "models/plant_disease_cnn.keras"

def main():
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    train_ds, val_ds, test_ds, class_names = make_datasets(
        data_dir=DATA_DIR, batch_size=BATCH_SIZE
    )
    print("Classes:", class_names)
    model = build_cnn(num_classes=len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks
    )
    model.save(MODEL_PATH)

    # Save learning curves for the lab report.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("outputs/learning_curves.png", dpi=160)
    plt.close(fig)

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "classes": class_names,
        "epochs_completed": len(history.history["loss"]),
    }
    with open("outputs/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print("Next: python evaluate.py")

if __name__ == "__main__":
    main()
