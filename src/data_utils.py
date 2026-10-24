from pathlib import Path
import json
import tensorflow as tf

IMAGE_SIZE = (128, 128)
SEED = 42

def make_datasets(data_dir="data/raw", batch_size=32, image_size=IMAGE_SIZE):
    """Load class-folder images and create reproducible 70/15/15 splits.

    Expected structure:
      data/raw/
        class_a/*.jpg
        class_b/*.jpg
        ...
    The source directory must contain one folder per class.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}. Read README.md for setup instructions."
        )
    class_dirs = [p for p in data_dir.iterdir() if p.is_dir()]
    if len(class_dirs) < 2:
        raise ValueError(
            f"Expected at least 2 class folders in {data_dir}, found {len(class_dirs)}."
        )

    common = dict(
        directory=str(data_dir),
        labels="inferred",
        label_mode="int",
        class_names=sorted([p.name for p in class_dirs]),
        image_size=image_size,
        batch_size=batch_size,
        seed=SEED,
        shuffle=True,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        validation_split=0.30, subset="training", **common
    )
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        validation_split=0.30, subset="validation", **common
    )
    # Split the 30% holdout equally into validation and test (15% each).
    holdout_batches = tf.data.experimental.cardinality(val_test_ds).numpy()
    if holdout_batches < 2:
        raise ValueError(
            "Too few batches for a separate validation and test set. "
            "Use more images or lower batch_size."
        )
    val_batches = max(1, holdout_batches // 2)
    val_ds = val_test_ds.take(val_batches)
    test_ds = val_test_ds.skip(val_batches)

    class_names = sorted([p.name for p in class_dirs])
    Path("models").mkdir(exist_ok=True)
    with open("models/class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=autotune,
    ).prefetch(autotune)
    val_ds = val_ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=autotune,
    ).prefetch(autotune)
    test_ds = test_ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=autotune,
    ).prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names
