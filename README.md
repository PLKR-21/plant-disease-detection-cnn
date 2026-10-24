# Plant Disease Detection Using CNN

An intermediate-level Deep Learning lab project that classifies leaf images into the labeled classes present in a local dataset. It uses a custom Convolutional Neural Network (CNN), TensorFlow/Keras, scikit-learn metrics, and an optional Streamlit interface.

## What is included
- `train.py`: trains the CNN, saves the best model, learning curves, and basic metrics.
- `evaluate.py`: generates a classification report and confusion matrix on the held-out test split.
- `predict.py`: predicts the top classes for one image from the command line.
- `app.py`: optional Streamlit upload interface.
- `src/model.py`: CNN definition.
- `src/data_utils.py`: image loading, splits, normalization, and labels.
- `report/Plant_Disease_Detection_Report.pdf`: academic report template.
- `presentation/Plant_Disease_Detection_Presentation.pptx`: 5-slide presentation.
- `notebooks/Plant_Disease_Detection_CNN.ipynb`: notebook starter walkthrough.

> Important: The dataset, trained model weights, and measured results are not bundled. Dataset downloads can be large and have separate terms; the model must be trained on your machine. Do not report invented accuracy values. After running training/evaluation, use the files generated under `outputs/` to fill in the report and presentation.

## 1. Environment setup
Recommended: Python 3.10 or 3.11. Create a virtual environment:

### Windows
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Dataset setup
Use the PlantVillage dataset:
- Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease
- GitHub: https://github.com/spMohanty/PlantVillage-Dataset

Download the dataset from the source and extract it. Place image files under `data/raw/`, with one folder per class, for example:

```text
data/raw/
  Tomato___healthy/
    image001.jpg
    image002.jpg
  Tomato___Early_blight/
    image003.jpg
  Tomato___Late_blight/
    image004.jpg
```

Choose 3–5 classes for a quick lab run. The loader discovers class folders automatically. Ensure there are enough images per class and that classes are not empty. For a more rigorous experiment, create a fixed split manifest and remove duplicate/near-duplicate images across train/validation/test sets. This project uses a reproducible random split but does not perform duplicate detection automatically.

## 3. Train and evaluate
From the project root:

```bash
python train.py
python evaluate.py
```

The loader allocates 70% of the images to training and splits the remaining 30% into validation and test subsets. The exact per-class counts can vary with the folder contents. For a formal report, record the number of images per class and verify class balance before training.

Generated outputs:
- `models/plant_disease_cnn.keras`
- `models/class_names.json`
- `outputs/learning_curves.png`
- `outputs/metrics.json`
- `outputs/classification_report.json`
- `outputs/confusion_matrix.png`

If the dataset is very small, `evaluate.py` may report too few batches. Increase the dataset size or reduce the batch size in both scripts. For best scientific practice, use a dedicated test split that is never used to choose hyperparameters.

## 4. Predict a single image
```bash
python predict.py path/to/leaf_image.jpg
```

## 5. Run the web app
Train the model first, then:

```bash
streamlit run app.py
```

## Model design
- Input: RGB image resized to 128 × 128 and normalized to [0, 1].
- Augmentation: random horizontal flip, small rotation, and zoom during training.
- Convolution blocks: 32, 64, and 128 filters with ReLU.
- Pooling and batch normalization: reduce spatial size and stabilize learning.
- Global average pooling and dense layer: aggregate features.
- Dropout: regularization.
- Softmax output: one probability per class.
- Optimizer: Adam; loss: sparse categorical cross-entropy; metric: accuracy.
- Early stopping and learning-rate reduction are used during training.

## Metrics to report
Accuracy, per-class precision/recall/F1, macro and weighted averages, confusion matrix, training time, and any observed overfitting. Use actual test results from your run. Accuracy on the PlantVillage dataset may not transfer to field images because backgrounds, lighting, camera quality, and disease appearance differ.

## Troubleshooting
- `ModuleNotFoundError`: activate the virtual environment and install `requirements.txt`.
- `Dataset folder not found`: check that `data/raw/` contains class subfolders.
- TensorFlow installation issues: use Python 3.10/3.11 and follow the official TensorFlow install guidance for your operating system.
- Out-of-memory: reduce batch size to 16 or image size to 96 × 96, and update the code consistently.
- Missing model in Streamlit: run `python train.py` before `streamlit run app.py`.

## Academic integrity / limitations
This is a learning prototype, not a certified agricultural diagnostic tool. State the exact dataset subset, split, software versions, and measured metrics in the submitted report. Do not claim field validation unless it was actually performed.
