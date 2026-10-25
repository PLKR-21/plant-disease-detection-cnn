# Plant Disease Detection Using CNN

A deep learning application that classifies tomato leaf images into four categories using a Convolutional Neural Network (CNN). The project uses TensorFlow/Keras for image classification and Streamlit for an interactive web interface.

## Live Application

**Try the application:** https://tomatoleafdiseasedetection.streamlit.app/

Upload a tomato leaf image to view the predicted disease class and the model's confidence scores.

## Features

- Classifies tomato leaves into four categories.
- Uses a custom CNN built with TensorFlow and Keras.
- Supports image uploads through a Streamlit interface.
- Displays the predicted class and class-wise probability scores.
- Includes model evaluation with a classification report, confusion matrix, and learning curves.
- Provides scripts for training, evaluation, and individual image prediction.

## Supported Classes

1. Tomato Healthy
2. Tomato Early Blight
3. Tomato Late Blight
4. Tomato Bacterial Spot

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Image Processing:** Pillow, NumPy
- **Web Interface:** Streamlit
- **Dataset:** PlantVillage

## Project Structure

```text
plant-disease-detection-cnn/
├── app.py
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── data_utils.py
│   └── inference.py
├── models/
│   ├── plant_disease_cnn.keras
│   └── class_names.json
├── data/
│   └── raw/
├── outputs/
│   ├── classification_report.json
│   ├── confusion_matrix.png
│   ├── learning_curves.png
│   └── metrics.json
└── notebooks/
    └── Plant_Disease_Detection_CNN.ipynb
```

## Installation

Python 3.13 is recommended for the local environment used during development.

Clone the repository:

```bash
git clone https://github.com/PLKR-21/plant-disease-detection-cnn.git
cd plant-disease-detection-cnn
```

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Download the PlantVillage dataset from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease).

Place the four required tomato class folders inside `data/raw/`:

```text
data/raw/
├── Tomato___healthy/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
└── Tomato___Bacterial_spot/
```

The dataset images are not included in this repository.

## Training

After placing the dataset in the expected directory, run:

```bash
python train.py
```

## Model Evaluation

Run:

```bash
python evaluate.py
```

Evaluation outputs are saved in the `outputs/` directory.

## Run the Web Application

Start the Streamlit app locally:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## Individual Image Prediction

Run the prediction script with an image path:

```bash
python predict.py path/to/leaf_image.jpg
```

## Limitations

- The model is trained on selected PlantVillage tomato leaf classes.
- Predictions on real-world images may be affected by lighting, background, image quality, and differences from the training dataset.
- Model confidence is not a guarantee of prediction accuracy.
- This application is an educational prototype and is not a substitute for professional plant-health diagnosis.

## License

No license has been specified for this repository. Contact the repository owner before reusing or redistributing the project.

## Author

**PLKR-21**

GitHub: https://github.com/PLKR-21