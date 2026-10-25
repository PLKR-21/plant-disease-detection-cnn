# 🌿 Plant Disease Detection Using CNN

A deep learning project that classifies tomato leaf images into disease and healthy categories using a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras. The project includes a trained model, prediction scripts, evaluation outputs, and an interactive Streamlit web application.

## 🚀 Live Application

Try the deployed application:

**[🌱 Plant Disease Detection App](https://tomatoleafdiseasedetection.streamlit.app/)**

Upload a tomato leaf image to receive a predicted class and its associated confidence score.

## 🧪 Supported Classes

The model supports the following four categories:

- **Tomato — Bacterial Spot**
- **Tomato — Early Blight**
- **Tomato — Healthy**
- **Tomato — Late Blight**

## ✨ Features

- CNN-based tomato leaf classification.
- Interactive image upload and prediction through Streamlit.
- Prediction confidence scores.
- Model training and evaluation scripts.
- Classification report and confusion matrix outputs.
- Reusable inference functionality for local predictions and the web application.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | CNN model development and inference |
| Streamlit | Interactive web application |
| NumPy | Numerical operations |
| Matplotlib | Visualization of training and evaluation results |
| scikit-learn | Model evaluation metrics |

## 📁 Project Structure

```text
plant-disease-detection-cnn/
├── app.py
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── inference.py
│   └── model.py
├── models/
│   ├── plant_disease_cnn.keras
│   └── class_names.json
├── outputs/
│   ├── classification_report.json
│   ├── confusion_matrix.png
│   ├── learning_curves.png
│   └── metrics.json
├── notebooks/
├── data/
│   └── raw/
└── README.md
```

*Note: The structure above highlights the main project files. Some generated files and local-only folders may not be included in the repository.*

## ⚙️ Installation and Setup

### Prerequisites

- Python compatible with the TensorFlow version in `requirements.txt`
- Git
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/PLKR-21/plant-disease-detection-cnn.git
cd plant-disease-detection-cnn
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**

```cmd
.venv\Scripts\activate.bat
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

## 🧠 Model Training

The project uses a Convolutional Neural Network to learn visual patterns from tomato leaf images.

To train the model, first prepare the dataset in the directory structure expected by the training script. Then run:

```bash
python train.py
```

The training process generates a trained model and associated class information according to the project's configuration.

## 📊 Model Evaluation

To evaluate the trained model, run:

```bash
python evaluate.py
```

The evaluation script can generate performance metrics and visualizations, depending on the configured dataset and output paths.

## 🔍 Local Prediction

Use the prediction script to classify an image:

```bash
python predict.py --help
```

Follow the arguments shown by the script to provide the input image path.

## 📚 Dataset

The model was developed using tomato leaf image categories from the PlantVillage dataset.

The four classes used in this project are Bacterial Spot, Early Blight, Healthy, and Late Blight.

**Dataset note:** Dataset images are not included in the repository. Obtain the dataset separately and organize it according to the paths expected by the training and evaluation scripts.

## 📓 Model Implementation Notebook

The Jupyter Notebook demonstrates the complete CNN implementation, including dataset preprocessing, model architecture, training, accuracy and loss visualization, model evaluation, confusion matrix, and sample image prediction.

**[View the CNN Model Implementation Notebook](notebooks/Plant_Disease_Detection_CNN_Model_Implementation.ipynb)**

## ⚠️ Limitations

- The model is limited to the four tomato leaf categories used during training.
- Predictions may be affected by image quality, lighting, backgrounds, and differences between training images and real-world images.
- A high confidence score does not guarantee that a prediction is correct.
- This project is an educational prototype and should not replace professional agricultural diagnosis.

## 📄 License

This project is licensed under the **MIT License**. See the [`LICENSE`](LICENSE) file for the full license terms.

## 👨‍💻 Author

**P. Laxmi Kanth Reddy**

- GitHub: [PLKR-21](https://github.com/PLKR-21)
- Repository: [Plant Disease Detection Using CNN](https://github.com/PLKR-21/plant-disease-detection-cnn)
- Live Application: [Plant Disease Detection App](https://tomatoleafdiseasedetection.streamlit.app/)