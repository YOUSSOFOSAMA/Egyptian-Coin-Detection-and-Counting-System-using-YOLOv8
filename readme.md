# 🤟 AI-Powered ASL Predictor

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge\&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge\&logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0-green?style=for-the-badge\&logo=opencv)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-orange?style=for-the-badge\&logo=scikit-learn)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-blue?style=for-the-badge)

A **real-time American Sign Language (ASL) alphabet recognition system** built with Python, MediaPipe, Scikit-Learn, OpenCV, and Streamlit.

The system extracts 3D hand landmarks using **MediaPipe**, applies landmark normalization and feature scaling, and uses a trained **Support Vector Machine (SVM)** classifier to recognize ASL alphabet signs in real time.

---

## 🌟 Overview

This project demonstrates a complete computer vision and machine learning pipeline for recognizing American Sign Language hand gestures.

The system processes hand images through **MediaPipe Hand Landmark Detection**, converts the detected landmarks into normalized numerical features, and classifies the resulting hand pose using an **SVM with an RBF kernel**.

A Streamlit interface provides an interactive way to capture hand poses, predict letters, and build complete words.

---

🌐 Live Demo
---

*https://egyptian-coin-detection-and-counting-system-using-yolov8-8vlct.streamlit.app/*


## 🚀 Features

* **Real-Time Hand Detection** using MediaPipe.
* **21 Hand Landmark Detection** with 3D coordinates.
* **Translation Normalization** using the wrist as the reference point.
* **Scale Normalization** to reduce sensitivity to different hand sizes and camera distances.
* **StandardScaler** for machine learning feature standardization.
* **SVM Classification** using an RBF kernel.
* **Real-Time ASL Prediction** through webcam input.
* **Interactive Word Builder** for combining predicted letters.
* **Undo and Clear Controls** for word construction.
* **Prediction Visualization** directly on the camera feed.
* **MySQL Integration** for storing prediction records.
* **Streamlit Web Interface** for an interactive user experience.
* **Cloud Deployment Support** through Streamlit Community Cloud.

---

## 🧠 How It Works

### 1. Hand Landmark Detection

The webcam frame is processed using MediaPipe Hands to detect **21 landmarks** on the user's hand.

Each landmark contains:

```text
x coordinate
y coordinate
z coordinate
```

This produces:

```text
21 landmarks × 3 coordinates = 63 features
```

---

### 2. Landmark Normalization

Raw landmark coordinates are normalized before being passed to the classifier.

The preprocessing pipeline is:

```text
Camera Frame
     ↓
MediaPipe Hand Detection
     ↓
21 Hand Landmarks
     ↓
Wrist-Based Translation Normalization
     ↓
Hand-Scale Normalization
     ↓
63-Dimensional Feature Vector
     ↓
StandardScaler
     ↓
SVM Classifier
     ↓
ASL Letter
```

### Why normalization?

Normalization reduces the influence of factors that should not change the meaning of a gesture, such as:

* Hand position in the image
* Different hand sizes
* Different distances from the camera

This allows the classifier to focus more on the **relative geometric structure of the hand**.

> The exact normalization function used during training must also be used during prediction.

---

## 🤖 Machine Learning Model

The project uses a **Support Vector Machine (SVM)** classifier with an RBF kernel.

```python
SVC(
    kernel="rbf",
    probability=True
)
```

Before training, the extracted features are standardized using:

```python
StandardScaler()
```

The scaler is fitted using the training data and saved for later use during inference.

```text
Training:

Features
   ↓
StandardScaler.fit_transform()
   ↓
SVM Training
```

During prediction:

```text
New Features
   ↓
Saved StandardScaler.transform()
   ↓
SVM Prediction
```

The scaler is **never fitted again during inference**.

---

## 📊 Dataset

The model was trained using an ASL alphabet image dataset containing multiple gesture categories.

The feature extraction pipeline processes the dataset and converts images containing detectable hands into numerical landmark representations.

The extracted features and labels are stored for model training.

---

## 🗂️ Repository Structure

```text
asl-sign-language-recognition/
│
├── app.py                   # Streamlit application
│
├── feature_extraction.py    # MediaPipe landmark extraction
|
├── predict.py               # Real-time inference
|
├── train.py                 # Model training
|
├── asl_classifier.pkl       # Trained SVM + label encoder
|
|── scaler.pkl               # Fitted StandardScaler
│
├── dataset/
│   └── ...                      # Training dataset
├── training.ipynb
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/asl-sign-language-recognition.git
cd asl-sign-language-recognition
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```cmd
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Launch the Streamlit application with:

```bash
python -m streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

## 🏋️ Model Training

If you want to reproduce the training pipeline:

### Step 1 — Extract Hand Features

```bash
python src/feature_extraction.py
```

### Step 2 — Train the SVM

```bash
python src/train.py
```

The trained model and scaler will be saved:

---

## 🗄️ MySQL Integration

The original application also supports MySQL integration for storing prediction records.

The database can store information such as:

* Predicted letter
* Captured image
* Prediction history

For security and portability, database credentials should be stored in environment variables rather than hard-coded in the source code.

---

## 📐 System Architecture

```text
                 ┌───────────────────┐
                 │   Webcam / Image  │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │    MediaPipe      │
                 │   Hand Detection  │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ 21 Hand Landmarks │
                 │    (x, y, z)      │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │    Landmark       │
                 │   Normalization   │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   StandardScaler  │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   SVM Classifier   │
                 │      RBF Kernel    │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   ASL Prediction  │
                 └─────────┬─────────┘
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
          ┌───────────────┐  ┌──────────────┐
          │ Word Builder  │  │ MySQL Logging│
          └───────────────┘  └──────────────┘
```

---

## 📈 Model Evaluation

The training pipeline evaluates the classifier using:

* Training Accuracy
* Test Accuracy
* Confusion Matrix

The confusion matrix is generated using the test set to visualize classification performance across the ASL alphabet classes.

---

## 🔮 Future Improvements

* Improve robustness to hand rotation and perspective changes.
* Add support for two-hand gestures.
* Add temporal gesture recognition for dynamic signs.
* Improve prediction smoothing for real-time inference.
* Add confidence-based prediction filtering.
* Replace local MySQL dependency with an optional cloud database.
* Expand the system beyond alphabet signs to complete ASL words and phrases.
