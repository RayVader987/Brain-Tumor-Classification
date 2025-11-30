# Brain Tumor MRI Scan Classification

Brain Tumor MRI Scan Classification is an end‑to‑end deep learning project that trains multiple models (ResNet50, VGG16, a custom CNN, SVM, and Random Forest) to automatically detect and classify brain tumors from MRI scans into four classes: **glioma**, **meningioma**, **pituitary**, and **no tumor**.

## Overview

Brain Tumor MRI Classification is a full pipeline for automated brain tumor detection and multi‑class classification using MRI images. It combines transfer learning, a custom CNN, and classical machine learning models to provide both high‑accuracy deep learning and lightweight baseline solutions.

---

## Features

- Classifies MRI scans into four classes: **glioma**, **meningioma**, **pituitary**, and **notumor**.  
- Trains **five** models in a single script:
  - ResNet50 (transfer learning)
  - VGG16 (transfer learning)
  - Custom CNN
  - SVM (on deep features)
  - Random Forest (on deep features)
- Uses transfer learning with **ImageNet‑pretrained ResNet50 and VGG16** as frozen feature extractors.
- Data augmentation (rotation, shifts, zoom, horizontal flip) to improve generalization.
- Early stopping and learning‑rate scheduling for stable training.
- Exports trained models and a label encoder for easy deployment in a separate backend (e.g., Flask/FastAPI app).

---

## Project Structure
NNFL_PROJECT/
├─ archive/
│ ├─ Training/
│ │ ├─ glioma/
│ │ ├─ meningioma/
│ │ ├─ pituitary/
│ │ └─ notumor/
│ └─ Testing/ # Optional extra test set (same subfolders)
│
├─ backend/
│ ├─ app.py # Backend API / inference server
│ └─ requirements.txt # Backend Python dependencies
│
├─ frontend/
│ └─ index.html # Simple UI for image upload & prediction
│
├─ models/ # Saved models (.h5, .pkl, label_encoder.pkl) – NOT tracked by git
│
├─ train_models.py # Main training script (trains 5 models)
├─ train_results.py # Helper for evaluation / plotting
├─ train_sv_rf.py # Helper for SVM / RF experiments
├─ results.py # Extra analysis / reporting logic
│
├─ confusion_matrices.png
├─ training_accuracy_bar_chart.png
├─ training_history.png
├─ test_accuracy_comparison.png
│
├─ .gitignore # Ignores venv, models, and large/binary artifacts
└─ README.md


---

## Data & Preprocessing

- **Input data**: brain tumor MRI images stored in per‑class folders under `./archive/Training`.
- **Classes**:
  - `glioma`
  - `meningioma`
  - `pituitary`
  - `notumor`
- **Preprocessing (in `train_models.py`)**:
  - Load images with OpenCV.
  - Convert BGR → RGB.
  - Resize to **224 × 224**.
  - Normalize pixel values to `[0, 1]`.
- **Label encoding**:
  - Class names are taken from folder names.
  - Encoded numerically using `sklearn.preprocessing.LabelEncoder`.
- **Train–test split**:
  - 90% training / 10% testing.
  - Stratified split to preserve class distribution.

---

## Models

### 1. ResNet50 (Transfer Learning)

- Base model:
tf.keras.applications.ResNet50(
weights="imagenet",
include_top=False,
input_shape=(224, 224, 3)
)

- Base is **frozen** (`trainable = False`).
- Custom head:
- `GlobalAveragePooling2D`
- `Dense(256, activation="relu")`
- `Dropout(0.5)`
- `Dense(num_classes, activation="softmax")`
- Loss: `sparse_categorical_crossentropy`  
- Optimizer: `Adam(learning_rate=1e-4)`

### 2. VGG16 (Transfer Learning)

- Base model:
tf.keras.applications.VGG16(
weights="imagenet",
include_top=False,
input_shape=(224, 224, 3)
)

- Base is **frozen**.
- Head architecture similar to ResNet50: global pooling → Dense(256) → Dropout(0.5) → Dense(num_classes).

### 3. Custom CNN

- Stack of Conv2D + MaxPooling2D blocks:
- Conv(32) → MaxPool
- Conv(64) → MaxPool
- Conv(128) → MaxPool
- Conv(256) → MaxPool
- Then:
- `Flatten`
- `Dense(512, activation="relu")`
- `Dropout(0.5)`
- `Dense(num_classes, activation="softmax")`
- Trained from scratch on the MRI dataset.

### 4. SVM (on ResNet Features)

- A Keras `Model` is built from the ResNet pipeline to output the **penultimate Dense layer** (e.g., 256‑dim feature vector).
- For each image:
- Extract feature vector with `feature_extractor.predict(...)`.
- Train `sklearn.svm.SVC`:
- `kernel="rbf"`, `gamma="scale"`, `C=1.0`, `probability=True`.
- Saved as:
- `models/svm_brain_tumor.pkl`

### 5. Random Forest (on ResNet Features)

- Uses the **same 256‑dim ResNet features**.
- Train `sklearn.ensemble.RandomForestClassifier`:
- `n_estimators=100`, `n_jobs=-1`, `random_state=42`.
- Saved as:
- `models/rf_brain_tumor.pkl`

All models evaluate on the held‑out test set. Test accuracies are printed at the end and compared across models.

---

## Training

### 1. Set Up Environment
From project root (recommended: use a virtual environment)
cd backend
pip install -r requirements.txt
cd ..


### 2. Prepare Dataset

Download the brain tumor MRI dataset (with four classes) and place the folders as:

archive/
└─ Training/
├─ glioma/
├─ meningioma/
├─ pituitary/
└─ notumor/


Optionally, you can also place an additional test set under `archive/Testing/` with the same structure for manual evaluation.

### 3. Run Training

From the **project root**:

python train_models.py


The script will:

- Load and preprocess all images.
- Split data into training and testing sets.
- Train:
  - ResNet50
  - VGG16
  - Custom CNN
  - SVM (using ResNet features)
  - Random Forest (using ResNet features)
- Save trained models and label encoder into `./models/`.
- Generate plots such as:
  - `confusion_matrices.png`
  - `training_accuracy_bar_chart.png`
  - `training_history.png`
  - `test_accuracy_comparison.png`

---

## Saved Artifacts (in `models/`)

After successful training, you should have:

- `resnet50_brain_tumor.h5`
- `vgg16_brain_tumor.h5`
- `custom_cnn_brain_tumor.h5`
- `svm_brain_tumor.pkl`
- `rf_brain_tumor.pkl`
- `label_encoder.pkl`

> These files are **large** and should not be committed to git. Keep `models/` in `.gitignore`. For sharing, zip the folder (e.g., `brain_tumor_models.zip`) and upload it to a release or external storage, then link it here.

---

## Inference / Backend

Once models are trained and saved under `models/`, you can run the backend server:

cd backend

(optional) activate your venv
python app.py


`app.py` loads the trained models and exposes an API endpoint that accepts an MRI image, runs it through the selected model (e.g., ResNet50 / VGG16 / Custom CNN / SVM / RF), and returns:

- Predicted tumor class (glioma / meningioma / pituitary / no tumor)
- Optionally, prediction probabilities per class

---

## Frontend

The frontend in `frontend/index.html` is a minimal UI to:

- Upload an MRI image.
- Call the backend prediction API.
- Display the predicted class and (optionally) confidence scores.

This frontend can be deployed on platforms like **Vercel** or any static hosting service, while the Python backend runs on a separate server (e.g., Render, Railway, EC2, etc.).

---

## Future Improvements

- Add support for Grad‑CAM / heatmaps to visualize tumor‑relevant regions.  
- Experiment with fine‑tuning upper layers of ResNet50/VGG16 for higher accuracy.  
- Add cross‑validation and more metrics (precision, recall, F1‑score, ROC curves).  
- Containerize the backend with Docker for easier deployment.


## How to download the trained models from this repo?
1. Download brain_tumor_models.zip from the Releases page
2. Unzip it into the models/ folder in the project root
3. Then run:  python backend/app.py
