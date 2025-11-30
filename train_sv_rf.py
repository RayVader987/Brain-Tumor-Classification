"""
Train ONLY SVM and Random Forest using features from the already trained ResNet50 model.

Prerequisites (from previous training):
- models/resnet50_brain_tumor.h5
- models/label_encoder.pkl   (optional; will be recreated if missing)
- Dataset in ./brain-tumor-mri-dataset/Training/ with 4 folders:
  ['glioma', 'meningioma', 'pituitary', 'notumor']

Run: python train_svm_rf_only.py
"""

import os
import numpy as np
import cv2
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

# ---------------- CONFIG ----------------
IMG_SIZE = 224
TEST_SIZE = 0.1
RANDOM_STATE = 42

DATASET_PATH = './archive/Training'
MODELS_DIR = './models'
CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'notumor']

os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 80)
print("🧠 TRAINING SVM + RANDOM FOREST (USING TRAINED RESNET50 FEATURES)")
print("=" * 80)

# ---------------- OPTIONAL GPU SETUP ----------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("✅ GPU detected and memory growth enabled:", gpus)
    except Exception as e:
        print("⚠️ Could not set GPU memory growth:", e)
else:
    print("⚠️ No GPU visible to TensorFlow – running on CPU")

# ---------------- LOAD DATASET ----------------
def load_dataset(dataset_path):
    images = []
    labels = []
    print("\n📂 Loading dataset from:", dataset_path)

    for class_name in CLASS_NAMES:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            print(f"⚠️ Warning: {class_path} not found, skipping.")
            continue

        print(f"   Loading {class_name}...", end=" ")
        count = 0
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype('float32') / 255.0
            images.append(img)
            labels.append(class_name)
            count += 1
        print(f"{count} images")

    images = np.array(images, dtype='float32')
    labels = np.array(labels)
    print(f"\n✅ Total images loaded: {len(images)}")
    print("   Image shape:", images.shape[1:])
    return images, labels

X, y = load_dataset(DATASET_PATH)

# ---------------- LABEL ENCODING ----------------
print("\n📦 Loading / creating label encoder...")
label_encoder_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')
if os.path.exists(label_encoder_path):
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    print("   ✅ Loaded existing label_encoder.pkl")
    y_encoded = label_encoder.transform(y)
else:
    print("   ⚠️ label_encoder.pkl not found, creating new one.")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print("   ✅ New label_encoder.pkl saved")

# ---------------- TRAIN / TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print("\n📊 Data split for SVM/RF:")
print(f"   Training: {len(X_train)} images")
print(f"   Testing:  {len(X_test)} images")

# ---------------- LOAD TRAINED RESNET50 ----------------
resnet_path = os.path.join(MODELS_DIR, 'resnet50_brain_tumor.h5')
if not os.path.exists(resnet_path):
    raise FileNotFoundError(f"❌ {resnet_path} not found. Make sure ResNet50 was trained and saved.")

print("\n📦 Loading trained ResNet50 model...")
resnet_model = load_model(resnet_path)
print("   ✅ ResNet50 loaded.")

# Build feature extractor from the penultimate dense layer (256-dim)
feature_extractor = Model(
    inputs=resnet_model.input,
    outputs=resnet_model.layers[-3].output  # Dense(256) layer
)
test_feat = feature_extractor.predict(X_train[:1], verbose=0)
print("   ✅ Feature extractor built. Example feature shape:", test_feat.shape)

# ---------------- BATCHED FEATURE EXTRACTION ----------------
def extract_features_in_batches(model, X, batch_size=16):
    """Extract features in small batches to avoid RAM issues."""
    features_list = []
    n = X.shape[0]
    for i in range(0, n, batch_size):
        batch = X[i:i+batch_size]
        feats = model.predict(batch, verbose=0)
        features_list.append(feats)
        print(f"\r   Processed {min(i+batch_size, n)}/{n} images", end="")
    print()
    return np.concatenate(features_list, axis=0)

print("\n🔍 Extracting features for training set...")
X_train_features = extract_features_in_batches(feature_extractor, X_train, batch_size=16)
print("   Training features shape:", X_train_features.shape)

print("\n🔍 Extracting features for test set...")
X_test_features = extract_features_in_batches(feature_extractor, X_test, batch_size=16)
print("   Test features shape:", X_test_features.shape)

# ---------------- TRAIN SVM ----------------
print("\n[1/2] 🚀 Training SVM on ResNet features...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=RANDOM_STATE)
svm_model.fit(X_train_features, y_train)

svm_path = os.path.join(MODELS_DIR, 'svm_brain_tumor.pkl')
with open(svm_path, 'wb') as f:
    pickle.dump(svm_model, f)
print(f"   💾 SVM model saved to {svm_path}")

y_pred_svm = svm_model.predict(X_test_features)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"   ✅ SVM Test Accuracy: {acc_svm*100:.2f}%")

# ---------------- TRAIN RANDOM FOREST ----------------
print("\n[2/2] 🚀 Training Random Forest on ResNet features...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train_features, y_train)

rf_path = os.path.join(MODELS_DIR, 'rf_brain_tumor.pkl')
with open(rf_path, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"   💾 Random Forest model saved to {rf_path}")

y_pred_rf = rf_model.predict(X_test_features)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"   ✅ Random Forest Test Accuracy: {acc_rf*100:.2f}%")

print("\n" + "=" * 80)
print("🎯 SVM + RANDOM FOREST TRAINING COMPLETE")
print("=" * 80)