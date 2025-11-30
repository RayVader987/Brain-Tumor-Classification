"""
train_results.py - Simple Accuracy Visualization (No Training History Needed)
Run: python train_results.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os
import cv2

# Config
IMG_SIZE = 224
DATASET_PATH = "./archive/Training"  # Use Training for validation-like accuracy
MODELS_DIR = "./models"
CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'notumor']

print("="*80)
print("📊 QUICK ACCURACY VISUALIZATION (VALIDATION SET)")
print("="*80)

# Load dataset
def load_dataset(dataset_path):
    images, labels = [], []
    for class_name in CLASS_NAMES:
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype('float32') / 255.0
            images.append(img)
            labels.append(class_name)
    return np.array(images), np.array(labels)

print("Loading dataset...")
X, y_names = load_dataset(DATASET_PATH)

# Encode labels
with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)
y = label_encoder.transform(y_names)

# Take 10% as validation (same as training split)
from sklearn.model_selection import train_test_split
_, X_val, _, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

print(f"Validation set: {len(X_val)} images")

# Load models
print("\nLoading models...")
resnet = load_model(os.path.join(MODELS_DIR, "resnet50_brain_tumor.h5"))
vgg16 = load_model(os.path.join(MODELS_DIR, "vgg16_brain_tumor.h5"))
custom_cnn = load_model(os.path.join(MODELS_DIR, "custom_cnn_brain_tumor.h5"))

with open(os.path.join(MODELS_DIR, "svm_brain_tumor.pkl"), "rb") as f:
    svm = pickle.load(f)
with open(os.path.join(MODELS_DIR, "rf_brain_tumor.pkl"), "rb") as f:
    rf = pickle.load(f)

# Get predictions
print("\nGetting predictions...")
y_pred_resnet = np.argmax(resnet.predict(X_val, verbose=0), axis=1)
y_pred_vgg16 = np.argmax(vgg16.predict(X_val, verbose=0), axis=1)
y_pred_custom = np.argmax(custom_cnn.predict(X_val, verbose=0), axis=1)

# Feature extraction for SVM/RF
feature_extractor = tf.keras.Model(inputs=resnet.input, outputs=resnet.layers[-3].output)
features = feature_extractor.predict(X_val, verbose=0)
y_pred_svm = svm.predict(features)
y_pred_rf = rf.predict(features)

# Calculate accuracies
acc_resnet = accuracy_score(y_val, y_pred_resnet) * 100
acc_vgg16 = accuracy_score(y_val, y_pred_vgg16) * 100
acc_custom = accuracy_score(y_val, y_pred_custom) * 100
acc_svm = accuracy_score(y_val, y_pred_svm) * 100
acc_rf = accuracy_score(y_val, y_pred_rf) * 100

print(f"\nResNet50:     {acc_resnet:.2f}%")
print(f"VGG16:        {acc_vgg16:.2f}%")
print(f"Custom CNN:   {acc_custom:.2f}%")
print(f"SVM:          {acc_svm:.2f}%")
print(f"Random Forest: {acc_rf:.2f}%")

# ============ VISUALIZATIONS ============

# Bar Chart
plt.figure(figsize=(12, 7))
models = ['ResNet50', 'VGG16', 'Custom CNN', 'SVM', 'Random Forest']
accuracies = [acc_resnet, acc_vgg16, acc_custom, acc_svm, acc_rf]

bars = plt.bar(models, accuracies,
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
               edgecolor='black', linewidth=2)

plt.axhline(y=90, color='red', linestyle='--', linewidth=2, label='90% Threshold', alpha=0.7)
plt.xlabel('Model', fontsize=14, fontweight='bold')
plt.ylabel('Validation Accuracy (%)', fontsize=14, fontweight='bold')
plt.title('Brain Tumor Classification - Training Performance', fontsize=16, fontweight='bold')
plt.ylim([0, 100])
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1.5,
             f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig('training_accuracy_bar_chart.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: training_accuracy_bar_chart.png")
plt.show()

print("="*80)
print("🎉 VISUALIZATION COMPLETE!")
print("="*80)