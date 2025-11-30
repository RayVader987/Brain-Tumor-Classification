"""
Brain Tumor MRI Classification 
Trains 5 models: ResNet50, VGG16, Custom CNN, SVM, Random Forest
Run: python train_models.py
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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
TEST_SIZE = 0.1  # 90/10 split
RANDOM_STATE = 42

# Dataset paths (YOUR structure after mv archive brain-tumor-mri-dataset)
DATASET_PATH = './archive/Training'
MODELS_DIR = './models'
CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'notumor']

print("=" * 80)
print("🧠 BRAIN TUMOR CLASSIFICATION - MODEL TRAINING")
print("=" * 80)

# Create models directory
os.makedirs(MODELS_DIR, exist_ok=True)

# Load and preprocess dataset
def load_dataset(dataset_path):
    """Load images and labels from dataset folders"""
    images = []
    labels = []
    
    print("\n📂 Loading dataset from:", dataset_path)
    
    for class_name in CLASS_NAMES:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            print(f"⚠️  Warning: {class_path} not found!")
            continue
            
        print(f"   Loading {class_name}...", end=" ")
        count = 0
        
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img / 255.0
                    
                    images.append(img)
                    labels.append(class_name)
                    count += 1
                except Exception as e:
                    continue
        
        print(f"{count} images")
    
    return np.array(images), np.array(labels)

# Load data
X, y = load_dataset(DATASET_PATH)
print(f"\n✅ Total images loaded: {len(X)}")
print(f"   Image shape: {X.shape[1:]}")
print(f"   Classes: {CLASS_NAMES}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Save label encoder
with open(os.path.join(MODELS_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)
print("\n💾 Label encoder saved!")

# Train/test split (90/10)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
)
print(f"\n📊 Data split:")
print(f"   Training: {len(X_train)} images")
print(f"   Testing: {len(X_test)} images")

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)

# Results storage
results = {}

print("\n" + "=" * 80)
print("🚀 TRAINING MODELS")
print("=" * 80)

# ============================================================================
# MODEL 1: ResNet50
# ============================================================================
print("\n[1/5] 🔥 Training ResNet50...")
base_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_resnet.trainable = False

model_resnet = Sequential([
    base_resnet,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model_resnet.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_resnet = model_resnet.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

model_resnet.save(os.path.join(MODELS_DIR, 'resnet50_brain_tumor.h5'))
y_pred_resnet = np.argmax(model_resnet.predict(X_test), axis=1)
acc_resnet = accuracy_score(y_test, y_pred_resnet)
results['ResNet50'] = acc_resnet
print(f"✅ ResNet50 saved! Test Accuracy: {acc_resnet*100:.2f}%")

# ============================================================================
# MODEL 2: VGG16
# ============================================================================
print("\n[2/5] 🔥 Training VGG16...")
base_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_vgg.trainable = False

model_vgg = Sequential([
    base_vgg,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model_vgg.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_vgg = model_vgg.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

model_vgg.save(os.path.join(MODELS_DIR, 'vgg16_brain_tumor.h5'))
y_pred_vgg = np.argmax(model_vgg.predict(X_test), axis=1)
acc_vgg = accuracy_score(y_test, y_pred_vgg)
results['VGG16'] = acc_vgg
print(f"✅ VGG16 saved! Test Accuracy: {acc_vgg*100:.2f}%")

# ============================================================================
# MODEL 3: Custom CNN
# ============================================================================
print("\n[3/5] 🔥 Training Custom CNN...")
model_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model_cnn.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_cnn = model_cnn.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

model_cnn.save(os.path.join(MODELS_DIR, 'custom_cnn_brain_tumor.h5'))
y_pred_cnn = np.argmax(model_cnn.predict(X_test), axis=1)
acc_cnn = accuracy_score(y_test, y_pred_cnn)
results['Custom CNN'] = acc_cnn
print(f"✅ Custom CNN saved! Test Accuracy: {acc_cnn*100:.2f}%")

# ============================================================================
# MODEL 4: SVM (using ResNet50 features) - ✅ FIXED FEATURE EXTRACTION
# ============================================================================
print("\n[4/5] 🔥 Training SVM...")
# ✅ FIXED: Use model_resnet (256-dim features) NOT base_resnet (100k-dim)
feature_extractor = Model(inputs=model_resnet.input, outputs=model_resnet.layers[-3].output)

X_train_features = feature_extractor.predict(X_train, verbose=1)
X_test_features = feature_extractor.predict(X_test, verbose=1)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=RANDOM_STATE)
svm_model.fit(X_train_features, y_train)

with open(os.path.join(MODELS_DIR, 'svm_brain_tumor.pkl'), 'wb') as f:
    pickle.dump(svm_model, f)

y_pred_svm = svm_model.predict(X_test_features)
acc_svm = accuracy_score(y_test, y_pred_svm)
results['SVM'] = acc_svm
print(f"✅ SVM saved! Test Accuracy: {acc_svm*100:.2f}%")

# ============================================================================
# MODEL 5: Random Forest (using ResNet50 features)
# ============================================================================
print("\n[5/5] 🔥 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_model.fit(X_train_features, y_train)

with open(os.path.join(MODELS_DIR, 'rf_brain_tumor.pkl'), 'wb') as f:
    pickle.dump(rf_model, f)

y_pred_rf = rf_model.predict(X_test_features)
acc_rf = accuracy_score(y_test, y_pred_rf)
results['Random Forest'] = acc_rf
print(f"✅ Random Forest saved! Test Accuracy: {acc_rf*100:.2f}%")

# Final summary
print("\n" + "=" * 80)
print("🎯 TRAINING COMPLETE - ACCURACY COMPARISON")
print("=" * 80)
print(f"{'Model':<20} {'Test Accuracy':<15}")
print("-" * 40)
for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name:<20} {accuracy*100:>6.2f}%")
print("=" * 80)
print(f"\n💾 All models saved to: {MODELS_DIR}/")
print("📦 Files created:")
print("   - resnet50_brain_tumor.h5")
print("   - vgg16_brain_tumor.h5")
print("   - custom_cnn_brain_tumor.h5")
print("   - svm_brain_tumor.pkl")
print("   - rf_brain_tumor.pkl")
print("   - label_encoder.pkl")
print("\n✅ Ready to copy models/ folder to your laptop and run app.py!")