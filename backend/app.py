"""
Brain Tumor Classification - Flask Web Application
"""

from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

# ===== FIX: CONFIGURE PATHS FOR YOUR FOLDER STRUCTURE =====
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Go up to "New folder (2)"

template_folder = os.path.join(BASE_DIR, 'frontend')
static_folder = os.path.join(BASE_DIR, 'frontend', 'static')
models_folder = os.path.join(BASE_DIR, 'models')
upload_folder = os.path.join(static_folder, 'uploads')

# Configuration
UPLOAD_FOLDER = upload_folder
MODELS_FOLDER = models_folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
IMG_SIZE = 224

# Create Flask app with custom folders
app = Flask(__name__, 
            template_folder=template_folder,
            static_folder=static_folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("=" * 80)
print("🧠 BRAIN TUMOR CLASSIFICATION WEB APP")
print("=" * 80)
print(f"📁 Templates: {template_folder}")
print(f"📁 Static: {static_folder}")
print(f"📁 Models: {MODELS_FOLDER}")
print(f"📁 Uploads: {UPLOAD_FOLDER}")
print("=" * 80)

# Load models at startup
print("\n📦 Loading YOUR trained models...")

# ... rest of your code stays the same ...

# Load CNN models
try:
    model_resnet = tf.keras.models.load_model(os.path.join(MODELS_FOLDER, 'resnet50_brain_tumor.h5'))
    print("   ✅ ResNet50 loaded")
except Exception as e:
    print(f"   ❌ ResNet50 failed: {e}")
    model_resnet = None

try:
    model_vgg = tf.keras.models.load_model(os.path.join(MODELS_FOLDER, 'vgg16_brain_tumor.h5'))
    print("   ✅ VGG16 loaded")
except Exception as e:
    print(f"   ❌ VGG16 failed: {e}")
    model_vgg = None

try:
    model_cnn = tf.keras.models.load_model(os.path.join(MODELS_FOLDER, 'custom_cnn_brain_tumor.h5'))
    print("   ✅ Custom CNN loaded")
except Exception as e:
    print(f"   ❌ Custom CNN failed: {e}")
    model_cnn = None

# Load SVM and Random Forest
try:
    with open(os.path.join(MODELS_FOLDER, 'svm_brain_tumor.pkl'), 'rb') as f:
        model_svm = pickle.load(f)
    print("   ✅ SVM loaded")
except Exception as e:
    print(f"   ❌ SVM failed: {e}")
    model_svm = None

try:
    with open(os.path.join(MODELS_FOLDER, 'rf_brain_tumor.pkl'), 'rb') as f:
        model_rf = pickle.load(f)
    print("   ✅ Random Forest loaded")
except Exception as e:
    print(f"   ❌ Random Forest failed: {e}")
    model_rf = None

# Load label encoder
try:
    with open(os.path.join(MODELS_FOLDER, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    print("   ✅ Label encoder loaded")
    CLASS_NAMES = label_encoder.classes_.tolist()
except Exception as e:
    print(f"   ❌ Label encoder failed: {e}")
    CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'notumor']

# Feature extractor for SVM/RF (from YOUR trained ResNet50)
feature_extractor = None
if model_resnet is not None:
    try:
        # Extract features from the penultimate Dense layer (same as training)
        feature_extractor = Model(
            inputs=model_resnet.input,
            outputs=model_resnet.layers[-3].output  # Dense(256) layer before Dropout
        )
        print("   ✅ Feature extractor ready (using YOUR ResNet50)")
    except Exception as e:
        print(f"   ❌ Feature extractor failed: {e}")
        feature_extractor = None
else:
    print("   ⚠️ Feature extractor unavailable (ResNet50 not loaded)")

print("\n✅ Model loading complete!")
print(f"   Classes: {CLASS_NAMES}")
print("=" * 80)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image exactly like training"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0  # Match training normalization
    return img

def extract_features(img_array):
    """Extract features for SVM/RF using YOUR trained ResNet50"""
    if feature_extractor is None:
        return None
    features = feature_extractor.predict(np.expand_dims(img_array, axis=0), verbose=0)
    return features.flatten().reshape(1, -1)  # Flatten to match training

@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Validate file
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use JPG/PNG only'}), 400
        
        # Get model selection
        model_name = request.form.get('model', 'ensemble').lower()
        
        # Save uploaded file
        filename = f"pred_{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        img_array = preprocess_image(filepath)
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Run prediction based on selected model
        if model_name == 'ensemble':
            # ENSEMBLE: Average predictions from all available models
            all_predictions = []
            models_used = []
            
            if model_resnet is not None:
                preds = model_resnet.predict(img_batch, verbose=0)[0]
                all_predictions.append(preds)
                models_used.append("ResNet50")
            
            if model_vgg is not None:
                preds = model_vgg.predict(img_batch, verbose=0)[0]
                all_predictions.append(preds)
                models_used.append("VGG16")
            
            if model_cnn is not None:
                preds = model_cnn.predict(img_batch, verbose=0)[0]
                all_predictions.append(preds)
                models_used.append("Custom CNN")
            
            if model_svm is not None and feature_extractor is not None:
                features = extract_features(img_array)
                preds = model_svm.predict_proba(features)[0]
                all_predictions.append(preds)
                models_used.append("SVM")
            
            if model_rf is not None and feature_extractor is not None:
                features = extract_features(img_array)
                preds = model_rf.predict_proba(features)[0]
                all_predictions.append(preds)
                models_used.append("Random Forest")
            
            if len(all_predictions) == 0:
                return jsonify({'error': 'No models available'}), 500
            
            # Average all predictions
            predictions = np.mean(all_predictions, axis=0)
            model_used = f"Ensemble ({len(models_used)} models: {', '.join(models_used)})"
            
        elif model_name == 'resnet50' and model_resnet is not None:
            predictions = model_resnet.predict(img_batch, verbose=0)[0]
            model_used = "ResNet50"
            
        elif model_name == 'vgg16' and model_vgg is not None:
            predictions = model_vgg.predict(img_batch, verbose=0)[0]
            model_used = "VGG16"
            
        elif model_name == 'custom_cnn' and model_cnn is not None:
            predictions = model_cnn.predict(img_batch, verbose=0)[0]
            model_used = "Custom CNN"
            
        elif model_name == 'svm' and model_svm is not None:
            if feature_extractor is None:
                return jsonify({'error': 'Feature extractor not available (ResNet50 needed)'}), 500
            features = extract_features(img_array)
            predictions = model_svm.predict_proba(features)[0]
            model_used = "SVM"
            
        elif model_name == 'rf' and model_rf is not None:
            if feature_extractor is None:
                return jsonify({'error': 'Feature extractor not available (ResNet50 needed)'}), 500
            features = extract_features(img_array)
            predictions = model_rf.predict_proba(features)[0]
            model_used = "Random Forest"
            
        else:
            return jsonify({'error': f'Model "{model_name}" not available'}), 400
        
        # Get prediction results
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx]) * 100
        
        # Format probabilities as percentages
        probabilities = {
            CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }
        
        # Prepare response
        response = {
            'prediction': predicted_class,
            'confidence': round(confidence, 2),
            'probabilities': probabilities,
            'model_used': model_used,
            'image_path': f'/static/uploads/{filename}',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"\n✅ Prediction: {predicted_class} ({confidence:.2f}%) - Model: {model_used}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    models_status = {
        'resnet50': model_resnet is not None,
        'vgg16': model_vgg is not None,
        'custom_cnn': model_cnn is not None,
        'svm': model_svm is not None,
        'rf': model_rf is not None,
        'feature_extractor': feature_extractor is not None
    }
    
    available_count = sum(models_status.values())
    
    return jsonify({
        'status': 'healthy',
        'models_loaded': available_count,
        'models': models_status,
        'classes': CLASS_NAMES,
        'ensemble_available': available_count >= 2
    })

@app.route('/models')
def models_info():
    """Return available models info"""
    available = []
    
    if model_resnet is not None:
        available.append({'name': 'resnet50', 'display': 'ResNet50'})
    if model_vgg is not None:
        available.append({'name': 'vgg16', 'display': 'VGG16'})
    if model_cnn is not None:
        available.append({'name': 'custom_cnn', 'display': 'Custom CNN'})
    if model_svm is not None and feature_extractor is not None:
        available.append({'name': 'svm', 'display': 'SVM'})
    if model_rf is not None and feature_extractor is not None:
        available.append({'name': 'rf', 'display': 'Random Forest'})
    
    if len(available) >= 2:
        available.insert(0, {'name': 'ensemble', 'display': 'Ensemble (All Models)'})
    
    return jsonify({'available_models': available})

if __name__ == '__main__':
    print("\n🚀 Starting Flask server...")
    print("📱 Open your browser at: http://localhost:5000")
    print("🔬 Available endpoints:")
    print("   - / (main page)")
    print("   - /predict (prediction API)")
    print("   - /health (health check)")
    print("   - /models (available models)")
    print("\nPress CTRL+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=5000)