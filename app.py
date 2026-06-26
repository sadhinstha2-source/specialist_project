# =============================================================
# MODULE 6: app.py
# Project: Early Detection of Skin Cancer Using CNN
# Company: TechHealthCare Solutions Ltd
# Description: Flask web application for skin cancer detection
# =============================================================

import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from predict import load_trained_model, predict

# -------------------------------------------------------
# App Configuration
# -------------------------------------------------------
app            = Flask(__name__)
app.secret_key = 'skincancer_techhealthcare_2024'

UPLOAD_FOLDER  = './static/uploads'
ALLOWED_EXT    = {'png', 'jpg', 'jpeg'}
MODEL_PATH     = './models/Custom_CNN.keras'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at startup
print("[INFO] Loading model at startup...")
model = load_trained_model(MODEL_PATH)
print("[INFO] Model ready.")

# -------------------------------------------------------
# Helper: Check File Extension
# -------------------------------------------------------
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# -------------------------------------------------------
# Route: Home Page
# -------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

# -------------------------------------------------------
# Route: Predict
# -------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename  = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        result = predict(save_path, model)

        return jsonify({
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'risk'      : result['risk'],
            'advice'    : result['advice'],
            'all_probs' : result['all_probs'],
            'image_path': f'/static/uploads/{filename}'
        })

    return jsonify({'error': 'Invalid file type. Use PNG or JPG.'}), 400

# -------------------------------------------------------
# Run App
# -------------------------------------------------------
if __name__ == '__main__':
    print("=" * 55)
    print("  SKIN CANCER DETECTION - FLASK WEB APP")
    print("  TechHealthCare Solutions Ltd")
    print("  Running at: http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True) 