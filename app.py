from flask import Flask, request, jsonify
import joblib
import numpy as np
import cv2
from skimage.feature import hog
import os

# Flask uygulaması
app = Flask(__name__)

# Model ve seçilmiş özellikleri yükle
MODEL_PATH = "pso_model.pkl"  # en iyi modelinizin adını buraya yazın
FEATURES_PATH = "pso_features.npy"

model = joblib.load(MODEL_PATH)
selected_features = np.load(FEATURES_PATH)

# HOG özelliği çıkarma fonksiyonu
def extract_hog_features(img):
    hog_features = hog(img, orientations=8, pixels_per_cell=(16, 16),
                       cells_per_block=(1, 1), block_norm='L2-Hys')
    return hog_features

# Görüntüyü işle
def process_image(file_storage):
    # Görüntüyü oku ve griye çevir
    file_bytes = file_storage.read()
    img_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None, "Görüntü okunamadı"

    # Boyutlandır
    img = cv2.resize(img, (128, 128))
    
    # Özellikleri çıkar
    features = extract_hog_features(img)
    selected = features[selected_features]

    return selected.reshape(1, -1), None

@app.route('/')  # 👈 Kök dizin için route ekleyin
def home():
    return "API çalışıyor!"

# Ana API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Görüntü dosyası eksik'}), 400

    file = request.files['image']
    features, error = process_image(file)

    if error:
        return jsonify({'error': error}), 400

    # Tahmin
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]  # hasarlı olma olasılığı

    return jsonify({
        'tahmin': 'Hasarlı' if prediction == 1 else 'Hasarsız',
        'hasar_orani': round(float(probability) * 100, 2)
    })
    
@app.route('/health',methods=["GET"]) #bu kısmı kod bloğunda ekle

def get_health():

    return {"status":"running"}
    
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
