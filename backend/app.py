from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load your trained model
model = load_model('monilia_cnn.h5')

# Define label mapping
label_map = {1: 'Monilia', 0: 'Healthy'}  

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if file:
        # Save temporarily
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        
        # Preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Important! Normalize same as training

        # Predict
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = label_map.get(predicted_class, "Unknown")
        
        # Delete temp file
        os.remove(filepath)
        
        return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5001, debug=True)