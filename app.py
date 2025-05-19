import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from flask import Flask, request, render_template, jsonify, send_from_directory
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

app = Flask(__name__)

# Constants
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'tumor_model.h5')

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model'), exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class labels
CLASSES = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to create and train model if it doesn't exist
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save the model
    model.save(MODEL_PATH)
    return model

# Load or create model
def get_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        return create_model()

# Preprocess image for prediction
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Generate visualization
def generate_visualization(img_array, prediction):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display the image
    ax1.imshow(img_array[0])
    ax1.set_title('Input MRI Scan')
    ax1.axis('off')
    
    # Create a bar chart for predictions
    bars = ax2.bar(CLASSES, prediction[0])
    ax2.set_title('Tumor Classification Probabilities')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Probability')
    
    # Color the highest probability bar differently
    max_idx = np.argmax(prediction[0])
    for i, bar in enumerate(bars):
        if i == max_idx:
            bar.set_color('red')
        else:
            bar.set_color('blue')
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the bytes buffer to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Load model
        model = get_model()
        
        # Preprocess image
        processed_img = preprocess_image(file_path)
        
        # Make prediction
        prediction = model.predict(processed_img)
        
        # Get the class with highest probability
        class_idx = np.argmax(prediction[0])
        class_name = CLASSES[class_idx]
        confidence = float(prediction[0][class_idx])
        
        # Generate visualization
        viz_img = generate_visualization(processed_img, prediction)
        
        return jsonify({
            'class': class_name,
            'confidence': confidence,
            'visualization': viz_img,
            'probabilities': {CLASSES[i]: float(prediction[0][i]) for i in range(len(CLASSES))}
        })
    
    return jsonify({'error': 'File type not allowed'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)