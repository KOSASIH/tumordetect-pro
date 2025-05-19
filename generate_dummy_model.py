import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Constants
IMG_WIDTH, IMG_HEIGHT = 150, 150
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'tumor_model.h5')

# Create model directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model'), exist_ok=True)

def create_dummy_model():
    """Create a dummy CNN model for testing purposes"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # 4 classes: No Tumor, Glioma, Meningioma, Pituitary
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save the model
    model.save(MODEL_PATH)
    print(f"Dummy model created and saved to {MODEL_PATH}")
    
    return model

if __name__ == "__main__":
    print("Generating dummy model for testing...")
    create_dummy_model()
    print("Done!")