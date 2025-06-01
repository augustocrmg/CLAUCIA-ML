import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Check if an image path was provided
if len(sys.argv) < 2:
    print("Usage: python predict_with_model.py <path_to_image>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found")
    sys.exit(1)

# Configuration
IMAGE_SIZE = (224, 224)
MODEL_PATH = "skin_condition_efficientnet_model.h5"
DATASET_PATH = "datasets/dataset_wagner_balanceado_2"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found. Train the model first using train_efficientnet.py")
    sys.exit(1)

# Get class names from directory structure
class_names = sorted([d for d in os.listdir(DATASET_PATH) 
                     if os.path.isdir(os.path.join(DATASET_PATH, d))])
NUM_CLASSES = len(class_names)

print(f"Classes: {class_names}")
print(f"Loading model from: {MODEL_PATH}")

# Load the trained model
try:
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Make prediction
def predict_image(model, image_path):
    # Load and preprocess the image
    img = keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Print results
    print("\n===== Prediction Results =====")
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")
    
    # Print all class probabilities
    print("\nProbabilities for all classes:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {predictions[0][i]:.4f}")
    
    # Show the image with predicted class
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[predicted_class]}\nConfidence: {confidence:.4f}")
    plt.axis("off")
    output_filename = 'prediction_result.png'
    plt.savefig(output_filename)
    print(f"\nPrediction visualization saved to '{output_filename}'")
    
    return class_names[predicted_class], confidence

# Make prediction on the provided image
predict_image(model, image_path) 