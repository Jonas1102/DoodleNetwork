import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# Load the trained model
model = load_model('doodle_recognition_model.h5')


# Function to load and preprocess the image
def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize to the input size used during training (28x28)
    img = cv2.resize(img, (28, 28))

    # Normalize the image (0-1 scale)
    img = img / 255.0

    # Reshape the image to match model input shape (1, 28, 28, 1)
    img = img.reshape(1, 28, 28, 1)

    return img


# Function to make a prediction
def predict_doodle(image_path):
    img = preprocess_image(image_path)

    # Make a prediction
    prediction = model.predict(img)

    # Get the predicted class (index of the highest probability)
    predicted_class = np.argmax(prediction)

    return predicted_class


# Path to the image you want to test
test_image_path = 'cloud.png'  # Change this to your image path

# Make the prediction
predicted_class = predict_doodle(test_image_path)

# Output the predicted class
if predicted_class == 0:
    object = "house"
if predicted_class == 1:
    object = "cloud"
print(f'The predicted class for the image is: {predicted_class} ({object})')
