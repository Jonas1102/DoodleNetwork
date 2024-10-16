import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt


# Step 1: Load and Preprocess the Images
def load_images_from_folder(folder, img_size=(28, 28)):
    images = []
    labels = []
    class_names = os.listdir(folder)  # List of class folders
    for label, class_folder in enumerate(class_names):
        class_path = os.path.join(folder, class_folder)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image as grayscale
                if img is not None:
                    img = cv2.resize(img, img_size)  # Resize image to 28x28
                    images.append(img)
                    labels.append(label)  # Assign numeric label to each class folder
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, class_names


# Step 2: Preprocess the Data (normalize, reshape, one-hot encode)
def preprocess_data(X, y, num_classes):
    # Normalize images to range [0, 1]
    X = X / 255.0
    # Reshape to include channel dimension (1 for grayscale)
    X = X.reshape(X.shape[0], 28, 28, 1)
    # One-hot encode the labels
    y = to_categorical(y, num_classes)
    return X, y


# Step 3: Define the Neural Network Model
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Optional: Add dropout to prevent overfitting
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Step 4: Visualize Training Results
def plot_training(history):
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.show()


# Step 5: Main Function to Run the Program
def main(folder_path):
    # Load and preprocess the data
    X, y, class_names = load_images_from_folder(folder_path)
    num_classes = len(class_names)
    X, y = preprocess_data(X, y, num_classes)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    input_shape = (28, 28, 1)
    model = create_model(input_shape, num_classes)

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10000, batch_size=32)

    # Plot training results
    plot_training(history)

    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save the model for future use
    model.save('10k.h5')
    print(f"Model saved as '10k.h5'")


# Run the script if this file is executed
if __name__ == "__main__":
    folder_path = "doodles"  # <-- CHANGE this to your doodle folder path
    main(folder_path)
