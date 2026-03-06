"""
train.py

This script trains the gesture recognition model.

"""

import os
import cv2
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier


# Path to dataset
DATASET_PATH = "dataset"

# Path where trained model will be saved
MODEL_PATH = "models/gesture_model.pkl"

# Image size used during training
IMG_SIZE = 224


def load_dataset():
    """
    Loads images from dataset folder and converts them into training data.
    """

    X = []  # features
    y = []  # labels

    gestures = os.listdir(DATASET_PATH)

    print("Loading dataset...\n")

    for gesture in gestures:

        gesture_folder = os.path.join(DATASET_PATH, gesture)

        if not os.path.isdir(gesture_folder):
            continue

        print(f"Processing gesture: {gesture}")

        for img_name in os.listdir(gesture_folder):

            img_path = os.path.join(gesture_folder, img_name)

            img = cv2.imread(img_path)

            if img is None:
                continue

            # Resize to fixed size
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Flatten image to 1D vector
            img = img.flatten()

            X.append(img)
            y.append(gesture)

    return np.array(X), np.array(y)


def train_model(X, y):
    """
    Train RandomForest classifier
    """

    print("\nTraining model...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    print("Training complete")

    return model


def save_model(model):
    """
    Save trained model to file
    """

    os.makedirs("models", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {MODEL_PATH}")


def main():

    X, y = load_dataset()

    print(f"\nTotal samples: {len(X)}")

    model = train_model(X, y)

    save_model(model)


if __name__ == "__main__":
    main()