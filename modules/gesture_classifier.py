"""
gesture_classifier.py

This module is responsible for classifying hand gestures based on
landmark coordinates produced by MediaPipe.

Input:
    Hand landmarks (21 points per hand)

Output:
    Gesture label (example: "thumbs_up", "peace", "stop")

The classifier loads a trained machine learning model
from the models directory and uses it to predict gestures.
"""

import pickle
import numpy as np


class GestureClassifier:
    """
    GestureClassifier handles loading the trained model
    and predicting gestures from hand landmarks.
    """

    def __init__(self, model_path="models/gesture_model.pkl"):
        """
        Constructor

        Loads the trained gesture recognition model from disk.

        """

        # Load the trained model from file
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def preprocess_landmarks(self, landmarks):
        """
        Convert landmark coordinates into a format
        suitable for the machine learning model.

        MediaPipe returns landmarks as:
            [[x1,y1], [x2,y2], ... [x21,y21]]

        ML models expect a flat vector:
            [x1,y1,x2,y2,...x21,y21]

        Returns
        -------
        numpy array
            Flattened landmark vector
        """

        flattened = []

        for point in landmarks:
            flattened.append(point[0])  # x coordinate
            flattened.append(point[1])  # y coordinate

        return np.array(flattened).reshape(1, -1)

    def predict(self, landmarks):
        """
        Predict gesture from hand landmarks.

        Returns
        -------
        str
            Predicted gesture label
        """

        # Convert landmarks to model input format
        input_data = self.preprocess_landmarks(landmarks)

        # Predict gesture using trained model
        prediction = self.model.predict(input_data)

        # Return predicted label
        return prediction[0]