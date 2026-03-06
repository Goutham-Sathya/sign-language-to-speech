"""
config.py

This file stores all global configuration settings
for the hand gesture recognition project.

"""

# Camera Settings

# Camera resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Frame rate target
TARGET_FPS = 25

# Hand Detection Settings

# Maximum number of hands to detect
MAX_HANDS = 2

# MediaPipe detection confidence
DETECTION_CONFIDENCE = 0.7

# MediaPipe tracking confidence
TRACKING_CONFIDENCE = 0.7

# Dataset Settings

# Path to dataset folder
DATASET_PATH = "dataset"

# Number of samples per gesture (used during data collection)
SAMPLES_PER_GESTURE = 200

# Model Settings

# Path where trained model will be saved
MODEL_PATH = "models/gesture_model.pkl"

# Gesture Mapping File

# File that maps gestures to sentences
GESTURE_MAP_FILE = "gestures.json"

# Speech Settings

# Speech speed
TTS_RATE = 150

# Speech volume (0.0 to 1.0)
TTS_VOLUME = 1.0