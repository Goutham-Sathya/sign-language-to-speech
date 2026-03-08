# Hand Gesture Recognition using MediaPipe and Python

Lightweight real-time hand-gesture recognition using MediaPipe landmarks and a simple ML classifier (RandomForest). The app captures landmark coordinates from a webcam, trains a small model on those landmarks, and performs live recognition with text-to-speech feedback. Designed to run on low-power devices (Raspberry Pi) without GPU or heavy DL frameworks.

Badges

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)


## Features

- Real-time hand tracking via MediaPipe (21 landmarks per hand)
- Lightweight classifier using landmark coordinates (RandomForest)
- Text-to-speech announcement of recognized gestures
- Small dataset footprint (npy landmark samples) and fast training
- Runs on Raspberry Pi (approx. 10–20 FPS depending on camera/resolution)

## Requirements

Supported Python versions: Python 3.10 or 3.11 (tested). Avoid Python < 3.10 or 3.12+ unless you verify package compatibility.

Recommended: use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate     # Windows
```

Install required Python packages (correct package names included):

```bash
pip install mediapipe==0.10.9 opencv-python opencv-contrib-python numpy scikit-learn==1.7.2 pyttsx3==2.99 pandas==2.3.3
```

Notes:
- On Raspberry Pi or some Linux systems you may need system packages for OpenCV and audio (ALSA). For TTS, install espeak:

```bash
sudo apt update
sudo apt install espeak libasound2-dev
```

If you run into MediaPipe installation issues on ARM (Raspberry Pi), consult the MediaPipe docs or use prebuilt wheels for your platform.

## steps in running the program

Quick start — three steps:

1) Capture gesture samples

```bash
python3 capture.py
```

- The script prompts for a gesture name (e.g., "thumbs_up").
- Press C to start capturing. By default it records 100 landmark samples and saves them under dataset/<gesture_name>/
- Capture multiple variations (angle, distance, lighting) to improve generalization.

2) Train the model

```bash
python3 train.py
```

- Scans dataset/ for .npy landmark files, trains a RandomForest classifier, and saves the model to models/gesture_model.pkl.

3) Run real-time recognition

```bash
python3 main.py
```

- Opens webcam, predicts gestures in real time, and uses text-to-speech to announce recognized gestures (pyttsx3/espeak).
- The system only announces repeated gestures once; removing the hand and re-showing it resets detection so it can be announced again.

### Usage/Examples

Example capture session:

1. python3 capture.py
2. Enter gesture name: thumbs_up
3. Press 'C' to record 100 landmark samples

Training and recognition:

```bash
python3 train.py       # produces models/gesture_model.pkl
python3 main.py        # starts webcam-based recognition
```

If you want fewer/more samples or custom parameters, update the capture.py/training config in config.py (see project config).

## Project Structure
```bash
project_folder/
│
├── main.py
├── train.py
├── config.py
│
├── modules/
│   ├── __init__.py
│   ├── hand_tracker.py
│   ├── gesture_classifier.py
│   └── tts_engine.py
│
├── models/
│    └── gesture_model.pkl
└── dataset/
```
Each gesture has its own folder inside the dataset directory, and each folder contains the saved landmark samples (npy files).

## How It Works

The system works in three stages.

1) Data collection: capture.py opens the webcam, detects a single hand via MediaPipe (21 landmarks), and saves landmark coordinate arrays as .npy files grouped by gesture name.

2) Training: train.py loads all .npy samples, encodes labels, trains a RandomForest classifier on the 63-dimensional feature vector (21 points × x,y,z), and saves the model and label mapping to models/gesture_model.pkl.

3) Inference: main.py opens the webcam, obtains landmarks per frame, uses the trained model to predict gestures with confidence. If confidence is above a threshold, the gesture is shown on-screen and announced with text-to-speech. When the hand leaves and returns, detection can re-trigger for the same gesture.

Notes About Training Data

- A small dataset can work, but diversity matters: capture slight angle, distance and lighting variations.
- Avoid capturing identical-looking samples — the model may overfit and misclassify.
- If you need more robustness, collect 200–500 samples per gesture and include negative samples (no-gesture) or multiple people/hand sizes.

Running on Raspberry Pi

- The app can run on Raspberry Pi 4 (4GB). MediaPipe CPU usage is the bottleneck.
- Expect ~10–20 FPS depending on camera and resolution. Lower resolution for better speed.
- If MediaPipe is slow, consider reducing the landmark detection frequency or using a faster camera backend.
- Ensure espeak/system audio is installed for TTS.

Acknowledgements & Resources

- MediaPipe docs: https://developers.google.com/mediapipe
- OpenCV: https://opencv.org
- pyttsx3 documentation: https://pyttsx3.readthedocs.io

Thanks to the authors of MediaPipe and OpenCV for the underlying tools.