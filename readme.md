#Hand Gesture Recognition using MediaPipe and Python

This is a small project that recognizes hand gestures through a webcam and speaks the detected gesture aloud. It uses MediaPipe to track hand landmarks and a simple machine learning model to classify gestures.

The goal was to build something lightweight that can run even on devices like a Raspberry Pi without needing a GPU or large deep-learning frameworks.

Instead of training on raw images, the program uses hand landmark coordinates. MediaPipe extracts 21 points from the hand, each containing x, y, and z coordinates. Because of that, the dataset stays small and training becomes much faster compared to image-based models.

Once everything is set up, the workflow is simple: capture gesture data, train the model, and then run the real-time recognition script.

#How It Works

The system works in three stages.

First, gesture samples are collected. The capture script opens the webcam, detects the hand using MediaPipe, and records the landmark coordinates. Each sample is stored as a small .npy file inside a folder named after the gesture.

Next, the training script loads those files, trains a RandomForest classifier, and saves the trained model.

Finally, the main program runs the camera again and predicts gestures in real time. When the model recognizes a gesture with enough confidence, it prints the result and speaks it using text-to-speech.

If the hand leaves the camera frame and comes back again, the system resets so the gesture can be detected again.

#Project Structure

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
│    └──gesture_model.pkl
└── dataset/

Each gesture has its own folder inside the dataset directory, and each folder contains the saved landmark samples.

#Requirements

You will need Python 3 and the following libraries:

mediapipe==0.10.9
opencv
opencv-contrib
numpy
python <3.12 & python >3.10
pyttsx3==2.99
scikit-learn==1.7.2
pandas==2.3.3

Install the required libraries using pip:

pip install opencv-python mediapipe numpy scikit-learn pyttsx3

If you are running this on Linux or Raspberry Pi, you may also need to install espeak for text-to-speech support.

sudo apt install espeak

#steps in running the program

Step 1: Capture Gesture Data

Run the capture script:

python3 capture.py

The program will ask you for a gesture name.

Example:

Enter gesture name: thumbs_up

Your webcam will open. Press C to start capturing samples. The script automatically records 100 landmark samples and saves them in the dataset folder.

You can repeat this process for multiple gestures such as:

thumbs_up
thumbs_down
peace

Try to slightly change the angle and distance of your hand while capturing samples. If every sample looks identical, the model may struggle to recognize gestures later.

Step 2: Train the Model

Once you have collected gesture samples, run:

python3 train.py

The script scans the dataset folder, loads every .npy file, and trains a RandomForest classifier.

After training finishes, the model is saved as:

gesture_model.pkl

This file contains the trained classifier and the gesture labels.

Step 3: Run Gesture Recognition

Start the real-time recognition:

python3 main.py

Your webcam will open and the system will begin detecting hand gestures.

When a gesture is recognized with enough confidence, it will appear on the screen and be spoken using text-to-speech.

If you keep showing the same gesture, it will only say it once. If you remove your hand from the camera and show it again, the system will detect it again.

Notes About Training Data

This system can work with a relatively small dataset, but the quality of the samples matters more than the number.

Try to capture gestures in slightly different positions and lighting conditions so the model learns variations instead of memorizing one exact pose.

If all samples look identical, the model may start classifying every hand position as the same gesture.

Running on Raspberry Pi

The program can run on a Raspberry Pi 4 (4GB) without major issues. MediaPipe is the most CPU-intensive part, but it still performs well enough for real-time gesture detection.

You can expect roughly 10–20 frames per second depending on the camera used.

Lowering the camera resolution can help if performance becomes an issue.

