import cv2
import mediapipe as mp
import numpy as np
import os
import sys

#comment check if user provided gesture name
if len(sys.argv) != 2:
    print("Usage: python3 capture.py <gesture_name>")
    exit()

gesture_name = sys.argv[1]

#comment dataset root directory
DATASET_DIR = "dataset"

#comment create dataset directory if it doesn't exist
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

#comment create gesture directory
gesture_dir = os.path.join(DATASET_DIR, gesture_name)

if not os.path.exists(gesture_dir):
    os.makedirs(gesture_dir)

print("Saving data to:", gesture_dir)

#comment mediapipe hand setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

#comment open camera
cap = cv2.VideoCapture(0)

count = 0
target_samples = 100

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    #comment flip for natural camera view
    frame = cv2.flip(frame, 1)

    #comment convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            #comment draw landmarks on screen
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            #comment extract landmark coordinates
            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            landmarks = np.array(landmarks)

            #comment save landmark file
            file_path = os.path.join(gesture_dir, f"{count}.npy")
            np.save(file_path, landmarks)

            count += 1

            print("Saved sample:", count)

    #comment display counter
    cv2.putText(
        frame,
        f"Samples: {count}/{target_samples}",
        (10,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Capture", frame)

    #comment stop if 100 samples collected
    if count >= target_samples:
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Finished collecting samples.")