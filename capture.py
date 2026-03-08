import cv2
import mediapipe as mp
import numpy as np
import os

#comment ask user for gesture name
gesture_name = input("Enter gesture name: ").strip()

#comment dataset directory
DATASET_DIR = "dataset"

#comment create dataset directory if missing
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

#comment create gesture folder
gesture_dir = os.path.join(DATASET_DIR, gesture_name)

if not os.path.exists(gesture_dir):
    os.makedirs(gesture_dir)

print("Saving samples to:", gesture_dir)

#comment mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

#comment open webcam
cap = cv2.VideoCapture(0)

count = 0
TARGET = 100

#comment capture starts only after pressing C
capturing = False

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    #comment detect landmarks only if capturing started
    if results.multi_hand_landmarks and capturing:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            #comment extract 21 landmarks (x,y,z)
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
            print("Saved:", count)

    #comment show counter
    cv2.putText(frame, f"{count}/{TARGET}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    #comment show instruction if not capturing yet
    if not capturing:
        cv2.putText(frame, "Press C to start capture",
                    (10,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,255),
                    2)

    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    #comment start capturing when C is pressed
    if key == ord('c'):
        capturing = True
        print("Capture started")

    #comment exit with ESC
    if key == 27:
        break

    if count >= TARGET:
        break

cap.release()
cv2.destroyAllWindows()

print("Capture finished")