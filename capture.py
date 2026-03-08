"""
capture.py

Captures hand gesture images for dataset creation.

Features:
- Detects hand using MediaPipe
- Crops only the hand region
- Resizes to 224x224
- Automatically captures 70 images after pressing 'c'
"""

import cv2
import os
import mediapipe as mp

# Ask user for gesture name
gesture_name = input("Enter gesture name: ").strip().replace(" ", "_")

dataset_path = os.path.join("dataset", gesture_name)

os.makedirs(dataset_path, exist_ok=True)

print(f"\nSaving images to: {dataset_path}")
print("Press 'c' to start automatic capture")
print("Press 'q' to quit\n")


# -------- MediaPipe Setup --------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils


# -------- Camera Setup --------
cap = cv2.VideoCapture(0)

count = 0
max_images = 100
capture_mode = False


while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera failed.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    h, w, _ = frame.shape

    cropped_hand = None


    # -------- Detect Hand --------
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)

            padding = 40

            xmin = max(0, xmin - padding)
            ymin = max(0, ymin - padding)
            xmax = min(w, xmax + padding)
            ymax = min(h, ymax + padding)

            cropped_hand = frame[ymin:ymax, xmin:xmax]


    # -------- Capture Images Automatically --------
    if capture_mode and cropped_hand is not None and count < max_images:

        img = cv2.resize(cropped_hand, (224, 224))

        file_path = os.path.join(dataset_path, f"{count}.jpg")

        cv2.imwrite(file_path, img)

        count += 1

        print(f"Captured {count}/{max_images}")

        cv2.waitKey(100)  # small delay between captures


    # -------- Display Info --------
    cv2.putText(
        frame,
        f"Captured: {count}/{max_images}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gesture Capture", frame)


    key = cv2.waitKey(1) & 0xFF


    # Start auto capture
    if key == ord('c'):
        capture_mode = True
        print("Starting automatic capture...")


    # Quit program
    if key == ord('q'):
        print("Capture stopped.")
        break


    if count >= max_images:
        print("Captured 70 images. Dataset ready.")
        break


cap.release()
cv2.destroyAllWindows()