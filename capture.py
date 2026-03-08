import cv2
import os
import csv
import mediapipe as mp

# ask user for gesture name
gesture_name = input("Enter gesture name: ").strip().replace(" ","_")

# create dataset folder if it doesn't exist
os.makedirs("dataset", exist_ok=True)

# dataset file
file_path = "dataset/landmarks.csv"

# initialize mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)

# utility for drawing landmarks
mp_draw = mp.solutions.drawing_utils

# open webcam
cap = cv2.VideoCapture(0)

# capture counters
count = 0
max_samples = 100

# capture mode flag
capture_mode = False

print("Press C to start automatic capture")

while True:

    # read camera frame
    ret, frame = cap.read()
    if not ret:
        break

    # convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect hand
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # draw landmarks on screen
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # automatically capture when mode is active
            if capture_mode and count < max_samples:

                landmark_row = []

                # extract x,y,z coordinates of all 21 landmarks
                for lm in hand_landmarks.landmark:
                    landmark_row.append(lm.x)
                    landmark_row.append(lm.y)
                    landmark_row.append(lm.z)

                # append gesture label
                landmark_row.append(gesture_name)

                # save to csv
                with open(file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(landmark_row)

                count += 1

                print("Captured:", count)

                # small delay so frames are slightly different
                cv2.waitKey(80)

                # stop when 100 samples reached
                if count >= max_samples:
                    print("Captured 100 samples successfully")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

    # show capture count
    cv2.putText(
        frame,
        f"Samples: {count}/100",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    # show camera window
    cv2.imshow("Capture Landmarks", frame)

    key = cv2.waitKey(1) & 0xFF

    # press C once to start auto capture
    if key == ord('c'):
        capture_mode = True
        print("Auto capture started")

    # ESC exits
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()