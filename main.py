import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3

#comment load trained model
model = joblib.load("gesture_model.pkl")

#comment initialize TTS
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

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

previous_gesture = None
CONFIDENCE_THRESHOLD = 0.75

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_gesture = None
    confidence = 0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            #comment extract landmarks
            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            landmarks = np.array(landmarks).reshape(1, -1)

            #comment get prediction probabilities
            probs = model.predict_proba(landmarks)[0]
            index = np.argmax(probs)

            confidence = probs[index]
            gesture = model.classes_[index]

            if confidence > CONFIDENCE_THRESHOLD:
                current_gesture = gesture
            else:
                current_gesture = "Unknown"

            cv2.putText(
                frame,
                f"{current_gesture} ({confidence:.2f})",
                (10,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

    else:
        #comment reset when hand leaves frame
        previous_gesture = None

    #comment speak only when gesture changes
    if current_gesture != previous_gesture and current_gesture not in [None, "Unknown"]:

        print("Detected:", current_gesture)

        speak(current_gesture)

        previous_gesture = current_gesture

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()