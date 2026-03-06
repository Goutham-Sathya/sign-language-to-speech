"""
main.py

Real-time gesture recognition using RAW IMAGE training.
The frame is resized and flattened before prediction
because the model was trained on flattened images.
"""

import cv2
import pickle
import time

from modules.tts_engine import speak


# -------- Model path --------
MODEL_PATH = "models/gesture_model.pkl"


# -------- Load trained model --------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully")


# -------- Camera setup --------
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# -------- Control speech repetition --------
last_spoken = ""
last_time = 0
speech_delay = 2


while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera failure. Hardware occasionally enjoys rebellion.")
        break


    # Resize image to same size used during training
    img = cv2.resize(frame, (224, 224))


    # Flatten image to 1D vector
    img = img.flatten()


    # Predict gesture
    prediction = model.predict([img])[0]


    # Display prediction
    cv2.putText(
        frame,
        prediction,
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )


    # Speech control (avoid repeating every frame)
    current_time = time.time()

    if prediction != last_spoken and current_time - last_time > speech_delay:

        print("Detected:", prediction)

        speak(prediction)

        last_spoken = prediction
        last_time = current_time


    cv2.imshow("Gesture Recognition", frame)


    # ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()