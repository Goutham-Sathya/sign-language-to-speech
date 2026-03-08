import cv2
import pickle
import json
import mediapipe as mp

from modules.tts_engine import TTSEngine


# -------- Load model --------
with open("models/gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded")


# -------- Load labels --------
with open("models/gestures.json", "r") as f:
    labels = json.load(f)


# -------- TTS --------
tts = TTSEngine()


# -------- MediaPipe --------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

mp_draw = mp.solutions.drawing_utils


# -------- Camera --------
cap = cv2.VideoCapture(0)


# -------- Gesture stability --------
gesture_buffer = []
buffer_size = 5

last_spoken = ""


while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    prediction = ""
    hand_detected = False


    if results.multi_hand_landmarks:

        hand_detected = True

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = frame.shape

            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)

            padding = 40

            xmin = max(0, xmin-padding)
            ymin = max(0, ymin-padding)
            xmax = min(w, xmax+padding)
            ymax = min(h, ymax+padding)

            hand_crop = frame[ymin:ymax, xmin:xmax]

            img = cv2.resize(hand_crop, (224,224))
            img = img.flatten()

            pred_index = model.predict([img])[0]

            prediction = labels[str(pred_index)]


    # -------- Stability Buffer --------
    if prediction != "":
        gesture_buffer.append(prediction)

        if len(gesture_buffer) > buffer_size:
            gesture_buffer.pop(0)

    else:
        gesture_buffer.clear()


    stable_prediction = ""

    if len(gesture_buffer) == buffer_size and len(set(gesture_buffer)) == 1:
        stable_prediction = gesture_buffer[0]


    # -------- Display --------
    if stable_prediction != "":
        cv2.putText(
            frame,
            stable_prediction,
            (20,60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0,255,0),
            3
        )


    # -------- Speak when gesture changes --------
    if stable_prediction != "" and stable_prediction != last_spoken:

        print("Detected:", stable_prediction)

        tts.Speak_text(stable_prediction)

        last_spoken = stable_prediction


    # Reset if hand disappears
    if not hand_detected:
        last_spoken = ""
        gesture_buffer.clear()


    cv2.imshow("Gesture Recognition", frame)


    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()