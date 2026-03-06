import cv2
import os

# Ask user for gesture name
gesture_name = input("Enter gesture name: ").strip().replace(" ", "_")

dataset_path = os.path.join("dataset", gesture_name)

# Create folder if it doesn't exist
os.makedirs(dataset_path, exist_ok=True)

print(f"\nSaving images to: {dataset_path}")
print("Press 'c' to capture image")
print("Press 'q' to quit\n")

cap = cv2.VideoCapture(0)

count = 0
max_images = 70

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera failed. Hardware likes disappointing people.")
        break

    frame_display = frame.copy()

    cv2.putText(
        frame_display,
        f"Captured: {count}/{max_images}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Capture Gesture", frame_display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and count < max_images:

        img = cv2.resize(frame, (224, 224))

        file_path = os.path.join(dataset_path, f"{count}.jpg")
        cv2.imwrite(file_path, img)

        count += 1
        print(f"Captured {count}")

        if count == max_images:
            print("Captured 70 images. Your dataset thanks you.")
            break

    elif key == ord('q'):
        print("Capture stopped early.")
        break

cap.release()
cv2.destroyAllWindows()