import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATASET_DIR = "dataset"

def load_dataset():

    X = []
    y = []

    print("Loading dataset...\n")

    #comment go through each gesture folder
    for gesture in os.listdir(DATASET_DIR):

        gesture_path = os.path.join(DATASET_DIR, gesture)

        if not os.path.isdir(gesture_path):
            continue

        print("Loading:", gesture)

        for file in os.listdir(gesture_path):

            if file.endswith(".npy"):

                file_path = os.path.join(gesture_path, file)

                data = np.load(file_path)

                X.append(data)
                y.append(gesture)

    return np.array(X), np.array(y)


def train_model(X, y):

    #comment split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    #comment model
    model = RandomForestClassifier(n_estimators=100)

    print("\nTraining model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print("Accuracy:", acc)

    return model


def main():

    X, y = load_dataset()

    print("\nTotal samples:", len(X))

    if len(X) == 0:
        print("Dataset is empty. Capture gestures first.")
        return

    model = train_model(X, y)

    joblib.dump(model, "gesture_model.pkl")

    print("Model saved as gesture_model.pkl")


if __name__ == "__main__":
    main()