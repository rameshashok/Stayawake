import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_score, recall_score, f1_score
)

sys.path.insert(0, os.path.dirname(__file__))
from model_hub import ensure_model

DATA_FILE = os.path.join("data", "driver_drowsiness.csv")
MODEL_FILE = os.path.join("model", "drowsiness_model.keras")
SCALER_FILE = os.path.join("model", "scaler.pkl")
CM_FILE = os.path.join("model", "confusion_matrix.png")

FEATURES = ["blink_rate", "yawning_rate", "steering_variation", "reaction_time", "driving_duration"]
CLASS_NAMES = ["Alert", "Drowsy", "Sleepy"]


def main():
    ensure_model()

    df = pd.read_csv(DATA_FILE)
    X, y = df[FEATURES], df["state"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    scaler = joblib.load(SCALER_FILE)
    model = load_model(MODEL_FILE)

    y_pred = np.argmax(model.predict(scaler.transform(X_test), verbose=0), axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\nAccuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")
    print(f"\n{classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0)}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot(ax=ax, values_format="d")
    plt.title("Driver Drowsiness Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CM_FILE, dpi=300)
    plt.close()
    print(f"Confusion matrix saved: {CM_FILE}")


if __name__ == "__main__":
    main()
