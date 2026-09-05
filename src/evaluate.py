import os
import sys
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

sys.path.insert(0, os.path.dirname(__file__))
from model_hub import ensure_model

DATA_FILE = os.path.join("data", "driver_drowsiness.csv")
MODEL_FILE = os.path.join("model", "drowsiness_model.keras")
SCALER_FILE = os.path.join("model", "scaler.pkl")
REPORT_FILE = os.path.join("model", "report.html")

FEATURES = ["blink_rate", "yawning_rate", "steering_variation", "reaction_time", "driving_duration"]
CLASS_NAMES = ["Alert", "Drowsy", "Sleepy"]
COLORS = ["#2ecc71", "#e67e22", "#e74c3c"]


def build_report(cm, accuracy, precision, recall, f1, per_class):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Confusion Matrix",
            "Overall Metrics",
            "Per-Class Precision",
            "Per-Class Recall"
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "bar"}],
            [{"type": "bar"},     {"type": "bar"}]
        ]
    )

    # Confusion matrix
    fig.add_trace(go.Heatmap(
        z=cm, x=CLASS_NAMES, y=CLASS_NAMES,
        colorscale="Blues",
        text=cm, texttemplate="%{text}",
        showscale=False
    ), row=1, col=1)

    # Overall metrics bar
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    values = [accuracy, precision, recall, f1]
    fig.add_trace(go.Bar(
        x=metrics, y=[v * 100 for v in values],
        marker_color=["#3498db", "#9b59b6", "#1abc9c", "#e74c3c"],
        text=[f"{v*100:.2f}%" for v in values],
        textposition="outside"
    ), row=1, col=2)

    # Per-class precision
    fig.add_trace(go.Bar(
        x=CLASS_NAMES, y=[v * 100 for v in per_class["precision"]],
        marker_color=COLORS,
        text=[f"{v*100:.2f}%" for v in per_class["precision"]],
        textposition="outside"
    ), row=2, col=1)

    # Per-class recall
    fig.add_trace(go.Bar(
        x=CLASS_NAMES, y=[v * 100 for v in per_class["recall"]],
        marker_color=COLORS,
        text=[f"{v*100:.2f}%" for v in per_class["recall"]],
        textposition="outside"
    ), row=2, col=2)

    fig.update_yaxes(range=[0, 110], ticksuffix="%", row=1, col=2)
    fig.update_yaxes(range=[0, 110], ticksuffix="%", row=2, col=1)
    fig.update_yaxes(range=[0, 110], ticksuffix="%", row=2, col=2)

    fig.update_layout(
        title_text="StayAwake — Driver Drowsiness Detection Report",
        title_font_size=20,
        showlegend=False,
        height=750,
        template="plotly_dark"
    )

    fig.write_html(REPORT_FILE)
    print(f"Report saved: {REPORT_FILE}")


def main():
    ensure_model()

    df = pd.read_csv(DATA_FILE)
    X, y = df[FEATURES], df["state"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    scaler = joblib.load(SCALER_FILE)
    model = load_model(MODEL_FILE)

    y_pred = np.argmax(model.predict(scaler.transform(X_test), verbose=0), axis=1)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    per_class = {
        "precision": precision_score(y_test, y_pred, average=None, zero_division=0),
        "recall":    recall_score(y_test, y_pred, average=None, zero_division=0),
    }

    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    build_report(cm, accuracy, precision, recall, f1, per_class)


if __name__ == "__main__":
    main()
