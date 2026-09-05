import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "data/driver_drowsiness.csv"
MODEL_PATH = "model/drowsiness_model.keras"
SCALER_PATH = "model/scaler.pkl"

CONFUSION_MATRIX_PATH = "model/confusion_matrix.png"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 60)
print("STEP 3 - MODEL EVALUATION")
print("=" * 60)

print("\n[1] Loading dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

data = pd.read_csv(DATA_PATH)

print(f"Dataset loaded successfully.")
print(f"Total samples: {len(data)}")


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

features = [
    "blink_rate",
    "yawning_rate",
    "steering_variation",
    "reaction_time",
    "driving_duration"
]

target = "state"

X = data[features]
y = data[target]


# ============================================================
# 4. RECREATE THE SAME TEST SPLIT
# ============================================================

print("\n[2] Creating test dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")


# ============================================================
# 5. LOAD SCALER
# ============================================================

print("\n[3] Loading scaler...")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

scaler = joblib.load(SCALER_PATH)

X_test_scaled = scaler.transform(X_test)

print("Scaler loaded successfully.")


# ============================================================
# 6. LOAD TRAINED MODEL
# ============================================================

print("\n[4] Loading trained model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# 7. MAKE PREDICTIONS
# ============================================================

print("\n[5] Generating predictions...")

probabilities = model.predict(X_test_scaled, verbose=0)

y_pred = np.argmax(probabilities, axis=1)

print("Predictions generated successfully.")


# ============================================================
# 8. CLASS NAMES
# ============================================================

class_names = [
    "Alert",
    "Drowsy",
    "Sleepy"
]


# ============================================================
# 9. ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("OVERALL PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Accuracy : {accuracy * 100:.2f}%")


# ============================================================
# 10. PRECISION, RECALL AND F1 SCORE
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print(report)


# ============================================================
# 12. PER-CLASS RECALL
# ============================================================

class_recalls = recall_score(
    y_test,
    y_pred,
    average=None,
    zero_division=0
)

print("=" * 60)
print("PER-CLASS RECALL")
print("=" * 60)

for class_name, class_recall in zip(class_names, class_recalls):
    print(f"{class_name:10s}: {class_recall:.4f} ({class_recall * 100:.2f}%)")


# ============================================================
# 13. CRITICAL STATE RECALL
# ============================================================
# Drowsy and Sleepy are considered critical states.

drowsy_recall = class_recalls[1]
sleepy_recall = class_recalls[2]

critical_recall = (drowsy_recall + sleepy_recall) / 2

print("\n" + "=" * 60)
print("CRITICAL STATE PERFORMANCE")
print("=" * 60)

print(f"Drowsy Recall : {drowsy_recall:.4f} ({drowsy_recall * 100:.2f}%)")
print(f"Sleepy Recall : {sleepy_recall:.4f} ({sleepy_recall * 100:.2f}%)")
print(
    f"Critical Recall: {critical_recall:.4f} "
    f"({critical_recall * 100:.2f}%)"
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

print("\n[6] Generating confusion matrix...")

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 15. SAVE CONFUSION MATRIX IMAGE
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    values_format="d"
)

plt.title("Driver Drowsiness Detection - Confusion Matrix")
plt.xlabel("Predicted State")
plt.ylabel("Actual State")
plt.tight_layout()

plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)

plt.close()

print(f"\nConfusion matrix saved to:")
print(CONFUSION_MATRIX_PATH)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(f"Test Samples       : {len(X_test)}")
print(f"Accuracy           : {accuracy * 100:.2f}%")
print(f"Weighted Precision : {precision * 100:.2f}%")
print(f"Weighted Recall    : {recall * 100:.2f}%")
print(f"Weighted F1 Score  : {f1 * 100:.2f}%")
print(f"Critical Recall    : {critical_recall * 100:.2f}%")

print("\nGenerated file:")
print(f"- {CONFUSION_MATRIX_PATH}")

print("\nStep 3 completed successfully!")