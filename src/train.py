import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

DATA_FILE = os.path.join("data", "driver_drowsiness.csv")
MODEL_FILE = os.path.join("model", "drowsiness_model.keras")
SCALER_FILE = os.path.join("model", "scaler.pkl")
RANDOM_STATE = 42

FEATURES = ["blink_rate", "yawning_rate", "steering_variation", "reaction_time", "driving_duration"]
TARGET = "state"


def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}. Run src/data_generation.py first.")
    df = pd.read_csv(DATA_FILE)
    return df[FEATURES], df[TARGET]


def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dropout(0.30),
        Dense(32, activation="relu"),
        Dropout(0.20),
        Dense(16, activation="relu"),
        Dense(3, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    classes = np.unique(y_train)
    class_weights = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_train)))

    model = build_model(X_train_scaled.shape[1])

    model.fit(
        X_train_scaled, y_train,
        validation_split=0.20,
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
        verbose=1
    )

    os.makedirs("model", exist_ok=True)
    model.save(MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    print(f"\nModel saved: {MODEL_FILE}")
    print(f"Scaler saved: {SCALER_FILE}")


if __name__ == "__main__":
    main()
