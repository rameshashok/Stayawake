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


# ============================================================
# DRIVER DROWSINESS DETECTION
# STEP 2: DATA PREPROCESSING + MODEL TRAINING
# ============================================================

print("=" * 70)
print("STAYAWAKE - DRIVER DROWSINESS DETECTION")
print("STEP 2: DATA PREPROCESSING + MODEL TRAINING")
print("=" * 70)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FILE = os.path.join(
    "data",
    "driver_drowsiness.csv"
)

MODEL_FOLDER = "model"

MODEL_FILE = os.path.join(
    MODEL_FOLDER,
    "drowsiness_model.keras"
)

SCALER_FILE = os.path.join(
    MODEL_FOLDER,
    "scaler.pkl"
)

RANDOM_STATE = 42


# ============================================================
# 2. CHECK DATASET
# ============================================================

print("\n[1/10] Checking dataset...")

if not os.path.exists(DATA_FILE):

    print("\nERROR: Dataset not found!")
    print(f"Expected file: {DATA_FILE}")
    print("\nPlease run data_generation.py first.")

    exit()


print("Dataset found successfully.")
print(f"Location: {DATA_FILE}")


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n[2/10] Loading dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully.")

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 4. DISPLAY DATASET INFORMATION
# ============================================================

print("\nDataset columns:")

for column in df.columns:

    print(f"  - {column}")


print("\nFirst 5 records:")

print(
    df.head().to_string(
        index=False
    )
)


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n[3/10] Checking missing values...")

missing_values = df.isnull().sum()

if missing_values.sum() == 0:

    print("No missing values found.")

else:

    print("Missing values detected:")

    print(missing_values)

    print("\nPlease fix missing values before training.")

    exit()


# ============================================================
# 6. DEFINE FEATURES AND TARGET
# ============================================================

print("\n[4/10] Preparing features and target...")


FEATURE_COLUMNS = [
    "blink_rate",
    "yawning_rate",
    "steering_variation",
    "reaction_time",
    "driving_duration"
]


TARGET_COLUMN = "state"


X = df[FEATURE_COLUMNS]

y = df[TARGET_COLUMN]


print("\nInput features:")

for feature in FEATURE_COLUMNS:

    print(f"  - {feature}")


print(f"\nTarget variable: {TARGET_COLUMN}")


# ============================================================
# 7. DISPLAY CLASS DISTRIBUTION
# ============================================================

class_names = {
    0: "Alert",
    1: "Drowsy",
    2: "Sleepy"
}


print("\nClass distribution:")

class_counts = y.value_counts().sort_index()


for class_id, count in class_counts.items():

    class_name = class_names.get(
        class_id,
        f"Class {class_id}"
    )

    print(
        f"  {class_id} - {class_name:<8}: {count}"
    )


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

print("\n[5/10] Splitting dataset...")


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y
)


print("\nData split completed.")

print(
    f"Training samples : {len(X_train)}"
)

print(
    f"Testing samples  : {len(X_test)}"
)


# ============================================================
# 9. FEATURE SCALING
# ============================================================

print("\n[6/10] Scaling input features...")


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)


print("Feature scaling completed.")


# Display scaler information

print("\nFeature means after scaling:")

print(
    np.round(
        X_train_scaled.mean(axis=0),
        4
    )
)


print("\nFeature standard deviations after scaling:")

print(
    np.round(
        X_train_scaled.std(axis=0),
        4
    )
)


# ============================================================
# 10. CALCULATE CLASS WEIGHTS
# ============================================================

print("\n[7/10] Calculating class weights...")


classes = np.unique(y_train)


class_weights_array = compute_class_weight(

    class_weight="balanced",

    classes=classes,

    y=y_train
)


class_weights = dict(
    zip(
        classes,
        class_weights_array
    )
)


print("\nClass weights:")


for class_id, weight in class_weights.items():

    print(
        f"  {class_names[class_id]:<8}: {weight:.4f}"
    )


# ============================================================
# 11. BUILD NEURAL NETWORK
# ============================================================

print("\n[8/10] Building neural network...")


model = Sequential([

    # Input layer
    Input(
        shape=(X_train_scaled.shape[1],)
    ),

    # Hidden Layer 1
    Dense(
        64,
        activation="relu"
    ),

    # Prevent overfitting
    Dropout(0.30),

    # Hidden Layer 2
    Dense(
        32,
        activation="relu"
    ),

    # Prevent overfitting
    Dropout(0.20),

    # Hidden Layer 3
    Dense(
        16,
        activation="relu"
    ),

    # Output layer
    # 3 neurons = Alert, Drowsy, Sleepy
    Dense(
        3,
        activation="softmax"
    )
])


# ============================================================
# 12. DISPLAY MODEL ARCHITECTURE
# ============================================================

print("\nNeural network architecture:")

model.summary()


# ============================================================
# 13. COMPILE MODEL
# ============================================================

print("\nCompiling model...")


model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


print("Model compiled successfully.")


# ============================================================
# 14. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True
)


# ============================================================
# 15. TRAIN MODEL
# ============================================================

print("\n[9/10] Starting model training...")

print("-" * 70)


history = model.fit(

    X_train_scaled,

    y_train,

    validation_split=0.20,

    epochs=100,

    batch_size=32,

    class_weight=class_weights,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


print("-" * 70)

print("\nModel training completed!")


# ============================================================
# 16. TRAINING INFORMATION
# ============================================================

trained_epochs = len(
    history.history["loss"]
)


print(
    f"\nTraining stopped after "
    f"{trained_epochs} epoch(s)."
)


best_val_accuracy = max(
    history.history["val_accuracy"]
)


best_val_loss = min(
    history.history["val_loss"]
)


print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)


print(
    f"Best validation loss    : "
    f"{best_val_loss:.4f}"
)


# ============================================================
# 17. CREATE MODEL DIRECTORY
# ============================================================

print("\n[10/10] Saving model...")


os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ============================================================
# 18. SAVE TRAINED MODEL
# ============================================================

model.save(
    MODEL_FILE
)


print(
    f"Model saved successfully:"
)

print(
    f"  {MODEL_FILE}"
)


# ============================================================
# 19. SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    SCALER_FILE
)


print(
    f"\nScaler saved successfully:"
)

print(
    f"  {SCALER_FILE}"
)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nProject information:")

print(
    f"Total dataset samples : {len(df)}"
)

print(
    f"Training samples      : {len(X_train)}"
)

print(
    f"Testing samples       : {len(X_test)}"
)

print(
    f"Input features        : {len(FEATURE_COLUMNS)}"
)

print(
    "Output classes        : 3"
)

print(
    "Classes               : Alert, Drowsy, Sleepy"
)

print(
    "Algorithm             : Neural Network / MLP"
)

print(
    "Framework              : TensorFlow + Keras"
)

print(
    f"Best validation acc.  : {best_val_accuracy:.4f}"
)


print("\nGenerated files:")

print(
    f"  [OK] {MODEL_FILE}"
)

print(
    f"  [OK] {SCALER_FILE}"
)


print("\nNext step:")

print(
    "Evaluate the trained model using "
    "accuracy, precision, recall, F1-score "
    "and confusion matrix."
)


print("\n" + "=" * 70)