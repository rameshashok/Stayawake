import numpy as np
import pandas as pd
import os


# ============================================================
# DRIVER DROWSINESS DETECTION
# STEP 1: SYNTHETIC DATASET GENERATION
# ============================================================

# Set random seed so that the same dataset can be reproduced
np.random.seed(42)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLES_PER_CLASS = 1500

OUTPUT_FOLDER = "data"

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "driver_drowsiness.csv"
)


# ============================================================
# FUNCTION: GENERATE ALERT DRIVER DATA
# ============================================================

def generate_alert_data(samples):

    data = []

    for _ in range(samples):

        # Blink rate: normal/healthy blinking
        blink_rate = np.random.normal(
            loc=20,
            scale=4
        )

        # Very few yawns
        yawning_rate = np.random.normal(
            loc=1,
            scale=0.8
        )

        # Low steering variation
        steering_variation = np.random.normal(
            loc=5,
            scale=2
        )

        # Fast reaction time
        reaction_time = np.random.normal(
            loc=0.6,
            scale=0.15
        )

        # Relatively shorter driving duration
        driving_duration = np.random.uniform(
            10,
            120
        )

        data.append([
            blink_rate,
            yawning_rate,
            steering_variation,
            reaction_time,
            driving_duration,
            0
        ])

    return data


# ============================================================
# FUNCTION: GENERATE DROWSY DRIVER DATA
# ============================================================

def generate_drowsy_data(samples):

    data = []

    for _ in range(samples):

        # Reduced blinking
        blink_rate = np.random.normal(
            loc=12,
            scale=3
        )

        # Increased yawning
        yawning_rate = np.random.normal(
            loc=4,
            scale=1.2
        )

        # Increased steering variation
        steering_variation = np.random.normal(
            loc=10,
            scale=3
        )

        # Slower reaction time
        reaction_time = np.random.normal(
            loc=1.2,
            scale=0.3
        )

        # Longer driving duration
        driving_duration = np.random.uniform(
            90,
            220
        )

        data.append([
            blink_rate,
            yawning_rate,
            steering_variation,
            reaction_time,
            driving_duration,
            1
        ])

    return data


# ============================================================
# FUNCTION: GENERATE SLEEPY DRIVER DATA
# ============================================================

def generate_sleepy_data(samples):

    data = []

    for _ in range(samples):

        # Very low blink rate
        blink_rate = np.random.normal(
            loc=7,
            scale=2
        )

        # Frequent yawning
        yawning_rate = np.random.normal(
            loc=7,
            scale=1.5
        )

        # High steering variation
        steering_variation = np.random.normal(
            loc=16,
            scale=3
        )

        # Slow reaction time
        reaction_time = np.random.normal(
            loc=1.9,
            scale=0.3
        )

        # Long continuous driving duration
        driving_duration = np.random.uniform(
            180,
            300
        )

        data.append([
            blink_rate,
            yawning_rate,
            steering_variation,
            reaction_time,
            driving_duration,
            2
        ])

    return data


# ============================================================
# GENERATE DATA
# ============================================================

print("=" * 60)
print("DRIVER DROWSINESS DETECTION")
print("Synthetic Dataset Generation")
print("=" * 60)


print("\nGenerating Alert driver data...")

alert_data = generate_alert_data(
    SAMPLES_PER_CLASS
)


print("Generating Drowsy driver data...")

drowsy_data = generate_drowsy_data(
    SAMPLES_PER_CLASS
)


print("Generating Sleepy driver data...")

sleepy_data = generate_sleepy_data(
    SAMPLES_PER_CLASS
)


# Combine all data
all_data = (
    alert_data
    + drowsy_data
    + sleepy_data
)


# ============================================================
# CREATE DATAFRAME
# ============================================================

columns = [
    "blink_rate",
    "yawning_rate",
    "steering_variation",
    "reaction_time",
    "driving_duration",
    "state"
]


df = pd.DataFrame(
    all_data,
    columns=columns
)


# ============================================================
# LIMIT VALUES TO REALISTIC RANGES
# ============================================================

df["blink_rate"] = df[
    "blink_rate"
].clip(
    lower=1,
    upper=40
)


df["yawning_rate"] = df[
    "yawning_rate"
].clip(
    lower=0,
    upper=12
)


df["steering_variation"] = df[
    "steering_variation"
].clip(
    lower=0,
    upper=25
)


df["reaction_time"] = df[
    "reaction_time"
].clip(
    lower=0.2,
    upper=3.0
)


df["driving_duration"] = df[
    "driving_duration"
].clip(
    lower=1,
    upper=300
)


# ============================================================
# SHUFFLE DATASET
# ============================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DATASET GENERATED SUCCESSFULLY")
print("=" * 60)


print(
    f"\nTotal number of samples: {len(df)}"
)


print(
    f"Number of features: {len(columns) - 1}"
)


print(
    f"Output file: {OUTPUT_FILE}"
)


# ============================================================
# CLASS INFORMATION
# ============================================================

class_names = {
    0: "Alert",
    1: "Drowsy",
    2: "Sleepy"
}


print("\nClass Distribution:")
print("-" * 30)


class_counts = df["state"].value_counts().sort_index()


for state, count in class_counts.items():

    print(
        f"{class_names[state]:<10}: {count}"
    )


# ============================================================
# DISPLAY FIRST 10 RECORDS
# ============================================================

print("\nFirst 10 records:")
print("-" * 60)

print(
    df.head(10).to_string(
        index=False
    )
)


# ============================================================
# DISPLAY STATISTICS
# ============================================================

print("\nDataset Statistics:")
print("-" * 60)

print(
    df.describe().round(2)
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("STEP 1 COMPLETED")
print("=" * 60)

print(
    "\nDataset is ready for the next ML step."
)

print(
    f"Saved at: {OUTPUT_FILE}"
)