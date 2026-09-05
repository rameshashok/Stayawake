import os
import numpy as np
import pandas as pd

np.random.seed(42)

SAMPLES_PER_CLASS = 1500
OUTPUT_FILE = os.path.join("data", "driver_drowsiness.csv")


def generate_alert_data(samples):
    data = []
    for _ in range(samples):
        data.append([
            np.random.normal(20, 4),
            np.random.normal(1, 0.8),
            np.random.normal(5, 2),
            np.random.normal(0.6, 0.15),
            np.random.uniform(10, 120),
            0
        ])
    return data


def generate_drowsy_data(samples):
    data = []
    for _ in range(samples):
        data.append([
            np.random.normal(12, 3),
            np.random.normal(4, 1.2),
            np.random.normal(10, 3),
            np.random.normal(1.2, 0.3),
            np.random.uniform(90, 220),
            1
        ])
    return data


def generate_sleepy_data(samples):
    data = []
    for _ in range(samples):
        data.append([
            np.random.normal(7, 2),
            np.random.normal(7, 1.5),
            np.random.normal(16, 3),
            np.random.normal(1.9, 0.3),
            np.random.uniform(180, 300),
            2
        ])
    return data


def main():
    columns = ["blink_rate", "yawning_rate", "steering_variation", "reaction_time", "driving_duration", "state"]

    all_data = (
        generate_alert_data(SAMPLES_PER_CLASS)
        + generate_drowsy_data(SAMPLES_PER_CLASS)
        + generate_sleepy_data(SAMPLES_PER_CLASS)
    )

    df = pd.DataFrame(all_data, columns=columns)

    df["blink_rate"] = df["blink_rate"].clip(1, 40)
    df["yawning_rate"] = df["yawning_rate"].clip(0, 12)
    df["steering_variation"] = df["steering_variation"].clip(0, 25)
    df["reaction_time"] = df["reaction_time"].clip(0.2, 3.0)
    df["driving_duration"] = df["driving_duration"].clip(1, 300)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Dataset saved to {OUTPUT_FILE} ({len(df)} samples)")


if __name__ == "__main__":
    main()
