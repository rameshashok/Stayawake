# StayAwake — Driver Drowsiness Detection

A machine learning system that classifies driver alertness into three states: **Alert**, **Drowsy**, and **Sleepy** using a neural network trained on behavioral sensor data.

## Project Structure

```
Stayawake/
├── src/
│   ├── data_generation.py  # Generate synthetic training dataset
│   ├── train.py            # Train and save the model
│   ├── evaluate.py         # Evaluate model performance
│   └── model_hub.py        # Upload / download model from Hugging Face
├── data/
│   └── driver_drowsiness.csv
├── model/
│   └── confusion_matrix.png
├── requirements.txt
└── README.md
```

> Model files (`drowsiness_model.keras`, `scaler.pkl`) are hosted on [Hugging Face](https://huggingface.co/rameshashok/stayawake-model) and downloaded automatically at runtime.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Generate dataset**
```bash
python src/data_generation.py
```

**2. Train model**
```bash
python src/train.py
```

**3. Evaluate model**
```bash
python src/evaluate.py
```

**4. Upload model to Hugging Face** *(run once after training)*
```bash
huggingface-cli login
python src/model_hub.py
```

## Model

| Feature             | Description                        |
|---------------------|------------------------------------|
| `blink_rate`        | Eye blinks per minute              |
| `yawning_rate`      | Yawns per minute                   |
| `steering_variation`| Steering wheel movement variance   |
| `reaction_time`     | Driver reaction time (seconds)     |
| `driving_duration`  | Continuous driving time (minutes)  |

**Architecture:** MLP (64 → 32 → 16 → 3) with Dropout regularization  
**Framework:** TensorFlow / Keras  
**Classes:** Alert (0), Drowsy (1), Sleepy (2)
