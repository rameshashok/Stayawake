import os
from huggingface_hub import hf_hub_download

REPO_ID = "rameshashok/stayawake-model"
MODEL_DIR = "model"


def ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    for filename in ["drowsiness_model.keras", "scaler.pkl"]:
        dest = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(dest):
            print(f"Downloading {filename} from Hugging Face...")
            path = hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=MODEL_DIR)
            print(f"Saved to {path}")


if __name__ == "__main__":
    ensure_model()
