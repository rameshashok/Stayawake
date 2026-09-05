import os
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "rameshashok/stayawake-model"
MODEL_DIR = "model"
MODEL_FILES = ["drowsiness_model.keras", "scaler.pkl"]


def ensure_model():
    """Download model files from Hugging Face if not present locally."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    for filename in MODEL_FILES:
        dest = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(dest):
            print(f"Downloading {filename} from Hugging Face...")
            hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=MODEL_DIR)
            print(f"Saved: {dest}")


def upload_model():
    """Upload model files to Hugging Face Hub."""
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    for filename in MODEL_FILES:
        path = os.path.join(MODEL_DIR, filename)
        api.upload_file(path_or_fileobj=path, path_in_repo=filename, repo_id=REPO_ID)
        print(f"Uploaded: {filename}")
    print(f"\nModel hosted at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    upload_model()
