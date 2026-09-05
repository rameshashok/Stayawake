from huggingface_hub import HfApi

REPO_ID = "rameshashok/stayawake-model"

api = HfApi()

api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

api.upload_file(path_or_fileobj="model/drowsiness_model.keras", path_in_repo="drowsiness_model.keras", repo_id=REPO_ID)
api.upload_file(path_or_fileobj="model/scaler.pkl", path_in_repo="scaler.pkl", repo_id=REPO_ID)

print(f"Model uploaded to https://huggingface.co/{REPO_ID}")
