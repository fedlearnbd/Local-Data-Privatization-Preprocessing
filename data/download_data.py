"""
download_data.py
Script to automatically download MedMNIST datasets from Google Drive.
"""

import gdown
import os

# Mapping datasets -> Google Drive file IDs
DATASETS = {
    "bloodmnist": "1XMw4kR_8oEQ5z7yuxkK-SMAIoJY812u3",
    "organamnist": "1aCbYfGVMLp9DK3_8mCO94PB3eoxvTpxc",
    "pathmnist": "1hkznwSVSdOaY53SryQbp5qtt5JXAOOfR"
}

# Create data directory if not exists
os.makedirs("data", exist_ok=True)

def download_dataset(name, file_id):
    output_path = f"data/{name}.npz"
    if os.path.exists(output_path):
        print(f"{name}.npz already exists, skipping download.")
        return
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"⬇Downloading {name} dataset...")
    gdown.download(url, output_path, quiet=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    for dataset, file_id in DATASETS.items():
        download_dataset(dataset, file_id)
    print("\n All datasets downloaded successfully!")
