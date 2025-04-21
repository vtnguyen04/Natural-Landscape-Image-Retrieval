import torch
import clip 
from PIL import Image
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import h5py

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# choose model CLIP. "ViT-B/32" for balancing. 
MODEL_NAME = "ViT-B/32"
RAW_IMAGE_DIR = Path("data/raw_images")
PROCESSED_DATA_DIR = Path("data/processed")
FEATURES_FILE = PROCESSED_DATA_DIR / "clip_features.h5" 

print(f"Using device: {DEVICE}")
print(f"Loading CLIP model: {MODEL_NAME}...")

try:
    model, preprocess = clip.load(MODEL_NAME, device = DEVICE)
    model.eval() 
    print("CLIP model loaded successfully.")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    print("Ensure you have run 'pip install git+https://github.com/openai/CLIP.git'")
    exit()

# --- Helper Function ---
def extract_features(image_path):
    """Extracts CLIP features for a single image."""
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(DEVICE)
     
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        # L2 normalization (often for cosine similarity)
        image_features /= image_features.norm(dim = -1, keepdim = True)
        return image_features.cpu().numpy().squeeze()
    except FileNotFoundError:
        print(f"Warning: Image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"Warning: Could not process image {image_path}: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    PROCESSED_DATA_DIR.mkdir(parents = True, exist_ok = True)

    image_paths = []
    print(f"Scanning for images in: {RAW_IMAGE_DIR}")
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_paths.extend(list(RAW_IMAGE_DIR.rglob(ext)))

    if not image_paths:
        print(f"Error: No images found in {RAW_IMAGE_DIR}. Did you run the data_crawler.py script?")
        exit()

    print(f"Found {len(image_paths)} images to process.")

    all_features = []
    valid_paths = []

    # feature extraction process
    for img_path in tqdm(image_paths, desc = "Extracting Features"):
        features = extract_features(img_path)
        if features is not None:
            all_features.append(features)
            relative_path = img_path.relative_to(RAW_IMAGE_DIR)
            valid_paths.append(relative_path.as_posix())

    if not all_features:
        print("Error: No features were extracted. Check image paths and model loading.")
        exit()

    features_array = np.array(all_features, dtype = np.float32)
    paths_array = np.array(valid_paths) 

    print(f"Successfully extracted features for {len(valid_paths)} images.")
    print(f"Feature vector shape: {features_array.shape}") 

    print(f"Saving features and paths to {FEATURES_FILE}...")
    try:
        with h5py.File(FEATURES_FILE, 'w') as hf:
            hf.create_dataset('features', data = features_array)
            string_dt = h5py.string_dtype(encoding = 'utf-8')
            hf.create_dataset('paths', data = paths_array.astype(string_dt))
        print("Data saved successfully using HDF5.")
    except Exception as e:
        print(f"Error saving data with HDF5: {e}")

