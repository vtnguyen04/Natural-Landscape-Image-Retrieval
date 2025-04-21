import faiss
import numpy as np
from pathlib import Path
import h5py # Sử dụng h5py

# --- Configuration ---
PROCESSED_DATA_DIR = Path("data/processed")
FEATURES_FILE = PROCESSED_DATA_DIR / "clip_features.h5" 
INDEX_FILE = PROCESSED_DATA_DIR / "clip_index.faiss"

# --- Main Execution ---
if __name__ == "__main__":
    print("Loading features...")
    try:
        with h5py.File(FEATURES_FILE, 'r') as hf:
            features = hf['features'][:] # Load toàn bộ vào memory

        if features.size == 0:
             print("Error: Features file is empty or not loaded correctly.")
             exit()
        print(f"Features loaded. Shape: {features.shape}")

        if features.dtype != np.float32:
            print("Converting features to float32...")
            features = features.astype(np.float32)

        dimension = features.shape[1] # Số chiều của vector CLIP
        print(f"Feature dimension: {dimension}")

        # Xây dựng Faiss Index
        # IndexFlatIP: Inner Product - Tương đương Cosine Similarity khi vector đã chuẩn hóa L2
        # IndexFlatL2: Euclidean Distance
        # Chọn IndexFlatIP vì CLIP embeddings thường được chuẩn hóa L2
        print("Building Faiss index (IndexFlatIP)...")
        index = faiss.IndexFlatIP(dimension)

        # Thêm các vector vào index
        index.add(features)

        print(f"Index built successfully with {index.ntotal} vectors.")

        # Lưu index vào file
        print(f"Saving index to {INDEX_FILE}...")
        faiss.write_index(index, str(INDEX_FILE)) # faiss cần string path
        print("Index saved successfully.")

    except FileNotFoundError:
        print(f"Error: Features file not found at {FEATURES_FILE}. Run feature_extractor.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")