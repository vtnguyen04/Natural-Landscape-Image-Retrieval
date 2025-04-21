import os
from pathlib import Path
import numpy as np
import faiss
from PIL import Image
import torch
import clip 
from flask import Flask, request, render_template, url_for, send_from_directory
from werkzeug.utils import secure_filename
import h5py 

# --- Configuration ---
UPLOAD_FOLDER = Path('static/uploads')
# Folder chứa dataset ảnh mà web sẽ truy cập 
DATASET_STATIC_FOLDER = 'dataset_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# --- Model & Index Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/32"
PROCESSED_DATA_DIR = Path("data/processed")
FEATURES_FILE = PROCESSED_DATA_DIR / "clip_features.h5"
INDEX_FILE = PROCESSED_DATA_DIR / "clip_index.faiss"

# --- Global Variables  ---
model = None
preprocess = None
faiss_index = None
image_paths = None # List of relative paths (strings)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max upload size

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_resources():
    """Loads CLIP model, Faiss index, and image paths."""
    global model, preprocess, faiss_index, image_paths
    if model is None:
        print(f"Loading CLIP model ({MODEL_NAME}) on {DEVICE}...")
        try:
            model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
            model.eval()
            print("CLIP model loaded.")
        except Exception as e:
            print(f"Fatal Error: Could not load CLIP model: {e}")
            exit()

    if faiss_index is None:
        print(f"Loading Faiss index from {INDEX_FILE}...")
        try:
            faiss_index = faiss.read_index(str(INDEX_FILE))
            print(f"Faiss index loaded. Contains {faiss_index.ntotal} vectors.")
        except Exception as e:
            print(f"Fatal Error: Could not load Faiss index: {e}")
            exit()

    if image_paths is None:
        print(f"Loading image paths from {FEATURES_FILE}...")
        try:
             # Load paths từ HDF5
            with h5py.File(FEATURES_FILE, 'r') as hf:
                paths_data = hf['paths'][:]
                if paths_data.dtype.kind in ('S', 'O'): # Check if bytes or object (might contain bytes)
                    image_paths = [p.decode('utf-8') for p in paths_data]
                else:
                    image_paths = list(paths_data) # Assume it's already string type compatible

  
            print(f"Loaded {len(image_paths)} image paths.")
            if faiss_index and len(image_paths) != faiss_index.ntotal:
                 print(f"Warning: Number of paths ({len(image_paths)}) does not match number of vectors in index ({faiss_index.ntotal})!")

        except FileNotFoundError:
             print(f"Fatal Error: Features/Paths file not found at {FEATURES_FILE}.")
             exit()
        except Exception as e:
            print(f"Fatal Error: Could not load image paths: {e}")
            exit()

def extract_image_features(image_path):
    """Extracts CLIP features for a single image file."""
    if model is None or preprocess is None:
        load_resources() # Ensure resources are loaded
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy() # Shape (1, embedding_dim)
    except Exception as e:
        print(f"Error extracting features from image {image_path}: {e}")
        return None

def extract_text_features(text):
    """Extracts CLIP features for a text query."""
    if model is None:
        load_resources() # Ensure resources are loaded
    try:
        # Tokenize text
        text_input = clip.tokenize([text]).to(DEVICE)
        with torch.no_grad():
            text_features = model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy() # Shape (1, embedding_dim)
    except Exception as e:
        print(f"Error extracting features from text '{text}': {e}")
        return None

def search_index(query_vector, k=12):
    """Searches the Faiss index for the k nearest neighbors."""
    if faiss_index is None or image_paths is None:
        load_resources() # Ensure resources are loaded
    if query_vector is None:
        return []

    try:
        query_vector = query_vector.astype(np.float32)

        distances, indices = faiss_index.search(query_vector, k)

        results_paths = [image_paths[i] for i in indices[0] if i < len(image_paths)] # Lấy chỉ số từ indices[0]

        # Chuyển đổi đường dẫn hệ thống thành URL web
        # Đường dẫn trong image_paths là tương đối với RAW_IMAGE_DIR (ví dụ: mountains/term/pexels_123.jpg)
        # Cần chuyển thành /static/dataset_images/mountains/term/pexels_123.jpg
        results_urls = [url_for('static', filename=f'{DATASET_STATIC_FOLDER}/{p}') for p in results_paths]

        # Có thể trả về cả distances
        # results_distances = distances[0].tolist()
        return results_urls

    except Exception as e:
        print(f"Error during Faiss search: {e}")
        return []

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    # Chỉ render trang chính
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    load_resources() # Đảm bảo mọi thứ đã được load

    query_image = request.files.get('query_image')
    query_text = request.form.get('query_text', '').strip()
    num_results = int(request.form.get('num_results', 12)) # Lấy số lượng kết quả mong muốn
    k = max(1, min(num_results, 50)) # Giới hạn k hợp lý

    results = []
    query_image_url = None
    search_performed = False # Flag để biết đã thực hiện tìm kiếm chưa

    # Ưu tiên tìm kiếm bằng ảnh nếu có
    if query_image and query_image.filename != '' and allowed_file(query_image.filename):
        search_performed = True
        try:
            filename = secure_filename(query_image.filename)
            # Lưu file tạm thời vào uploads
            upload_path = UPLOAD_FOLDER / filename
            UPLOAD_FOLDER.mkdir(exist_ok=True) # Đảm bảo thư mục tồn tại
            query_image.save(upload_path)
            query_image_url = url_for('static', filename=f'uploads/{filename}') # URL để hiển thị ảnh query

            print(f"Processing uploaded image: {upload_path}")
            query_vector = extract_image_features(upload_path)
            if query_vector is not None:
                print("Searching by image features...")
                results = search_index(query_vector, k)
            else:
                 print("Could not extract features from uploaded image.")

        except Exception as e:
            print(f"Error processing uploaded image: {e}")

    # Nếu không có ảnh hoặc tìm bằng ảnh thất bại, và có text query
    elif query_text:
        search_performed = True
        print(f"Processing text query: '{query_text}'")
        query_vector = extract_text_features(query_text)
        if query_vector is not None:
            print("Searching by text features...")
            results = search_index(query_vector, k)
        else:
            print("Could not extract features from text query.")

    else:
        # Không có ảnh hợp lệ và không có text query
        print("No valid query (image or text) provided.")

    print(f"Found {len(results)} results.")

    # Render lại trang index với kết quả
    return render_template('index.html',
                           results = results,
                           query_image_url = query_image_url,
                           query_text = query_text,
                           search_performed = search_performed) # Truyền flag để biết nên hiển thị "No results" hay không

# --- Main Execution Guard ---
if __name__ == '__main__':
    load_resources() # Load trước khi chạy app
    # Chạy Flask development server
    # Trong production, dùng Gunicorn hoặc Waitress
    app.run(debug = True, host = '0.0.0.0', port = 5000)