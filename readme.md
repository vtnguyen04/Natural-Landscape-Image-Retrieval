# Natural Landscape Image Search (CBIR + TBIR) 🌄

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Replace with your chosen license badge -->

A web application demonstrating Content-Based Image Retrieval (CBIR) and Text-Based Image Retrieval (TBIR) for natural landscape images. Users can search for visually similar images by uploading a sample image or by describing the desired scene using text.

This project leverages the power of OpenAI's CLIP model for generating joint image and text embeddings, Faiss for efficient similarity search, and Flask for the web interface. Image data is sourced responsibly using the Pexels API.

---

## ✨ Features

*   **Image-based Search (CBIR):** Upload a landscape image to find visually similar images in the dataset.
*   **Text-based Search (TBIR):** Enter a text description (e.g., "snowy mountains at sunset", "tropical beach with palm trees") to find matching images.
*   **Unified Search Space:** Utilizes CLIP embeddings, allowing both image and text queries to be searched against the same image index.
*   **Efficient Similarity Search:** Employs Faiss for fast Approximate Nearest Neighbor (ANN) search on image features.
*   **Simple Web Interface:** Built with Flask for easy interaction.
*   **Automated Data Acquisition:** Includes scripts to download landscape images from Pexels API based on specified categories.

---

## 🚀 Demo
<video controls src="video_demo.mp4" title="Title"></video>
---

## 🛠️ Technology Stack

*   **Backend:** Python 3.9+
*   **Web Framework:** Flask
*   **Deep Learning / Embeddings:**
    *   OpenAI CLIP (via `openai/clip` package or Hugging Face `transformers`)
    *   PyTorch (as backend for CLIP/Transformers)
*   **Similarity Search:** Faiss (Facebook AI Similarity Search)
*   **Data Handling:** NumPy, H5py, Pillow (PIL Fork)
*   **API Interaction:** Requests
*   **Environment Management:** python-dotenv
*   **Containerization (Optional):** Docker

---

## 📂 Project Structure
``` 
landscape_search/
├── app.py # Main Flask application file
├── data_crawler.py # Script to fetch data from Pexels API
├── feature_extractor.py # Script to extract CLIP features
├── index_builder.py # Script to build the Faiss index
├── static/ # Folder for static web files
│ ├── css/
│ │ └── style.css # Basic CSS styling
│ ├── uploads/ # Temporary storage for user uploads (created automatically)
│ └── dataset_images/ # IMPORTANT: Copied/Symlinked dataset images for web display
├── templates/ # Folder for HTML templates
│ └── index.html # Main HTML page
├── data/ # Data storage (ignored by git)
│ ├── raw_images/ # Raw images downloaded by the crawler
│ └── processed/ # Processed features, paths, and index files
├── .env # Environment variables (contains API Key - DO NOT COMMIT)
├── requirements.txt # Python dependencies
└── README.md # This file
└── .gitignore # Git ignore configuration
```

---


---

## ⚙️ Setup and Installation

1.  **Prerequisites:**
    *   Python 3.9 or higher
    *   `pip` package installer
    *   Git
    *   (Optional) Docker and Docker Compose

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/vtnguyen04/Natural-Landscape-Image-Retrieval.git # Replace with your repo URL
    cd Natural-Landscape-Image-Retrieval
    ```


3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Pexels API Key:**
    *   Sign up for a free API key at [Pexels API](https://www.pexels.com/api/).
    *   Create a file named `.env` in the project root directory.
    *   Add your API key to the `.env` file:
        ```dotenv
        PEXELS_API_KEY=YOUR_ACTUAL_PEXELS_API_KEY_HERE
        ```
    *   **Important:** Ensure `.env` is listed in your `.gitignore` file to avoid committing your key.

5.  **Create Necessary Directories:**
    ```bash
    mkdir -p data/raw_images data/processed static/uploads static/dataset_images
    ```
---

## 💾 Data Pipeline (Essential Steps)

Execute these scripts **in order** to prepare the data for the application:

1.  **Run the Data Crawler:**
    *   This script uses your Pexels API key to download landscape images based on the categories defined within it. It saves images to `data/raw_images/`.
    *   This step can take a significant amount of time depending on the number of images requested (`IMAGES_PER_CATEGORY` in the script).
    ```bash
    python data_crawler.py
    ```

2.  **IMPORTANT - Prepare Static Dataset Images:**
    *   The web application needs access to the downloaded images to display search results. Copy or create symbolic links of **all** images from `data/raw_images/` (maintaining the subdirectory structure) into the `static/dataset_images/` folder.
    *   **Example (Linux/macOS):**
        ```bash
        cp -r data/raw_images/* static/dataset_images/
        ```
    *   **Example (Windows):** Use File Explorer to copy the contents.
    *   *Failure to do this step will result in broken images in the web interface.*

3.  **Run the Feature Extractor:**
    *   This script loads the CLIP model and processes all images in `data/raw_images/`. It extracts image embeddings and saves them along with their relative paths into an HDF5 file (`data/processed/clip_features.h5`).
    *   This is computationally intensive and may take a long time, especially without a GPU.
    ```bash
    python feature_extractor.py
    ```

4.  **Run the Index Builder:**
    *   This script loads the extracted features and builds a Faiss index for efficient similarity searching. The index is saved to `data/processed/clip_index.faiss`.
    ```bash
    python index_builder.py
    ```

---

## ▶️ Running the Application

Once the data pipeline is complete:

1.  **Start the Flask Development Server:**
    ```bash
    python app.py
    ```

2.  **Access the Web Interface:**
    *   Open your web browser and navigate to `http://127.0.0.1:5000` (or the address provided by Flask).

---

## 🖱️ Usage

1.  Open the web application in your browser.
2.  **To search by image:** Click "Choose File", select a landscape image from your computer, and click "Search".
3.  **To search by text:** Enter a description in the text input field (e.g., "forest path in autumn") and click "Search".
4.  The results (similar images from the dataset) will be displayed below the search form.

---

## 🌱 Potential Improvements / Future Work

*   Implement different Faiss index types (e.g., `IndexIVFPQ`) for better scalability/performance trade-offs.
*   Experiment with different CLIP models (e.g., ViT-L/14).
*   Add pagination for search results.
*   Improve UI/UX (e.g., loading indicators, better result display).
*   Implement fine-tuning of the CLIP model on the landscape dataset (advanced).
*   Add metadata filtering (requires collecting metadata during crawling).
*   Develop evaluation metrics to quantitatively assess search performance.
*   Deploy to a cloud platform (Heroku, AWS, Google Cloud, Azure).
*   Integrate other data sources or APIs.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

---

## 📜 License
This project is maintained under the MIT License, allowing for free use and distribution with proper attribution.
---

## 🙏 Acknowledgements

*   **Pexels:** For providing the high-quality images via their API. [Pexels API](https://www.pexels.com/api/)
*   **OpenAI:** For the CLIP model. [CLIP GitHub](https://github.com/openai/CLIP)
*   **Facebook AI Research (Meta AI):** For the Faiss library. [Faiss GitHub](https://github.com/facebookresearch/faiss)
*   **Flask Team:** For the Flask web framework. [Flask Website](https://flask.palletsprojects.com/)
*   Hugging Face Transformers library (if used).