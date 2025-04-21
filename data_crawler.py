import requests
import os
import time
import random
import logging
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
import string
from urllib.parse import urlparse

# --- Configuration ---
logging.basicConfig(level = logging.INFO, 
                    format = '%(asctime)s - %(levelname)s - %(message)s')

load_dotenv() 

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
if not PEXELS_API_KEY:
    logging.error("Pexels API key not found in .env file. Please create a .env file with PEXELS_API_KEY=YOUR_KEY")
    exit()

PEXELS_API_URL = "https://api.pexels.com/v1/search"
DOWNLOAD_BASE_DIR = Path("data/raw_images")
IMAGES_PER_CATEGORY = 80 # number of images to download per category
PEXELS_PER_PAGE = 80 # Pexels return maximum 80 images per page
DOWNLOAD_TIMEOUT = 30 # Timeout for download requests

# --- landscape Categories ---
LANDSCAPE_CATEGORIES = {
    "mountains": ["mountains", "snowy mountains", "mountain range", "alps"],
    "beach": ["beach", "tropical beach", "sunset beach", "coastline"],
    "forest": ["forest", "deep forest", "rainforest", "woods"],
    "desert": ["desert", "sand dunes", "arid landscape"],
    "water": ["lake", "river", "waterfall", "ocean", "sea"],
    "other": ["valley", "canyon", "glacier", "volcano", "field", "meadow", "aurora borealis"]
}

HEADERS = {'Authorization': PEXELS_API_KEY}

# --- Helper Functions ---
def sanitize_filename(filename):
    """Removes invalid characters for filenames and replaces spaces."""
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    cleaned = ''.join(c for c in filename if c in valid_chars)
    cleaned = cleaned.replace(' ', '_').lower()
    return cleaned

def download_single_image(img_url, save_path):
    """Downloads and saves a single image."""
    try:
        if save_path.exists():
            logging.debug(f"Skipping existing file: {save_path}")
            return True

        time.sleep(random.uniform(0.1, 0.4))
        response = requests.get(img_url, stream = True, timeout = DOWNLOAD_TIMEOUT)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(1024 * 8):
                f.write(chunk)
        logging.debug(f"Successfully downloaded: {save_path}")
        return True
    except requests.exceptions.Timeout:
        logging.error(f"Timeout downloading {img_url}")
        return False
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading {img_url}: {e}")
        return False
    except IOError as e:
        logging.error(f"Error saving file {save_path}: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error for {img_url}: {e}")
        return False

def fetch_pexels_images(query, num_images):
    """Fetches image URLs from Pexels API for a given query."""
    image_urls = []
    page = 1
    fetched_count = 0
    max_pages = (num_images // PEXELS_PER_PAGE) + 2 # Lấy dư page phòng trường hợp thiếu ảnh

    logging.info(f"Fetching up to {num_images} images for query: '{query}'...")

    while fetched_count < num_images and page <= max_pages:
        params = {'query': query, 'per_page': PEXELS_PER_PAGE, 'page': page}
        try:
            time.sleep(random.uniform(0.8, 1.5))
            response = requests.get(PEXELS_API_URL, headers = HEADERS, params = params, timeout = 20)
            response.raise_for_status()

            data = response.json()
            photos = data.get('photos', [])
            if not photos:
                logging.warning(f"No more photos found for '{query}' on page {page}.")
                break 

            for photo in photos:
                if 'src' in photo and 'large' in photo['src']: 
                    image_urls.append({
                        'id': photo.get('id'),
                        'url': photo['src']['large'] # Có thể chọn original, large2x...
                    })
                    fetched_count += 1
                    if fetched_count >= num_images:
                        break 

            page += 1
            remaining = response.headers.get('X-Ratelimit-Remaining')
            logging.debug(f"Query: '{query}', Page: {page-1}, Fetched this page: {len(photos)}, Total URLs: {len(image_urls)}, RateLimit Remaining: {remaining}")
            if remaining and int(remaining) < 10:
                logging.warning("Approaching Pexels rate limit, pausing for 5s...")
                time.sleep(5)

        except requests.exceptions.HTTPError as e:
             logging.error(f"HTTP Error fetching page {page} for '{query}': {e.response.status_code} {e.response.reason}")
             if e.response.status_code == 429: # Rate limit
                 logging.warning("Rate limit hit. Pausing for 10s...")
                 time.sleep(10)
             else:
                 break # Lỗi khác thì dừng query này
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for '{query}', page {page}: {e}")
            time.sleep(5) # Chờ rồi thử lại page tiếp theo có thể không hiệu quả, nên dừng
            break
        except Exception as e:
            logging.error(f"Unexpected error processing query '{query}', page {page}: {e}")
            break

    return image_urls[:num_images] # Trả về đúng số lượng yêu cầu

# --- Main Execution ---
if __name__ == "__main__":
    DOWNLOAD_BASE_DIR.mkdir(parents = True, exist_ok = True)
    total_downloaded_count = 0
    total_failed_count = 0

    for category, terms in LANDSCAPE_CATEGORIES.items():
        category_dir = DOWNLOAD_BASE_DIR / sanitize_filename(category)
        category_dir.mkdir(exist_ok = True)
        logging.info(f"\n--- Processing Category: {category} ---")

        images_for_category = 0
        max_images_per_term = IMAGES_PER_CATEGORY // len(terms) if terms else IMAGES_PER_CATEGORY

        for term in terms:
            term_dir = category_dir / sanitize_filename(term)
            term_dir.mkdir(exist_ok = True)
            logging.info(f"--- Processing Term: '{term}' (Target: {max_images_per_term}) ---")

            fetched_urls_info = fetch_pexels_images(term, max_images_per_term)

            if not fetched_urls_info:
                logging.warning(f"No images fetched for term: '{term}'")
                continue

            logging.info(f"Attempting to download {len(fetched_urls_info)} images for term '{term}'...")
            term_downloaded = 0
            term_failed = 0
            for img_info in tqdm(fetched_urls_info, desc=f"Downloading '{term}'"):
                img_id = img_info.get('id')
                img_url = img_info.get('url')
                if not img_id or not img_url:
                    continue

                try:
                     # Cố gắng lấy phần mở rộng từ URL
                     url_path = urlparse(img_url).path
                     ext = os.path.splitext(url_path)[1] if os.path.splitext(url_path)[1] else '.jpg'
                except:
                     ext = '.jpg' # Fallback

                filename = f"pexels_{img_id}{ext}"
                save_path = term_dir / filename

                if download_single_image(img_url, save_path):
                    term_downloaded += 1
                else:
                    term_failed += 1

            logging.info(f"Term '{term}' completed. Downloaded: {term_downloaded}, Failed: {term_failed}")
            total_downloaded_count += term_downloaded
            total_failed_count += term_failed
            images_for_category += term_downloaded

        logging.info(f"Category '{category}' completed. Total downloaded for category: {images_for_category}")

    logging.info(f"\n--- Crawling Finished ---")
    logging.info(f"Total images downloaded successfully: {total_downloaded_count}")
    logging.info(f"Total images failed to download: {total_failed_count}")