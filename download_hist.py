import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("LANDREGISTRY_API")
# Base URL
BASE_URL = "https://use-land-property-data.service.gov.uk/api/v1/"

# Headers
headers = {
    "Authorization": API_KEY,
    "Accept": "application/json"
}

# Create output directory if it doesn't exist
OUTPUT_DIR = "./data/ocod_history"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get the full history of the OCOD dataset
def get_ocod_history():
    response = requests.get(f"{BASE_URL}datasets/history/ocod", headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error getting history: {response.status_code}")
        print(response.text)
        return None

# Get download link for a historical file
def get_download_link(file_name):
    response = requests.get(f"{BASE_URL}datasets/history/ocod/{file_name}", headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error getting download link for {file_name}: {response.status_code}")
        print(response.text)
        return None

# Download a file with retry mechanism
def download_file(file_name):
    output_path = os.path.join(OUTPUT_DIR, file_name)
    
    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"File {file_name} already exists. Skipping.")
        return True
    
    # Get download link
    download_data = get_download_link(file_name)
    if not download_data or not download_data.get("success"):
        print(f"Failed to get download link for {file_name}")
        return False
    
    download_url = download_data.get("result", {}).get("download_url")
    if not download_url:
        print(f"No download URL found for {file_name}")
        return False
    
    # Download the file
    try:
        response = requests.get(download_url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                desc=file_name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
            
            print(f"Successfully downloaded {file_name}")
            return True
        else:
            print(f"Download error for {file_name}: {response.status_code}")
            return False
    except Exception as e:
        print(f"Exception while downloading {file_name}: {str(e)}")
        return False

# Main function to download all FULL files from OCOD history
def main():
    # Get the history data
    history_data = get_ocod_history()
    
    if not history_data or not history_data.get("success", False):
        print("Failed to retrieve OCOD history data")
        return
    
    # Filter only FULL files (not COU files)
    full_files = [item["filename"] for item in history_data.get("dataset_history", []) 
                  if "OCOD_FULL_" in item["filename"]]
    
    print(f"Found {len(full_files)} FULL files in OCOD history")
    
    # Sort files chronologically (newest first is typical default, but we'll sort just to be sure)
    full_files.sort(reverse=True)
    
    # Print file list
    print("\nFiles to download:")
    for i, file_name in enumerate(full_files):
        print(f"{i+1}. {file_name}")
    
    # Confirm with user
    confirm = input(f"\nDo you want to download all {len(full_files)} files? (y/n): ")
    if confirm.lower() != 'y':
        print("Download canceled")
        return
    
    # Download files sequentially (URLs expire quickly, so parallel would be problematic)
    success_count = 0
    for file_name in full_files:
        print(f"\nDownloading {file_name}...")
        if download_file(file_name):
            success_count += 1
        # Small delay between downloads to avoid overwhelming the server
        time.sleep(1)
    
    print(f"\nDownload complete. Successfully downloaded {success_count} out of {len(full_files)} files.")

if __name__ == "__main__":
    main()