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

# Create output directory if it doesn't exist
OUTPUT_DIR = "./data/ocod_history"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_session():
    """Create a requests session for connection reuse"""
    session = requests.Session()
    session.headers.update({
        "Authorization": API_KEY,
        "Accept": "application/json"
    })
    return session

def get_api_data(session, endpoint, context="API request", max_retries=3):
    """
    Generic function to make API requests with retry logic for SSL errors
    
    Args:
        session: requests session object
        endpoint (str): The API endpoint path (without base URL)
        context (str): Description for error messages
        max_retries (int): Maximum number of retry attempts
    
    Returns:
        dict: JSON response data if successful, None if failed
    """
    url = f"{BASE_URL}{endpoint}"
    
    for attempt in range(max_retries):
        try:
            response = session.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error with {context}: {response.status_code}")
                print(response.text)
                return None
                
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"Connection error on attempt {attempt + 1}/{max_retries} for {context}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                print(f"Failed after {max_retries} attempts for {context}: {str(e)}")
                return None
                
        except Exception as e:
            print(f"Unexpected error with {context}: {str(e)}")
            return None
    
    return None

def download_file(session, file_name):
    output_path = os.path.join(OUTPUT_DIR, file_name)
    
    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"File {file_name} already exists. Skipping.")
        return True
    
    # Get download link
    download_data = get_api_data(session, f"datasets/history/ocod/{file_name}",
                            f"getting download link for {file_name}")
    if not download_data or not download_data.get("success"):
        print(f"Failed to get download link for {file_name}")
        return False
    
    download_url = download_data.get("result", {}).get("download_url")
    if not download_url:
        print(f"No download URL found for {file_name}")
        return False
    
    # Download the file
    try:
        response = session.get(download_url, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            print(f"Download error for {file_name}: {response.status_code}")
            return False
    except Exception as e:
        print(f"Exception while downloading {file_name}: {str(e)}")
        return False

# Main function to download all FULL files from OCOD history
def main():
    # Create session for connection reuse
    session = create_session()
    
    # Get the history data
    history_data = get_api_data(session, "datasets/history/ocod", "getting OCOD history")

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
    for file_name in tqdm(full_files):
        if download_file(session, file_name):
            success_count += 1
        # Small delay between downloads to avoid overwhelming the server
        time.sleep(1)
    
    print(f"\nDownload complete. Successfully downloaded {success_count} out of {len(full_files)} files.")

if __name__ == "__main__":
    main()