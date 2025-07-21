"""
download_hist.py

This script automates the process of downloading the full OCOD (Open Charges Over Data) history files from the UK Land Registry API.
It performs the following steps:

1. Authenticates with the Land Registry API using an API key loaded from environment variables.
2. Retrieves metadata for all available OCOD history datasets.
3. Filters for "FULL" OCOD files and sorts them chronologically.
4. Skips files that have already been downloaded to the local output directory.
5. Prompts the user for confirmation before downloading any missing files.
6. For each file to be downloaded:
   - Requests a fresh, pre-signed S3 download URL from the API.
   - Downloads the file using the pre-signed URL (no Authorization header).
   - Saves the file to the output directory.

The script uses a progress bar for download status and provides clear logging for errors and skipped files.
It is designed to be robust against network errors and to avoid redundant downloads.

Environment variable required:
    LANDREGISTRY_API: The API key for authenticating with the Land Registry API.

Typical usage:
    python download_hist.py
"""

import requests
import os
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

load_dotenv()
API_KEY = os.environ.get("LANDREGISTRY_API")
# Base URL
BASE_URL = "https://use-land-property-data.service.gov.uk/api/v1/"

# Create output directory if it doesn't exist
OUTPUT_DIR = str(SCRIPT_DIR / ".." / "data" / "ocod_history")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_session():
    """Create a requests session for connection reuse"""
    session = requests.Session()
    session.headers.update({"Authorization": API_KEY, "Accept": "application/json"})
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
                wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                print(
                    f"Connection error on attempt {attempt + 1}/{max_retries} for {context}"
                )
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
        return True

    # Get download link
    download_data = get_api_data(
        session,
        f"datasets/history/ocod/{file_name}",
        f"getting download link for {file_name}",
    )
    if not download_data or not download_data.get("success"):
        print(f"Failed to get download link for {file_name}")
        return False

    download_url = download_data.get("result", {}).get("download_url")
    if not download_url:
        print(f"No download URL found for {file_name}")
        return False

    # Download the file using a plain requests.get (no Authorization header!)
    try:
        response = requests.get(download_url, stream=True)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            print(f"Download error for {file_name}: {response.status_code}")
            print(f"Download URL: {download_url}")
            print(f"Full response text: {response.text}")
            return False
    except Exception as e:
        print(f"Exception while downloading {file_name}: {str(e)}")
        print(f"Download URL: {download_url}")
        return False


# Main function to download all FULL files from OCOD history
def main():
    # Create session for connection reuse
    session = create_session()

    # Get the history data
    history_data = get_api_data(
        session, "datasets/history/ocod", "getting OCOD history"
    )

    if not history_data or not history_data.get("success", False):
        print("Failed to retrieve OCOD history data")
        return

    # Filter only FULL files (not COU files)
    full_files = [
        item["filename"]
        for item in history_data.get("dataset_history", [])
        if "OCOD_FULL_" in item["filename"]
    ]

    print(f"Found {len(full_files)} FULL files in OCOD history")

    # Sort files chronologically (newest first is typical default, but we'll sort just to be sure)
    full_files.sort(reverse=True)

    # Filter out files that already exist in OUTPUT_DIR
    files_to_download = [
        f for f in full_files if not os.path.exists(os.path.join(OUTPUT_DIR, f))
    ]

    print(
        f"\n{len(files_to_download)} files need to be downloaded (skipping {len(full_files) - len(files_to_download)} already present).\n"
    )

    # Print file list
    print("\nFiles to download:")
    for i, file_name in enumerate(files_to_download):
        print(f"{i + 1}. {file_name}")

    # Confirm with user
    confirm = input(
        f"\nDo you want to download all {len(files_to_download)} files? (y/n): "
    )
    if confirm.lower() != "y":
        print("Download canceled")
        return

    success_count = 0
    for file_name in tqdm(files_to_download):
        if download_file(session, file_name):
            success_count += 1
        # Small delay between downloads to avoid overwhelming the server
        time.sleep(1)

    print(
        f"\nDownload complete. Successfully downloaded {success_count} out of {len(full_files)} files."
    )


if __name__ == "__main__":
    main()
