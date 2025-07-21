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
"""
download_hist.py

This script uses the get_data module to download OCOD history files.
"""

from pathlib import Path
from enhance_ocod.get_data import download_ocod_history, download_csv


# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

# Create output directory path
OUTPUT_DIR = SCRIPT_DIR / ".." / "data" / "ocod_history"

def main():
    """Download key files for project"""

    # Download the price_paid dataset, this is hefty and will take some time. Size july 2025 was 4.3 GB
    download_csv("http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-complete.csv",
                 save_path = 'enhance_ocod/data/price_paid_data/price_paid_complete.csv')
    
    # download the ocod history
    success_count, total_files = download_ocod_history(
        output_dir=OUTPUT_DIR,
        file_type="FULL",
        confirm=True,
        show_progress=True
    )
    
    print(f"Process completed: {success_count}/{total_files} files successfully downloaded")

if __name__ == "__main__":
    main()