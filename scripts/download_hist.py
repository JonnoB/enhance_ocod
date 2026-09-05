"""
download_hist.py

Downloads all the bulk datasets the project depends on, using the
enhance_ocod.get_data module. Each dataset is skipped if its target directory
already contains files, so the script is safe to re-run.

Datasets downloaded (in order):

1. Land Registry Price Paid Data (complete CSV, ~4.3 GB as of July 2025)
   -> data/price_paid_data/price_paid_complete.csv
2. ONS Postcode Directory (ONSPD), latest release from ArcGIS
   -> data/onspd/
3. VOA rating list, latest "baseline" listentries file from Azure blob storage
   -> data/voa/
4. OCOD history files ("FULL" only) from the Land Registry API
   -> data/ocod_history/

All downloads run non-interactively (confirm=False); nothing prompts for
confirmation.

Environment variable required:
    LANDREGISTRY_API: API key for the Land Registry API, used for the OCOD
    history download. The Price Paid, ONSPD and VOA sources need no
    authentication.

Typical usage:
    python download_hist.py
"""


from pathlib import Path
from enhance_ocod.get_data import download_ocod_history, download_csv, download_latest_onspd, get_voa_file_list, VOARatingListDownloader


# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

# Create output directory paths, all anchored to the repo root so the script
# can be run from any working directory
DATA_DIR = (SCRIPT_DIR / ".." / "data").resolve()
OUTPUT_DIR = DATA_DIR / "ocod_history"
PRICE_PAID_DIR = DATA_DIR / "price_paid_data"
VOA_DIR = DATA_DIR / "voa"
ONSPD_DIR = DATA_DIR / "onspd"

def check_directory_has_files(directory_path):
    """
    Check if a directory exists and contains any files.
    
    Args:
        directory_path (Path): Path to the directory to check
        
    Returns:
        bool: True if directory exists and has files, False otherwise
    """
    if not directory_path.exists():
        return False
    
    # Check if directory has any files (not just subdirectories)
    return any(item.is_file() for item in directory_path.iterdir())

def main():
    """Download key files for project"""

    # Download the price_paid dataset from landregistry only if no files exist in the target directory
    PRICE_PAID_DIR.mkdir(parents=True, exist_ok=True)
    
    if check_directory_has_files(PRICE_PAID_DIR):
        print(f"Price paid data directory already contains files. Skipping download.")
    else:
        print("Downloading price paid dataset (this is hefty and will take some time. Size july 2025 was 4.3 GB)")
        download_csv("http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-complete.csv",
                     save_path = PRICE_PAID_DIR / 'price_paid_complete.csv')

    # DOwnload the ONSPD files this is only about 10 seconds
    if check_directory_has_files(ONSPD_DIR):
        print(f"ONSPD directory already contains files. Skipping download.")
    else:
        print("Downloading ONSPD dataset")
        download_latest_onspd(ONSPD_DIR, confirm  = False)

    # Download VOA data only if no files exist in the target directory
    VOA_DIR.mkdir(parents=True, exist_ok=True)
    
    if check_directory_has_files(VOA_DIR):
        print(f"VOA data directory already contains files. Skipping download.")
    else:
        print("Downloading latest baseline listentries file")
        # Get latest baseline listentries file
        files = get_voa_file_list(name_contains="listentries")
        baseline_files = [f for f in files if "baseline" in f['name']]
        latest_baseline = baseline_files[-1]

        # Download it
        downloader = VOARatingListDownloader()
        downloader.download_files(str(VOA_DIR),name_contains = latest_baseline['name'] , confirm = False)
    
    # Download the OCOD history (this already has built-in file checking)
    print("Downloading OCOD history files")
    success_count, total_files = download_ocod_history(
        output_dir=OUTPUT_DIR,
        file_type="FULL",
        confirm=False,
        show_progress=True
    )
    
    print(f"Process completed: {success_count}/{total_files} files successfully downloaded")

if __name__ == "__main__":
    main()