"""
get_data.py

A module for downloading data from various sources, including the UK Land Registry API.
This module provides functions for downloading OCOD (Open Charges Over Data) history files
and can be extended with other data downloading functions.

Environment variable required:
    LANDREGISTRY_API: The API key for authenticating with the Land Registry API.
"""

import requests
import os
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
from typing import Optional, Union
import io



#####################################################################
## Plain csv download

def download_csv(url: str, 
                 save_path: Optional[Union[str, Path]] = None, 
                 return_df: bool = False) -> Optional[pd.DataFrame]:
    """
    Download a CSV file from a URL with options to save and/or return as DataFrame.
    
    Args:
        url (str): The URL to download the CSV file from
        save_path (str or Path, optional): Path where to save the file. If None, file won't be saved.
        return_df (bool): If True, returns the data as a pandas DataFrame
        
    Returns:
        pd.DataFrame or None: DataFrame if return_df=True, otherwise None
        
    Raises:
        requests.RequestException: If download fails
        pd.errors.ParserError: If CSV parsing fails
        IOError: If file saving fails
    """
    
    # Check if both save_path and return_df are False/None
    if save_path is None and not return_df:
        print("Warning: Data will not be saved or returned. Terminating operation.")
        return None
    
    try:
        # Download the file
        print(f"Downloading CSV from: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        # Read CSV content into DataFrame
        csv_content = io.StringIO(response.text)
        df = pd.read_csv(csv_content)
        print(f"Successfully downloaded CSV with shape: {df.shape}")
        
        # Save to file if path is provided
        if save_path is not None:
            save_path = Path(save_path)
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"File saved to: {save_path}")
        
        # Return DataFrame if requested
        if return_df:
            print("Returning DataFrame")
            return df
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        raise
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV: {e}")
        raise
    except IOError as e:
        print(f"Error saving file: {e}")
        raise



######################################################
##
##  Land Registry OCOD downloader
##
######################################################

class LandRegistryDownloader:
    """Handler for Land Registry API downloads"""
    
    def __init__(self, api_key=None, base_url=None):
        """
        Initialize the downloader
        
        Args:
            api_key (str, optional): API key. If None, loads from LANDREGISTRY_API env var
            base_url (str, optional): Base URL for API. Uses default if None
        """
        load_dotenv()
        self.api_key = api_key or os.environ.get("LANDREGISTRY_API")
        self.base_url = base_url or "https://use-land-property-data.service.gov.uk/api/v1/"
        
        if not self.api_key:
            raise ValueError("API key must be provided or set in LANDREGISTRY_API environment variable")
    
    def create_session(self):
        """Create a requests session for connection reuse"""
        session = requests.Session()
        session.headers.update({"Authorization": self.api_key, "Accept": "application/json"})
        return session

    def get_api_data(self, session, endpoint, context="API request", max_retries=3):
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
        url = f"{self.base_url}{endpoint}"

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

    def download_file(self, session, file_name, output_dir):
        """
        Download a single file from the Land Registry API
        
        Args:
            session: requests session object
            file_name (str): Name of the file to download
            output_dir (str): Directory to save the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        output_path = os.path.join(output_dir, file_name)

        # Skip if file already exists
        if os.path.exists(output_path):
            return True

        # Get download link
        download_data = self.get_api_data(
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

    def get_ocod_history_files(self, file_type="FULL"):
        """
        Get list of available OCOD history files
        
        Args:
            file_type (str): Type of files to retrieve ("FULL" or "COU")
            
        Returns:
            list: List of filenames, or empty list if failed
        """
        session = self.create_session()
        
        # Get the history data
        history_data = self.get_api_data(
            session, "datasets/history/ocod", "getting OCOD history"
        )

        if not history_data or not history_data.get("success", False):
            print("Failed to retrieve OCOD history data")
            return []

        # Filter files by type
        files = [
            item["filename"]
            for item in history_data.get("dataset_history", [])
            if f"OCOD_{file_type}_" in item["filename"]
        ]

        # Sort files chronologically (newest first)
        files.sort(reverse=True)
        return files

    def download_ocod_history(self, output_dir, file_type="FULL", confirm=True, show_progress=True):
        """
        Download OCOD history files
        
        Args:
            output_dir (str or Path): Directory to save files
            file_type (str): Type of files to download ("FULL" or "COU")
            confirm (bool): Whether to ask for user confirmation before downloading
            show_progress (bool): Whether to show progress bar
            
        Returns:
            tuple: (success_count, total_files)
        """
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of files
        all_files = self.get_ocod_history_files(file_type)
        
        if not all_files:
            print(f"No {file_type} files found in OCOD history")
            return 0, 0

        print(f"Found {len(all_files)} {file_type} files in OCOD history")

        # Filter out files that already exist
        files_to_download = [
            f for f in all_files if not (output_dir / f).exists()
        ]

        print(
            f"\n{len(files_to_download)} files need to be downloaded "
            f"(skipping {len(all_files) - len(files_to_download)} already present).\n"
        )

        if not files_to_download:
            print("All files already downloaded.")
            return len(all_files), len(all_files)

        # Print file list
        print("\nFiles to download:")
        for i, file_name in enumerate(files_to_download):
            print(f"{i + 1}. {file_name}")

        # Confirm with user if requested
        if confirm:
            user_input = input(
                f"\nDo you want to download all {len(files_to_download)} files? (y/n): "
            )
            if user_input.lower() != "y":
                print("Download canceled")
                return 0, len(all_files)

        # Download files
        session = self.create_session()
        success_count = 0
        
        file_iterator = tqdm(files_to_download) if show_progress else files_to_download
        
        for file_name in file_iterator:
            if self.download_file(session, file_name, str(output_dir)):
                success_count += 1
            # Small delay between downloads to avoid overwhelming the server
            time.sleep(1)

        print(
            f"\nDownload complete. Successfully downloaded {success_count} "
            f"out of {len(files_to_download)} files."
        )
        
        return success_count, len(all_files)


def download_ocod_history(output_dir, file_type="FULL", api_key=None, confirm=True, show_progress=True):
    """
    Convenience function to download OCOD history files
    
    Args:
        output_dir (str or Path): Directory to save files
        file_type (str): Type of files to download ("FULL" or "COU")
        api_key (str, optional): API key. If None, loads from environment
        confirm (bool): Whether to ask for user confirmation
        show_progress (bool): Whether to show progress bar
        
    Returns:
        tuple: (success_count, total_files)
    """
    downloader = LandRegistryDownloader(api_key=api_key)
    return downloader.download_ocod_history(
        output_dir=output_dir,
        file_type=file_type,
        confirm=confirm,
        show_progress=show_progress
    )


def get_ocod_file_list(file_type="FULL", api_key=None):
    """
    Convenience function to get list of available OCOD files
    
    Args:
        file_type (str): Type of files to retrieve ("FULL" or "COU")
        api_key (str, optional): API key. If None, loads from environment
        
    Returns:
        list: List of available filenames
    """
    downloader = LandRegistryDownloader(api_key=api_key)
    return downloader.get_ocod_history_files(file_type=file_type)


##########################
##
## VOA list entries downloader
##
############################

import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import requests
from pathlib import Path
from typing import List, Dict, Optional, Union
import pandas as pd
from datetime import datetime

class VOARatingListDownloader:
    """Handler for VOA Rating List downloads from Azure blob storage"""
    
    def __init__(self, base_url="https://voaratinglists.blob.core.windows.net/downloads"):
        """
        Initialize the VOA downloader
        
        Args:
            base_url (str): Base URL for the blob storage
        """
        self.base_url = base_url
        self.list_url = f"{base_url}?restype=container&comp=list"
    
    def get_available_files(self) -> List[Dict]:
        """
        Get list of all available files from the VOA rating lists API
        
        Returns:
            List[Dict]: List of dictionaries containing file information
                       Each dict has keys: 'name', 'last_modified', 'size', 'url'
        """
        try:
            response = requests.get(self.list_url, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            files = []
            
            # The structure is: EnumerationResults -> Blobs -> Blob
            # No namespace is used
            blobs_element = root.find('Blobs')
            if blobs_element is not None:
                for blob in blobs_element.findall('Blob'):
                    name_elem = blob.find('Name')
                    properties = blob.find('Properties')
                    
                    if name_elem is not None and properties is not None:
                        # Extract properties
                        modified_elem = properties.find('Last-Modified')
                        size_elem = properties.find('Content-Length')
                        content_type_elem = properties.find('Content-Type')
                        
                        file_info = {
                            'name': name_elem.text,
                            'last_modified': modified_elem.text if modified_elem is not None else None,
                            'size': int(size_elem.text) if size_elem is not None else None,
                            'content_type': content_type_elem.text if content_type_elem is not None else None,
                            'url': f"{self.base_url}/{name_elem.text}"
                        }
                        files.append(file_info)
            
            print(f"Found {len(files)} files available for download")
            return files
            
        except requests.RequestException as e:
            print(f"Error fetching file list: {e}")
            return []
        except ET.ParseError as e:
            print(f"Error parsing XML response: {e}")
            return []
    
    def filter_files(self, files: List[Dict], 
                    name_contains: Optional[str] = None,
                    file_extension: Optional[str] = None) -> List[Dict]:
        """
        Filter files based on criteria
        
        Args:
            files (List[Dict]): List of file dictionaries from get_available_files()
            name_contains (str, optional): Filter files containing this string in name
            file_extension (str, optional): Filter files with this extension (e.g., '.csv', '.zip')
            
        Returns:
            List[Dict]: Filtered list of files
        """
        filtered = files
        
        if name_contains:
            filtered = [f for f in filtered if name_contains.lower() in f['name'].lower()]
        
        if file_extension:
            if not file_extension.startswith('.'):
                file_extension = '.' + file_extension
            filtered = [f for f in filtered if f['name'].lower().endswith(file_extension.lower())]
        
        print(f"Filtered to {len(filtered)} files")
        return filtered
    
    def download_file(self, file_info: Dict, 
                     output_dir: Union[str, Path],
                     overwrite: bool = False) -> bool:
        """
        Download a single file
        
        Args:
            file_info (Dict): File information dictionary from get_available_files()
            output_dir (str or Path): Directory to save the file
            overwrite (bool): Whether to overwrite existing files
            
        Returns:
            bool: True if successful, False otherwise
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = output_dir / file_info['name']
        
        # Check if file exists and skip if not overwriting
        if file_path.exists() and not overwrite:
            print(f"File {file_info['name']} already exists, skipping")
            return True
        
        try:
            print(f"Downloading {file_info['name']}...")
            response = requests.get(file_info['url'], stream=True, timeout=60)
            response.raise_for_status()
            
            # Write file with progress indication for large files
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Successfully downloaded {file_info['name']}")
            return True
            
        except requests.RequestException as e:
            print(f"Error downloading {file_info['name']}: {e}")
            return False
        except IOError as e:
            print(f"Error saving {file_info['name']}: {e}")
            return False
    
    def download_files(self, files: List[Dict], 
                      output_dir: Union[str, Path],
                      overwrite: bool = False,
                      confirm: bool = True) -> tuple:
        """
        Download multiple files
        
        Args:
            files (List[Dict]): List of file dictionaries to download
            output_dir (str or Path): Directory to save files
            overwrite (bool): Whether to overwrite existing files
            confirm (bool): Whether to ask for confirmation before downloading
            
        Returns:
            tuple: (successful_downloads, total_files)
        """
        if not files:
            print("No files to download")
            return 0, 0
        
        print(f"\nFiles to download:")
        for i, file_info in enumerate(files, 1):
            size_mb = file_info['size'] / (1024*1024) if file_info['size'] else 'Unknown'
            print(f"{i}. {file_info['name']} ({size_mb:.1f} MB)")
        
        if confirm:
            response = input(f"\nDownload {len(files)} files? (y/n): ")
            if response.lower() != 'y':
                print("Download cancelled")
                return 0, len(files)
        
        successful = 0
        for file_info in files:
            if self.download_file(file_info, output_dir, overwrite):
                successful += 1
        
        print(f"\nDownload complete: {successful}/{len(files)} files downloaded successfully")
        return successful, len(files)


# Convenience functions
def download_voa_rating_lists(output_dir: Union[str, Path],
                             name_contains: Optional[str] = None,
                             file_extension: Optional[str] = None,
                             overwrite: bool = False,
                             confirm: bool = True) -> tuple:
    """
    Convenience function to download VOA rating list files
    
    Args:
        output_dir (str or Path): Directory to save files
        name_contains (str, optional): Filter files containing this string in name
        file_extension (str, optional): Filter files with this extension
        overwrite (bool): Whether to overwrite existing files
        confirm (bool): Whether to ask for confirmation
        
    Returns:
        tuple: (successful_downloads, total_files)
    """
    downloader = VOARatingListDownloader()
    
    # Get all available files
    all_files = downloader.get_available_files()
    if not all_files:
        return 0, 0
    
    # Apply filters
    filtered_files = downloader.filter_files(all_files, name_contains, file_extension)
    
    # Download files
    return downloader.download_files(filtered_files, output_dir, overwrite, confirm)


def get_voa_file_list(name_contains: Optional[str] = None,
                     file_extension: Optional[str] = None) -> List[Dict]:
    """
    Get list of available VOA rating list files
    
    Args:
        name_contains (str, optional): Filter files containing this string in name
        file_extension (str, optional): Filter files with this extension
        
    Returns:
        List[Dict]: List of available files with their metadata
    """
    downloader = VOARatingListDownloader()
    all_files = downloader.get_available_files()
    return downloader.filter_files(all_files, name_contains, file_extension)