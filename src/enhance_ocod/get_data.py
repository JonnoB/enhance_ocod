"""
get_data.py

A module for downloading data from various sources, including the UK Land Registry API
and VOA Rating Lists. This module provides a unified interface for different data sources
with shared functionality extracted into base classes.
"""

import requests
import os
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
from typing import Optional, Union, List, Dict, Any
import io
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod


#####################################################################
## Plain csv download

def download_csv(url: str, 
                 save_path: Optional[Union[str, Path]] = None, 
                 return_df: bool = False,
                 chunk_size: int = 8192) -> Optional[pd.DataFrame]:
    """
    Download a CSV file from a URL with options to save and/or return as DataFrame.
    
    Args:
        url (str): The URL to download the CSV file from
        save_path (str or Path, optional): Path where to save the file. If None, file won't be saved.
        return_df (bool): If True, returns the data as a pandas DataFrame
        chunk_size (int): Size of chunks to download (in bytes)
        
    Returns:
        pd.DataFrame or None: DataFrame if return_df=True, otherwise None
    """
    if save_path is None and not return_df:
        print("Warning: Data will not be saved or returned. Terminating operation.")
        return None
    
    try:
        print(f"Downloading CSV from: {url}")
        
        # For large files, we should save first, then optionally load
        if save_path is None and return_df:
            # If we need to return df but have no save path, create a temp file
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv')
            save_path = Path(temp_file.name)
            temp_file.close()
            delete_after = True
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            delete_after = False
        
        # Stream download to file
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded / (1024**3):.2f} GB / {total_size / (1024**3):.2f} GB)", 
                              end='', flush=True)
        
        print(f"\nSuccessfully downloaded to: {save_path}")
        
        # Only load into DataFrame if requested
        if return_df:
            print("Loading CSV into DataFrame...")
            df = pd.read_csv(save_path)
            print(f"DataFrame loaded with shape: {df.shape}")
            
            if delete_after:
                save_path.unlink()  # Delete temp file
            
            return df
        
        return None
            
    except (requests.exceptions.RequestException, pd.errors.ParserError, IOError) as e:
        print(f"Error in download_csv: {e}")
        raise


######################################################
## Base Downloader Class
######################################################

class BaseDownloader(ABC):
    """Abstract base class for all downloaders"""
    
    def __init__(self):
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with basic configuration"""
        session = requests.Session()
        session.headers.update(self._get_default_headers())
        return session
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests. Override in subclasses if needed."""
        return {"Accept": "application/json"}
    
    def _make_request_with_retry(self, url: str, context: str = "Request", 
                                max_retries: int = 3, **kwargs) -> Optional[requests.Response]:
        """
        Make HTTP request with retry logic for connection errors
        
        Args:
            url (str): URL to request
            context (str): Description for error messages
            max_retries (int): Maximum retry attempts
            **kwargs: Additional arguments for requests
            
        Returns:
            requests.Response or None: Response if successful
        """
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, **kwargs)
                response.raise_for_status()
                return response
                
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Connection error on attempt {attempt + 1}/{max_retries} for {context}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Failed after {max_retries} attempts for {context}: {str(e)}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                print(f"Request error for {context}: {str(e)}")
                return None
        
        return None
    
    def _download_single_file(self, url: str, output_path: Path, 
                            overwrite: bool = False) -> bool:
        """Optimized download for large ONSPD files"""
        if output_path.exists() and not overwrite:
            print(f"File {output_path.name} already exists, skipping")
            return True

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use 1MB chunks instead of 8KB - this will be ~125x faster!
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    f.write(chunk)
            
            print(f"Successfully downloaded {output_path.name}")
            return True
            
        except Exception as e:
            print(f"Error downloading {output_path.name}: {e}")
            return False
    
    def _confirm_download(self, files: List[Any], confirm: bool = True) -> bool:
        """
        Show file list and get user confirmation
        
        Args:
            files: List of files to download
            confirm (bool): Whether to ask for confirmation
            
        Returns:
            bool: True if should proceed
        """
        if not files:
            print("No files to download")
            return False
        
        print(f"\nFiles to download:")
        for i, file_info in enumerate(files, 1):
            file_desc = self._format_file_description(file_info, i)
            print(file_desc)
        
        if confirm:
            response = input(f"\nDownload {len(files)} files? (y/n): ")
            if response.lower() != 'y':
                print("Download cancelled")
                return False
        
        return True
    
    @abstractmethod
    def _format_file_description(self, file_info: Any, index: int) -> str:
        """Format file description for confirmation display"""
        pass
    
    @abstractmethod
    def get_available_files(self, **kwargs) -> List[Any]:
        """Get list of available files"""
        pass
    
    @abstractmethod
    def download_files(self, output_dir: Union[str, Path], **kwargs) -> tuple:
        """Download files. Returns (successful_count, total_count)"""
        pass


######################################################
## Land Registry OCOD downloader
######################################################

class LandRegistryDownloader(BaseDownloader):
    """Handler for Land Registry API downloads"""
    
    def __init__(self, api_key=None, base_url=None):
        load_dotenv()
        self.api_key = api_key or os.environ.get("LANDREGISTRY_API")
        self.base_url = base_url or "https://use-land-property-data.service.gov.uk/api/v1/"
        
        if not self.api_key:
            raise ValueError("API key must be provided or set in LANDREGISTRY_API environment variable")
        
        super().__init__()
    
    def _get_default_headers(self) -> Dict[str, str]:
        return {
            "Authorization": self.api_key,
            "Accept": "application/json"
        }
    
    def _format_file_description(self, file_info: str, index: int) -> str:
        return f"{index}. {file_info}"
    
    def _get_api_data(self, endpoint: str, context: str = "API request") -> Optional[Dict]:
        """Get JSON data from API endpoint"""
        url = f"{self.base_url}{endpoint}"
        response = self._make_request_with_retry(url, context)
        
        if response and response.status_code == 200:
            return response.json()
        else:
            print(f"API error for {context}: {response.status_code if response else 'No response'}")
            return None
    
    def get_available_files(self, file_type: str = "FULL") -> List[str]:
        """
        Get list of available OCOD history files
        
        Args:
            file_type (str): Type of files ("FULL" or "COU")
            
        Returns:
            List[str]: List of filenames
        """
        history_data = self._get_api_data("datasets/history/ocod", "getting OCOD history")
        
        if not history_data or not history_data.get("success", False):
            print("Failed to retrieve OCOD history data")
            return []
        
        files = [
            item["filename"]
            for item in history_data.get("dataset_history", [])
            if f"OCOD_{file_type}_" in item["filename"]
        ]
        
        files.sort(reverse=True)  # Newest first
        return files
    
    def _download_ocod_file(self, filename: str, output_dir: Path) -> bool:
        """Download a single OCOD file"""
        output_path = output_dir / filename
        
        if output_path.exists():
            return True
        
        # Get download link
        download_data = self._get_api_data(
            f"datasets/history/ocod/{filename}",
            f"getting download link for {filename}"
        )
        
        if not download_data or not download_data.get("success"):
            print(f"Failed to get download link for {filename}")
            return False
        
        download_url = download_data.get("result", {}).get("download_url")
        if not download_url:
            print(f"No download URL found for {filename}")
            return False
        
        # Download using plain requests (no auth header needed for actual download)
        return self._download_single_file(download_url, output_path)
    
    def download_files(self, output_dir: Union[str, Path], file_type: str = "FULL",
                      confirm: bool = True, show_progress: bool = True) -> tuple:
        """
        Download OCOD history files
        
        Args:
            output_dir (str or Path): Directory to save files
            file_type (str): Type of files ("FULL" or "COU")
            confirm (bool): Whether to ask for confirmation
            show_progress (bool): Whether to show progress bar
            
        Returns:
            tuple: (success_count, total_files)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_files = self.get_available_files(file_type)
        
        if not all_files:
            print(f"No {file_type} files found")
            return 0, 0
        
        # Filter files that need downloading
        files_to_download = [f for f in all_files if not (output_dir / f).exists()]
        
        print(f"Found {len(all_files)} {file_type} files")
        print(f"{len(files_to_download)} files need downloading "
              f"(skipping {len(all_files) - len(files_to_download)} already present)")
        
        if not files_to_download:
            print("All files already downloaded.")
            return len(all_files), len(all_files)
        
        if not self._confirm_download(files_to_download, confirm):
            return 0, len(all_files)
        
        # Download files
        success_count = 0
        file_iterator = tqdm(files_to_download) if show_progress else files_to_download
        
        for filename in file_iterator:
            if self._download_ocod_file(filename, output_dir):
                success_count += 1
            time.sleep(1)  # Rate limiting
        
        print(f"\nDownload complete: {success_count}/{len(files_to_download)} files downloaded")
        return success_count, len(all_files)


######################################################
## VOA Rating Lists downloader
######################################################

class VOARatingListDownloader(BaseDownloader):
    """Handler for VOA Rating List downloads from Azure blob storage"""
    
    def __init__(self, base_url="https://voaratinglists.blob.core.windows.net/downloads"):
        self.base_url = base_url
        self.list_url = f"{base_url}?restype=container&comp=list"
        super().__init__()
    
    def _format_file_description(self, file_info: Dict, index: int) -> str:
        size_mb = file_info['size'] / (1024*1024) if file_info['size'] else 'Unknown'
        return f"{index}. {file_info['name']} ({size_mb:.1f} MB)"
    
    def get_available_files(self) -> List[Dict]:
        """
        Get list of all available files from VOA API
        
        Returns:
            List[Dict]: List of file information dictionaries
        """
        try:
            response = self._make_request_with_retry(self.list_url, "fetching VOA file list")
            
            if not response:
                return []
            
            root = ET.fromstring(response.content)
            files = []
            
            blobs_element = root.find('Blobs')
            if blobs_element is not None:
                for blob in blobs_element.findall('Blob'):
                    name_elem = blob.find('Name')
                    properties = blob.find('Properties')
                    
                    if name_elem is not None and properties is not None:
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
            
        except ET.ParseError as e:
            print(f"Error parsing XML response: {e}")
            return []
    
    def filter_files(self, files: List[Dict], 
                    name_contains: Optional[str] = None,
                    file_extension: Optional[str] = None) -> List[Dict]:
        """Filter files based on criteria"""
        filtered = files
        
        if name_contains:
            filtered = [f for f in filtered if name_contains.lower() in f['name'].lower()]
        
        if file_extension:
            if not file_extension.startswith('.'):
                file_extension = '.' + file_extension
            filtered = [f for f in filtered if f['name'].lower().endswith(file_extension.lower())]
        
        print(f"Filtered to {len(filtered)} files")
        return filtered
    
    def download_files(self, output_dir: Union[str, Path],
                      name_contains: Optional[str] = None,
                      file_extension: Optional[str] = None,
                      overwrite: bool = False,
                      confirm: bool = True) -> tuple:
        """
        Download VOA rating list files
        
        Args:
            output_dir (str or Path): Directory to save files
            name_contains (str, optional): Filter files containing this string
            file_extension (str, optional): Filter files with this extension
            overwrite (bool): Whether to overwrite existing files
            confirm (bool): Whether to ask for confirmation
            
        Returns:
            tuple: (successful_downloads, total_files)
        """
        output_dir = Path(output_dir)
        
        all_files = self.get_available_files()
        if not all_files:
            return 0, 0
        
        filtered_files = self.filter_files(all_files, name_contains, file_extension)
        
        if not self._confirm_download(filtered_files, confirm):
            return 0, len(filtered_files)
        
        successful = 0
        for file_info in filtered_files:
            output_path = output_dir / file_info['name']
            if self._download_single_file(file_info['url'], output_path, overwrite):
                successful += 1
        
        print(f"\nDownload complete: {successful}/{len(filtered_files)} files downloaded successfully")
        return successful, len(filtered_files)


######################################################
## Convenience Functions
######################################################

def download_ocod_history(output_dir, file_type="FULL", api_key=None, 
                         confirm=True, show_progress=True):
    """Convenience function to download OCOD history files"""
    downloader = LandRegistryDownloader(api_key=api_key)
    return downloader.download_files(
        output_dir=output_dir,
        file_type=file_type,
        confirm=confirm,
        show_progress=show_progress
    )

def get_ocod_file_list(file_type="FULL", api_key=None):
    """Convenience function to get list of available OCOD files"""
    downloader = LandRegistryDownloader(api_key=api_key)
    return downloader.get_available_files(file_type=file_type)

def download_voa_rating_lists(output_dir: Union[str, Path],
                             name_contains: Optional[str] = None,
                             file_extension: Optional[str] = None,
                             overwrite: bool = False,
                             confirm: bool = True) -> tuple:
    """Convenience function to download VOA rating list files"""
    downloader = VOARatingListDownloader()
    return downloader.download_files(
        output_dir, name_contains, file_extension, overwrite, confirm
    )

def get_voa_file_list(name_contains: Optional[str] = None,
                     file_extension: Optional[str] = None) -> List[Dict]:
    """Get list of available VOA rating list files"""
    downloader = VOARatingListDownloader()
    all_files = downloader.get_available_files()
    return downloader.filter_files(all_files, name_contains, file_extension)


######################################################
## ONSPD (ONS Postcode Directory) downloader
######################################################

class ONSPDDownloader(BaseDownloader):
    """Handler for ONS Postcode Directory downloads from ArcGIS"""
    
    def __init__(self, base_url="https://www.arcgis.com/sharing/rest"):
        self.base_url = base_url
        super().__init__()
    
    def _format_file_description(self, file_info: Dict, index: int) -> str:
        size_mb = file_info['size'] / (1024*1024) if file_info.get('size') else 'Unknown'
        modified = file_info.get('modified_readable', 'Unknown')
        return f"{index}. {file_info['title']} ({size_mb:.1f} MB, Modified: {modified})"
    
    def _search_onspd_items(self) -> List[Dict]:
        """Search for ONSPD items on ArcGIS"""
        search_url = f"{self.base_url}/search"
        params = {
            'q': 'ONSPD OR "ONS Postcode Directory"',
            'f': 'json',
            'num': 100,
            'sortField': 'modified',
            'sortOrder': 'desc'
        }
        
        response = self._make_request_with_retry(
            search_url, "searching for ONSPD items", params=params
        )
        
        if not response:
            return []
        
        try:
            data = response.json()
            return data.get('results', [])
        except ValueError as e:
            print(f"Error parsing search response: {e}")
            return []
    
    def _get_item_details(self, item_id: str) -> Optional[Dict]:
        """Get detailed information about a specific item"""
        url = f"{self.base_url}/content/items/{item_id}"
        params = {'f': 'json'}
        
        response = self._make_request_with_retry(
            url, f"getting details for item {item_id}", params=params
        )
        
        if not response:
            return None
        
        try:
            return response.json()
        except ValueError as e:
            print(f"Error parsing item details: {e}")
            return None
    
    def get_available_files(self) -> List[Dict]:
        """
        Get list of available ONSPD files, prioritizing CSV Collections
        
        Returns:
            List[Dict]: List of ONSPD file information, sorted by recency
        """
        print("Searching for ONSPD datasets...")
        items = self._search_onspd_items()
        
        if not items:
            print("No ONSPD items found")
            return []
        
        # Filter and enhance items
        onspd_files = []
        for item in items:
            # Look for ONSPD-specific indicators
            title = item.get('title', '').lower()
            tags = [tag.lower() for tag in item.get('tags', [])]
            item_type = item.get('type', '')
            
            # Check if this looks like an ONSPD dataset
            is_onspd = any([
                'onspd' in title,
                'ons postcode directory' in title,
                'postcode directory' in title,
                any('onspd' in tag for tag in tags),
                any('ons postcode directory' in tag for tag in tags)
            ])
            
            if not is_onspd:
                continue
            
            # Get detailed information
            details = self._get_item_details(item['id'])
            if not details:
                continue
            
            # Convert timestamp to readable format
            from datetime import datetime
            modified_timestamp = details.get('modified')
            modified_readable = 'Unknown'
            if modified_timestamp:
                try:
                    modified_readable = datetime.fromtimestamp(
                        modified_timestamp / 1000
                    ).strftime('%Y-%m-%d')
                except (ValueError, OSError):
                    pass
            
            file_info = {
                'id': item['id'],
                'title': details.get('title', 'Unknown'),
                'type': details.get('type', 'Unknown'),
                'owner': details.get('owner', 'Unknown'),
                'size': details.get('size', 0),
                'modified': modified_timestamp,
                'modified_readable': modified_readable,
                'filename': details.get('name', f"onspd_{item['id']}.zip"),
                'description': details.get('description', ''),
                'download_url': f"{self.base_url}/content/items/{item['id']}/data",
                'is_csv_collection': item_type == 'CSV Collection'
            }
            
            onspd_files.append(file_info)
        
        # Sort by preference: CSV Collections first, then by modification date
        onspd_files.sort(key=lambda x: (
            not x['is_csv_collection'],  # CSV Collections first (False sorts before True)
            -(x['modified'] or 0)        # Then by most recent
        ))
        
        print(f"Found {len(onspd_files)} ONSPD datasets")
        return onspd_files
    
    def get_latest_onspd(self) -> Optional[Dict]:
        """
        Get the most recent ONSPD dataset, preferring CSV Collections
        
        Returns:
            Dict or None: Latest ONSPD file information
        """
        files = self.get_available_files()
        if not files:
            return None
        
        # The list is already sorted with CSV Collections first, then by date
        latest = files[0]
        
        print(f"Latest ONSPD: {latest['title']}")
        print(f"Type: {latest['type']}")
        print(f"Size: {latest['size'] / (1024*1024):.1f} MB")
        print(f"Modified: {latest['modified_readable']}")
        
        return latest
    
    def download_latest_onspd(self, output_dir: Union[str, Path], 
                             overwrite: bool = False,
                             confirm: bool = True) -> bool:
        """
        Download the latest ONSPD dataset
        
        Args:
            output_dir (str or Path): Directory to save the file
            overwrite (bool): Whether to overwrite existing files
            confirm (bool): Whether to ask for confirmation
            
        Returns:
            bool: True if successful
        """
        latest = self.get_latest_onspd()
        if not latest:
            print("No ONSPD dataset found")
            return False
        
        if confirm:
            size_mb = latest['size'] / (1024*1024)
            print(f"\nReady to download:")
            print(f"  File: {latest['title']}")
            print(f"  Size: {size_mb:.1f} MB")
            print(f"  Modified: {latest['modified_readable']}")
            print(f"  Type: {latest['type']}")
            
            response = input("\nDownload this file? (y/n): ")
            if response.lower() != 'y':
                print("Download cancelled")
                return False
        
        output_dir = Path(output_dir)
        output_path = output_dir / latest['filename']
        
        print(f"Downloading latest ONSPD to {output_path}...")
        
        success = self._download_single_file(
            latest['download_url'], 
            output_path, 
            overwrite=overwrite
        )
        
        if success:
            print(f"✅ Successfully downloaded latest ONSPD: {latest['filename']}")
        else:
            print(f"❌ Failed to download ONSPD")
        
        return success
    
    def download_files(self, output_dir: Union[str, Path],
                      download_all: bool = False,
                      overwrite: bool = False,
                      confirm: bool = True) -> tuple:
        """
        Download ONSPD files
        
        Args:
            output_dir (str or Path): Directory to save files
            download_all (bool): If True, download all available files; if False, download only latest
            overwrite (bool): Whether to overwrite existing files
            confirm (bool): Whether to ask for confirmation
            
        Returns:
            tuple: (successful_downloads, total_files)
        """
        if not download_all:
            # Download only the latest
            success = self.download_latest_onspd(output_dir, overwrite, confirm)
            return (1, 1) if success else (0, 1)
        
        # Download all available files
        all_files = self.get_available_files()
        
        if not all_files:
            return 0, 0
        
        if not self._confirm_download(all_files, confirm):
            return 0, len(all_files)
        
        output_dir = Path(output_dir)
        successful = 0
        
        for file_info in all_files:
            output_path = output_dir / file_info['filename']
            if self._download_single_file(file_info['download_url'], output_path, overwrite):
                successful += 1
        
        print(f"\nDownload complete: {successful}/{len(all_files)} files downloaded successfully")
        return successful, len(all_files)


######################################################
## Add to Convenience Functions section
######################################################

def download_latest_onspd(output_dir: Union[str, Path], 
                         overwrite: bool = False,
                         confirm: bool = True) -> bool:
    """
    Convenience function to download the latest ONSPD dataset
    
    Args:
        output_dir (str or Path): Directory to save the file
        overwrite (bool): Whether to overwrite existing files
        confirm (bool): Whether to ask for confirmation
        
    Returns:
        bool: True if successful
    """
    downloader = ONSPDDownloader()
    return downloader.download_latest_onspd(output_dir, overwrite, confirm)

def get_onspd_file_list() -> List[Dict]:
    """Get list of available ONSPD files"""
    downloader = ONSPDDownloader()
    return downloader.get_available_files()

def get_latest_onspd_info() -> Optional[Dict]:
    """Get information about the latest ONSPD dataset"""
    downloader = ONSPDDownloader()
    return downloader.get_latest_onspd()