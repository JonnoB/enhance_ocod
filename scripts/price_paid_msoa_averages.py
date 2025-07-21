"""
MSOA Average Price Calculator for OCOD Processing

This script processes historical OCOD (Overseas Companies Ownership Data) files to calculate
rolling 3-year average property prices at the MSOA (Middle Layer Super Output Area) level.
The script creates time-series average price data that can be used for property valuation
and market analysis in conjunction with OCOD data.

Purpose:
--------
For each OCOD file representing a specific month/year, this script:
1. Extracts the date from the OCOD filename
2. Defines a rolling 3-year period ending at that date
3. Loads and filters UK Land Registry Price Paid data for that period
4. Calculates mean and median property prices by MSOA code
5. Saves the results as parquet files for efficient storage and retrieval

Workflow:
---------
1. Iterates through all processed OCOD files in the input directory
2. For each OCOD file:
   - Extracts year and month from filename (format: OCOD_FULL_YYYY_MM)
   - Calculates 3-year rolling window (e.g., 2020-01 to 2023-01 for a 2023-01 OCOD file)
   - Loads relevant Price Paid data files covering the date range
   - Filters to standard residential property types (D, S, F, T) and category A transactions
   - Groups by MSOA code and calculates mean/median prices
   - Saves results as parquet file named by OCOD date

Input Requirements:
------------------
- data/ocod_history_processed/: Directory containing processed OCOD files
- data/processed_price_paid/: Directory containing preprocessed Price Paid parquet files
  organized by year (price_paid_YYYY.parquet format)

Output:
-------
- data/price_paid_msoa_averages/: Directory containing average price files
  - Files named: price_paid_YY_MM.parquet
  - Each file contains columns: msoa11cd, price_mean, price_median

Dependencies:
------------
- pandas: Data manipulation and analysis
- pathlib: Path handling
- enhance_ocod.price_paid_process: Custom functions for date extraction and price calculations

Usage Notes:
-----------
- Creates output directory if it doesn't exist
- Skips files that don't match expected OCOD filename format
- Handles missing data gracefully with appropriate logging
- Uses efficient parquet format for both input and output
- Memory efficient processing by loading only required years of price data

Example Output Structure:
------------------------
price_paid_msoa_averages/
├── price_paid_19_01.parquet  # Averages for Jan 2019 OCOD (using 2016-2019 price data)
├── price_paid_19_02.parquet  # Averages for Feb 2019 OCOD (using 2016-2019 price data)
└── ...

Each parquet file contains:
- msoa11cd: MSOA area code
- price_mean: Mean property price over 3-year period
- price_median: Median property price over 3-year period
"""

from enhance_ocod.price_paid_process import (
    get_date_from_ocod_filename,
    get_rolling_date_range,
    load_and_filter_price_data,
    calculate_average_prices,
)
from pathlib import Path

# Configuration
OCOD_FOLDER_PATH = Path("data/ocod_history_processed")
PRICE_PAID_FOLDER = Path("data/processed_price_paid")
OUTPUT_FOLDER = Path("data/price_paid_msoa_averages")


def main():
    # Create output directory safely
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Validate input directories exist
    if not OCOD_FOLDER_PATH.exists():
        raise FileNotFoundError(f"OCOD folder not found: {OCOD_FOLDER_PATH}")
    if not PRICE_PAID_FOLDER.exists():
        raise FileNotFoundError(f"Price paid folder not found: {PRICE_PAID_FOLDER}")

    # Get list of OCOD files and sort them
    ocod_files = [f for f in OCOD_FOLDER_PATH.iterdir() if f.is_file()]
    ocod_files.sort()  # Process in chronological order

    print(f"Found {len(ocod_files)} OCOD files to process")

    for ocod_file in ocod_files:
        try:
            process_ocod_file(ocod_file)
        except Exception as e:
            print(f"Error processing {ocod_file.name}: {e}")
            continue  # Continue with next file rather than crashing


def process_ocod_file(ocod_file):
    """Process a single OCOD file"""
    print(f"\nProcessing {ocod_file.name}")

    # Extract year and month from OCOD filename
    year, month = get_date_from_ocod_filename(ocod_file.name)

    if year is None or month is None:
        print(f"Could not extract date from {ocod_file.name}")
        return

    print(f"OCOD file date: {year}-{month:02d}")

    # Check if output already exists (skip if already processed)
    output_filename = OUTPUT_FOLDER / f"price_paid_{year:04d}_{month:02d}.parquet"
    if output_filename.exists():
        print(f"Output already exists, skipping: {output_filename.name}")
        return

    # Get rolling 3-year date range
    start_date, end_date = get_rolling_date_range(year, month, years_back=3)
    print(f"Date range: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")

    # Load and filter price paid data
    price_data = load_and_filter_price_data(PRICE_PAID_FOLDER, start_date, end_date)

    if price_data.empty:
        print("No price data found for date range")
        return

    print(f"Loaded {len(price_data)} price records")

    # Calculate average prices
    average_prices = calculate_average_prices(price_data)

    if average_prices.empty:
        print("No data after filtering")
        return

    print(f"Calculated averages for {len(average_prices)} MSOA areas")

    # Save results
    average_prices.to_parquet(output_filename, index=False)
    print(f"Saved {output_filename.name}")


if __name__ == "__main__":
    main()
