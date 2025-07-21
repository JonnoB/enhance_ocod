"""
Price Paid Dataset Processing Module

This module provides functionality for processing and managing the UK Land Registry Price Paid dataset,
which is a large-scale CSV containing historical property transaction data. The module addresses several key challenges
associated with processing such a massive dataset:

Key Features:
- Chunked processing to handle large files without memory overflow
- Year-based data segmentation and selective loading
- Data cleaning and standardization
- Efficient storage using Parquet format
- Postcode and geographical district mapping

Main Functions:
- process_single_chunk: Process individual data chunks
- preprocess_and_save_by_year: Convert raw CSV to preprocessed, year-specific Parquet files
- load_years_data: Selectively load data for specific years
- load_and_process_pricepaid_data: preprocessing workflow that checks if pre-processed data exists and creates it if necessary

Workflow:
1. Raw CSV is processed in memory-efficient chunks
2. Data is cleaned, standardized, and enriched
3. Data is segmented by year and saved as Parquet files
4. Years can be loaded selectively for analysis

Dependencies:
- pandas
- numpy
- tqdm
- pathlib
- os

Example:
    # Preprocess and load data for specific years
    processed_data = load_and_process_pricepaid_data_new(
        'price_paid.csv',
        postcode_district_lookup,
        years_needed=[2019, 2020]
    )

Notes:
- Requires a postcode district lookup table
- Assumes specific column names in input CSV
- Handles various data cleaning scenarios
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from enhance_ocod.locate_and_classify import clean_street_numbers
from pandas.api.types import CategoricalDtype


def process_single_chunk(df, postcode_district_lookup):
    """
    Modified version that includes price in the output
    """
    # Process strings inplace to save memory
    df["street"] = df["street"].astype(str).str.lower()
    df["street_name2"] = (
        df["street"]
        .str.replace(r"'", "", regex=True)
        .str.replace(r"s(s)?(?=\s)", "", regex=True)
        .str.replace(r"\s", "", regex=True)
    )

    df.drop("street", axis=1, inplace=True)

    df["locality"] = df["locality"].astype(str).str.lower()
    df["paon"] = df["paon"].astype(str).str.lower()

    # Clean street numbers
    df = clean_street_numbers(df, original_column="paon")

    # Process postcode
    df["postcode2"] = (
        df["postcode"].astype(str).str.lower().str.replace(r"\s", "", regex=True)
    )
    df.drop("postcode", axis=1, inplace=True)

    # Merge with lookup
    df = df.merge(
        postcode_district_lookup, how="left", left_on="postcode2", right_on="postcode2"
    )

    # Keep necessary columns including price
    # final_columns = ['street_name2', 'street_number', 'postcode2', 'district',
    #               'paon', 'lad11cd', 'oa11cd', 'lsoa11cd', 'msoa11cd', 'price', 'year']

    # available_columns = [col for col in final_columns if col in df.columns]

    return df  # [available_columns]


def combine_temp_files_for_year(year, output_dir, remaining_data):
    """
    Combine temporary files and remaining data for a specific year
    """
    all_data = []

    # Add any remaining data in accumulator
    if remaining_data:
        all_data.extend(remaining_data)

    # Find and load temp files for this year
    temp_files = [f for f in os.listdir(output_dir) if f.startswith(f"temp_{year}_")]

    for temp_file in temp_files:
        temp_path = os.path.join(output_dir, temp_file)
        temp_df = pd.read_parquet(temp_path)
        all_data.append(temp_df)
        os.remove(temp_path)  # Clean up temp file

    # Combine and save final file
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        output_path = os.path.join(output_dir, f"price_paid_{year}.parquet")
        final_df.to_parquet(output_path, index=False)
        # print(f"Saved {len(final_df)} records for year {year}")
        del final_df


def preprocess_and_save_by_year(
    file_path, postcode_district_lookup, output_dir="processed_price_paid"
):
    """
    Process CSV in chunks, filter by year during reading, and save by year
    """

    # making loading more efficient by being explicit about type, for the large file this should make it faster
    dtype_dict = {
        "paon": "string",
        "street": "string",
        "locality": "string",
        "postcode": "string",
        "price": "int32",
        "date_of_transfer": "string",
        "district": "string",
        "property_type": CategoricalDtype(categories=["D", "S", "T", "F", "O"]),
        "ppd_category_type": CategoricalDtype(categories=["A", "B"]),
    }

    price_paid_headers = [
        "transaction_unique_identifier",
        "price",
        "date_of_transfer",
        "postcode",
        "property_type",
        "old_new",
        "duration",
        "paon",
        "saon",
        "street",
        "locality",
        "town",
        "district",
        "county",
        "ppd_category_type",
        "record_status",
    ]

    columns_to_keep = [
        "paon",
        "street",
        "locality",
        "postcode",
        "price",
        "date_of_transfer",
        "district",
        "property_type",
        "ppd_category_type",
    ]

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Dictionary to accumulate data by year
    yearly_accumulators = {}
    chunk_size = 500000

    print("Counting total rows for progress tracking...")
    with open(file_path, "rb") as f:
        total_rows = sum(1 for _ in f)

    processed_rows = 0

    print("Processing CSV in chunks and filtering by year...")

    # Read the single mega CSV file
    with tqdm(
        total=total_rows, desc="Processing rows", unit="rows", unit_scale=True
    ) as pbar:
        for chunk in pd.read_csv(
            file_path,
            names=price_paid_headers,
            dtype=dtype_dict,
            usecols=columns_to_keep,
            chunksize=chunk_size,
        ):
            # Extract year from date in chunk
            chunk["year"] = pd.to_datetime(
                chunk["date_of_transfer"], errors="coerce"
            ).dt.year

            # Remove rows with invalid dates
            chunk = chunk.dropna(subset=["year"])
            chunk["year"] = chunk["year"].astype(int)

            # Process the chunk
            processed_chunk = process_single_chunk(chunk, postcode_district_lookup)

            if processed_chunk is not None and len(processed_chunk) > 0:
                # Group by year and accumulate
                for year, year_data in processed_chunk.groupby("year"):
                    if year not in yearly_accumulators:
                        yearly_accumulators[year] = []

                    yearly_accumulators[year].append(year_data.drop("year", axis=1))

                    # Save intermediate files if accumulator gets too large (optional memory management)
                    if len(yearly_accumulators[year]) >= 10:  # Every 10 chunks per year
                        temp_df = pd.concat(
                            yearly_accumulators[year], ignore_index=True
                        )
                        temp_file = os.path.join(
                            output_dir,
                            f"temp_{year}_{len(os.listdir(output_dir))}.parquet",
                        )
                        temp_df.to_parquet(temp_file, index=False)
                        yearly_accumulators[year] = []  # Reset accumulator
                        del temp_df

            # Update progress
            chunk_rows = len(chunk)
            processed_rows += chunk_rows
            pbar.update(chunk_rows)

            del chunk, processed_chunk

    # Final save: combine any remaining data and temp files
    print("Combining and finalizing yearly files...")
    for year in tqdm(yearly_accumulators.keys()):
        combine_temp_files_for_year(year, output_dir, yearly_accumulators[year])

    print(f"Preprocessing complete. Data saved to {output_dir}")


def check_and_preprocess_if_needed(
    raw_data_path, postcode_district_lookup, processed_dir="data/processed_price_paid"
):
    """
    Check if processed data exists, if not run preprocessing
    """
    if not os.path.exists(processed_dir) or len(os.listdir(processed_dir)) == 0:
        print("Processed data not found. Starting preprocessing...")
        preprocess_and_save_by_year(
            raw_data_path, postcode_district_lookup, processed_dir
        )
    else:
        print("Processed data found. Skipping preprocessing.")


def load_years_data(years, processed_dir="data/processed_price_paid"):
    """
    Load specific years of preprocessed data
    """
    if not isinstance(years, list):
        years = [years]

    dfs = []
    for year in years:
        file_path = os.path.join(processed_dir, f"price_paid_{year}.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            dfs.append(df)
            print(f"Loaded {len(df)} records for year {year}")
        else:
            print(f"Warning: No data found for year {year}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def load_and_process_pricepaid_data(
    file_path, processed_dir, postcode_district_lookup, years_needed
):
    """
    New main function that uses the preprocessing approach
    """
    # Check and preprocess if needed
    check_and_preprocess_if_needed(file_path, postcode_district_lookup, processed_dir)

    # Load only the years needed
    return load_years_data(years_needed, processed_dir)


def get_date_from_ocod_filename(filename):
    """Extract year and month from OCOD filename"""
    match = re.search(r"OCOD_FULL_(\d{4})_(\d{2})", filename)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        return year, month
    return None, None


def get_rolling_date_range(year, month, years_back=3):
    """Get the date range for rolling 3 years"""
    end_date = datetime(year, month, 1)
    start_date = datetime(year - years_back, month, 1)
    return start_date, end_date


def load_and_filter_price_data(price_folder, start_date, end_date):
    """Load and filter price paid data for a specified date range.

    Loads price paid data from parquet files for all years within the
    specified date range and filters the data to only include records
    within that range.

    Parameters
    ----------
    price_folder : pathlib.Path
        Path to the folder containing price paid parquet files.
        Files should be named 'price_paid_{year}.parquet'.
    start_date : datetime
        The start date for filtering the data (inclusive).
    end_date : datetime
        The end date for filtering the data (inclusive).

    Returns
    -------
    pandas.DataFrame
        A concatenated DataFrame containing all filtered price data
        within the specified date range. Returns an empty DataFrame
        if no data is found or loaded.

    Notes
    -----
    - The function expects parquet files to contain a 'date_of_transfer' column
    - Missing files are logged but do not cause the function to fail
    - The 'date_of_transfer' column is automatically converted to datetime format
    - Warnings are printed for files missing the expected date column

    Examples
    --------
    >>> from pathlib import Path
    >>> from datetime import datetime
    >>> folder = Path('/path/to/price/data')
    >>> start = datetime(2020, 1, 1)
    >>> end = datetime(2023, 12, 31)
    >>> df = load_and_filter_price_data(folder, start, end)
    """

    required_years = list(range(start_date.year, end_date.year + 1))

    dataframes = []

    for year in required_years:
        price_file = price_folder / f"price_paid_{year}.parquet"

        if price_file.exists():
            print(f"Loading {price_file.name}")
            df = pd.read_parquet(price_file)

            # Ensure date column exists and is datetime
            if "date_of_transfer" in df.columns:
                df["date_of_transfer"] = pd.to_datetime(df["date_of_transfer"])

                # Filter by date range
                mask = (df["date_of_transfer"] >= start_date) & (
                    df["date_of_transfer"] <= end_date
                )
                df_filtered = df[mask]

                if not df_filtered.empty:
                    dataframes.append(df_filtered)
            else:
                print(
                    f"Warning: 'date_of_transfer' column not found in {price_file.name}"
                )
        else:
            print(f"File not found: {price_file.name}")

    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        return pd.DataFrame()


def calculate_average_prices(df):
    """Calculate average prices by msoa11cd"""
    if df.empty:
        return pd.DataFrame()

    # Filter data according to your criteria
    filtered_df = df.loc[
        df["property_type"].isin(["D", "S", "F", "T"])
        & (df["ppd_category_type"] == "A"),
        ["msoa11cd", "price"],
    ]

    if filtered_df.empty:
        return pd.DataFrame()

    # Group by msoa11cd and calculate mean and median
    average_prices = filtered_df.groupby("msoa11cd")["price"].agg(["mean", "median"])

    # Flatten column names
    average_prices.columns = ["price_mean", "price_median"]
    average_prices = average_prices.reset_index()

    return average_prices
