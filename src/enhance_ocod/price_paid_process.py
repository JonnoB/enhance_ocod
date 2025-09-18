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
from enhance_ocod.labelling.weak_labelling import (process_dataframe_batch, 
remove_overlapping_spans, remove_zero_length_spans, convert_weakly_labelled_list_to_dataframe )
from enhance_ocod.labelling.ner_spans import lfs
from pandas.api.types import CategoricalDtype
import re
from datetime import datetime

def process_single_chunk(df, postcode_district_lookup):
    """Clean and standardize a chunk of Price Paid Data.

    Performs data cleaning operations on a DataFrame chunk including string
    normalization, postcode processing, and merging with geographical lookup
    data. Modifies data in-place where possible for memory efficiency.

    Args:
        df (pandas.DataFrame): DataFrame chunk containing raw Price Paid Data
            with columns including 'street', 'locality', 'paon', and 'postcode'.
        postcode_district_lookup (pandas.DataFrame): Lookup table containing
            postcode to geographical area mappings (LSOA, MSOA, etc.) with
            'postcode2' as the merge key.

    Returns:
        pandas.DataFrame: Cleaned DataFrame with standardized text fields,
            processed postcodes, and merged geographical data. Original
            'street' and 'postcode' columns are removed and replaced with
            'street_name2' and 'postcode2'.

    Note:
        - Performs in-place modifications to conserve memory
        - Street names are lowercased and cleaned of apostrophes and spaces
        - Postcodes are normalized to lowercase without spaces
        - Uses left join for geographical data merge
        - Depends on external `clean_street_numbers()` function
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
    """Combine temporary files and remaining data for a specific year.

    Helper function for `preprocess_and_save_by_year` that consolidates all
    temporary parquet files and any remaining in-memory data for a given year
    into a single final parquet file. Cleans up temporary files after
    processing to free disk space.

    Args:
        year (int): The year for which to combine data files.
        output_dir (str): Directory path containing temporary files and where
            the final combined file will be saved.
        remaining_data (list of pandas.DataFrame): List of DataFrames
            containing any remaining data in memory that hasn't been written
            to temporary files yet.

    Returns:
        None: Creates a single parquet file named "price_paid_{year}.parquet"
            in the output directory.

    Raises:
        OSError: If unable to read temporary files or write final output file.
        PermissionError: If unable to delete temporary files after processing.

    Note:
        - Automatically removes temporary files after successful combination
        - Handles empty data gracefully (no output file created if no data)
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
    """Process Land Registry Price Paid Data CSV and save by year.

    Processes a large CSV file containing complete Land Registry Price Paid Data
    in memory-efficient chunks. Filters and organizes data by year, adds LSOA
    geographical data, and saves as compressed parquet files to reduce storage
    requirements and improve loading performance.

    Args:
        file_path (str): Path to the input CSV file containing Price Paid Data.
        postcode_district_lookup (dict): Dictionary mapping postcodes to LSOA
            and other geographical data for government geography matching.
        output_dir (str, optional): Directory path where processed parquet files
            will be saved. Defaults to "processed_price_paid".

    Returns:
        None: Files are saved to disk with naming convention
            "price_paid_{year}.parquet".

    Raises:
        FileNotFoundError: If the input file_path does not exist.
        PermissionError: If unable to create output directory or write files.

    Note:
        - Processes data in chunks to prevent memory overflow on large datasets
        - Drops unnecessary columns to optimize storage and performance  
        - Uses categorical data types for efficiency
        - Creates intermediate temporary files for memory management
        - Requires external dependencies: pandas, tqdm, pathlib
    """

    # making loading more efficient by being explicit about type, for the large file this should make it faster
    dtype_dict = {
        "saon":"string",
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
        "saon",
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


def check_and_preprocess_price_paid_data(
    raw_data_path, postcode_district_lookup, processed_dir="data/processed_price_paid"
):
    """Check if processed price paid data exists, and preprocess if needed.

    This function checks whether processed price paid data already exists in the
    specified directory. If the processed data is not found or the directory is
    empty, it initiates the preprocessing pipeline. Otherwise, it skips the
    preprocessing step to avoid redundant processing.

    Parameters
    ----------
    raw_data_path : str or Path
        Path to the raw price paid data file that needs to be processed.
    postcode_district_lookup : dict or pandas.DataFrame
        Lookup table/mapping for postcode districts used during preprocessing.
    processed_dir : str or Path, optional
        Directory path where processed data should be stored or checked for
        existence. Default is "data/processed_price_paid".

    Returns
    -------
    None
        This function does not return any value. It either triggers preprocessing
        or confirms that processed data already exists.

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
    Loads the price paid data for the selected number of years. If no pre-processed data is available, it will pre-process then laod
    """
    # Check and preprocess if needed
    check_and_preprocess_price_paid_data(file_path, postcode_district_lookup, processed_dir)

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


def process_building_addresses(df):
    """
    Helper function to process building addresses using the new approach.
    
    Args:
        df (pd.DataFrame): Input dataframe with address components
        
    Returns:
        pd.DataFrame: Processed dataframe with text and geographic codes
    """
    # Create address column
    df['address'] = df['saon'].str.cat([
        df['paon'], 
        df['street_name2'], 
        df['locality'], 
        df['postcode2']
    ], sep=', ', na_rep='')

    df['address'] = df['address'].str.replace(r'^, ', '', regex=True)  # Remove leading ", "
    df['address'] = df['address'].str.replace(r', , ', ', ', regex=True)  # Fix double commas

    # Process with labelling functions
    weakly_labelled_dict = process_dataframe_batch(
        df,
        batch_size=10000,
        text_column="address",
        include_function_name=False,
        save_intermediate=False,
        verbose=False,
        functions=lfs[0:7]
    )

    # Clean up spans
    remove_overlapping_spans(weakly_labelled_dict)
    remove_zero_length_spans(weakly_labelled_dict)

    # Convert to dataframe
    processed_df = convert_weakly_labelled_list_to_dataframe(weakly_labelled_dict)

    # Merge with location data and clean up
    building_df = processed_df[['datapoint_id', 'text']].merge(
        df[['oa11cd', 'lsoa11cd', 'msoa11cd', 'lad11cd']], 
        left_on='datapoint_id', 
        right_index=True
    ).drop(columns='datapoint_id').dropna().drop_duplicates()
    
    return building_df

def gazetteer_generator(price_paid_folder='../data/processed_price_paid'):
    """
    Create gazetteers that match building names, districts, and streets to LSOA and associated geography.

    This function processes Land Registry price paid data to create gazetteers
    that map building names, districts, and streets to their corresponding geographic codes (OA, LSOA,
    MSOA, LAD). It is primarily used to geolocate new developments that often
    lack postcodes, enabling pricing statistics to be inferred for these
    properties.

    Parameters
    ----------
    price_paid_folder : str, optional
        Path to the folder containing processed price paid parquet files.
        Default is '../data/processed_price_paid'.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing three DataFrames:
        - building_gazetteer: DataFrame with building names and geographic codes
        - district_gazetteer: DataFrame with districts and LAD codes
        - street_gazetteer: DataFrame with streets and geographic codes
        
        building_gazetteer columns:
        - building_name : str, building name extracted using NLP approach
        - oa11cd : str, Output Area code (corresponding to the selected LSOA)
        - lsoa11cd : str, Lower Super Output Area code (most common for this building/LAD)
        - msoa11cd : str, Middle Super Output Area code
        - lad11cd : str, Local Authority District code
        - fraction : float, fraction this LSOA makes up for this building in this LAD
        
        district_gazetteer columns:
        - district : str, district name
        - lad11cd : str, Local Authority District code
        
        street_gazetteer columns:
        - street_name2 : str, street name
        - lsoa11cd : str, Lower Super Output Area code (most common for this street/LAD)
        - oa11cd : str, Output Area code (corresponding to the selected LSOA)
        - msoa11cd : str, Middle Super Output Area code
        - lad11cd : str, Local Authority District code
        - fraction : float, fraction this LSOA makes up for this street in this LAD

    Notes
    -----
    - Uses the standardized government Price Paid dataset wich has consistent column structure
    - Files are processed individually to prevent memory issues
    - Process is slow due to performing a REGEX based Named Entity Recognition to detect building names
    """
    
    folder = Path(price_paid_folder)
    parquet_files = list(folder.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {price_paid_folder}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Required columns for standardised government dataset
    required_cols = ['district', 'lad11cd', 'locality', 'lsoa11cd', 'msoa11cd', 
                    'oa11cd', 'paon', 'postcode2', 'saon', 'street_name2']
    
    # Lists to store data for each gazetteer type
    building_data = []
    district_data = []
    street_data = []
    
    for file in tqdm(parquet_files, desc="Processing parquet files"):
        try:
            df = pd.read_parquet(file)
            
            # Check if all required columns are present
            if all(col in df.columns for col in required_cols):
                # Process building addresses
                building_df = process_building_addresses(df)
                if not building_df.empty:
                    building_data.append(building_df)
                
                # Extract district data
                district_df = df[['district', 'lad11cd']].dropna()
                if not district_df.empty:
                    district_data.append(district_df)
                
                # Extract street data
                street_df = df[['street_name2', 'oa11cd', 'lsoa11cd', 'msoa11cd', 'lad11cd']].dropna()
                if not street_df.empty:
                    street_data.append(street_df)
            else:
                missing_cols = [col for col in required_cols if col not in df.columns]
                print(f"Skipping {file}: missing columns {missing_cols}")
                
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    # Create building gazetteer
    if building_data:
        building_combined = pd.concat(building_data, ignore_index=True).drop_duplicates()
        
        # CONVERT TO LOWERCASE FIRST, BEFORE GROUPBY
        building_combined['text'] = building_combined['text'].str.lower()
        
        # Get counts for each building-lad-lsoa combination
        building_counts = (building_combined.groupby(['text', 'lad11cd', 'lsoa11cd', 'oa11cd', 'msoa11cd'], observed=True)
                        .size()
                        .reset_index(name='counts'))
        
        # Calculate total counts per building-lad combination for fraction calculation
        building_totals = (building_counts.groupby(['text', 'lad11cd'], observed=True)['counts']
                        .sum()
                        .reset_index()
                        .rename(columns={'counts': 'total_counts'}))
        
        # Merge to get total counts and calculate fraction
        building_counts = building_counts.merge(building_totals, on=['text', 'lad11cd'])
        building_counts['fraction'] = building_counts['counts'] / building_counts['total_counts']
        
        # For each building-lad combination, keep the LSOA with the highest count
        building_gazetteer = (building_counts.sort_values('counts', ascending=False)
                            .groupby(['text', 'lad11cd'], observed=True)
                            .first()
                            .reset_index()[['text', 'oa11cd', 'lsoa11cd', 'msoa11cd', 'lad11cd', 'fraction']])
        
        building_gazetteer = building_gazetteer.loc[building_gazetteer['text']!='<na>']
        
        # Remove repetitive names with low uniqueness
        building_gazetteer = building_gazetteer[building_gazetteer['fraction']>0.2]
        building_gazetteer.rename(columns={'text':'building_name'}, inplace=True)
    else:
        building_gazetteer = pd.DataFrame()
    
    # Create district gazetteer
    if district_data:
        district_combined = pd.concat(district_data, ignore_index=True)
        district_counts = (district_combined.groupby(['district', 'lad11cd'], observed=True)
                          .size()
                          .reset_index(name='counts'))
        district_gazetteer = (district_counts.sort_values('counts', ascending=False)
                             .groupby('district')
                             .first()
                             .reset_index()[['district', 'lad11cd']])
    else:
        district_gazetteer = pd.DataFrame()
    
    # Create street gazetteer
    if street_data:
        street_combined = pd.concat(street_data, ignore_index=True)
        street_counts = (street_combined.groupby(['street_name2', 'lad11cd', 'lsoa11cd', 'oa11cd', 'msoa11cd'], observed=True)
                        .size()
                        .reset_index(name='counts'))
        
        street_totals = (street_counts.groupby(['street_name2', 'lad11cd'], observed=True)['counts']
                        .sum()
                        .reset_index()
                        .rename(columns={'counts': 'total_counts'}))
        
        street_counts = street_counts.merge(street_totals, on=['street_name2', 'lad11cd'])
        street_counts['fraction'] = street_counts['counts'] / street_counts['total_counts']
        
        street_gazetteer = (street_counts.sort_values('counts', ascending=False)
                           .groupby(['street_name2', 'lad11cd'], observed=True)
                           .first()
                           .reset_index()[['street_name2', 'lsoa11cd', 'oa11cd', 'msoa11cd', 'lad11cd', 'fraction']])
        
        street_gazetteer = street_gazetteer.loc[street_gazetteer['street_name2']!='<na>']
    else:
        street_gazetteer = pd.DataFrame()
    
    return building_gazetteer, district_gazetteer, street_gazetteer