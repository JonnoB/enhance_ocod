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
import numpy as np
from locate_and_classify_helper_functions import clean_street_numbers


def process_single_chunk(df, postcode_district_lookup):
    """
    Modified version that includes price in the output
    """
    # Process strings inplace to save memory
    df['street'] = df['street'].astype(str).str.lower()
    df['street_name2'] = (df['street']
                         .str.replace(r"'", "", regex=True)
                         .str.replace(r"s(s)?(?=\s)", "", regex=True) 
                         .str.replace(r"\s", "", regex=True))
    
    df.drop('street', axis=1, inplace=True)
    
    df['locality'] = df['locality'].astype(str).str.lower()
    df['paon'] = df['paon'].astype(str).str.lower()
    
    # Clean street numbers
    df = clean_street_numbers(df, original_column='paon')
    
    # Process postcode
    df['postcode2'] = df['postcode'].astype(str).str.lower().str.replace(r"\s", "", regex=True)
    df.drop('postcode', axis=1, inplace=True)
    
    # Merge with lookup
    df = df.merge(postcode_district_lookup, how='left', left_on="postcode2", right_on="postcode2")
    
    # Keep necessary columns including price
    final_columns = ['street_name2', 'street_number', 'postcode2', 'district', 
                    'paon', 'lad11cd', 'oa11cd', 'lsoa11cd', 'msoa11cd', 'price', 'year']
    
    available_columns = [col for col in final_columns if col in df.columns]
    
    return df[available_columns]


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
        #print(f"Saved {len(final_df)} records for year {year}")
        del final_df

def preprocess_and_save_by_year(file_path, postcode_district_lookup, output_dir="processed_price_paid"):
    """
    Process CSV in chunks, filter by year during reading, and save by year
    """
    dtype_dict = {
        'paon': 'string',
        'street': 'string', 
        'locality': 'string',
        'postcode': 'string',
        'price': 'int32',
        'date_of_transfer': 'string'
    }
    
    price_paid_headers = ['transaction_unique_identifier', 'price', 'date_of_transfer', 'postcode', 'property_type', 
                         'old_new', 'duration', 'paon', 'saon', 'street', 'locality', 'town', 'district', 'county',
                         'ppd_category_type', 'record_status']
    
    columns_to_keep = ['paon', 'street', 'locality', 'postcode', 'price', 'date_of_transfer', 'district']
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Dictionary to accumulate data by year
    yearly_accumulators = {}
    chunk_size = 500000
    
    print("Processing CSV in chunks and filtering by year...")
    
    # Read the single mega CSV file
    for chunk in tqdm(pd.read_csv(
        file_path,  # Now this is the single CSV file path
        names=price_paid_headers,
        dtype=dtype_dict,
        usecols=columns_to_keep,
        chunksize=chunk_size
    )):
        # Extract year from date in chunk
        chunk['year'] = pd.to_datetime(chunk['date_of_transfer'], errors='coerce').dt.year
        
        # Remove rows with invalid dates
        chunk = chunk.dropna(subset=['year'])
        chunk['year'] = chunk['year'].astype(int)
        
        # Process the chunk
        processed_chunk = process_single_chunk(chunk, postcode_district_lookup)
        
        if processed_chunk is not None and len(processed_chunk) > 0:
            # Group by year and accumulate
            for year, year_data in processed_chunk.groupby('year'):
                if year not in yearly_accumulators:
                    yearly_accumulators[year] = []
                
                yearly_accumulators[year].append(year_data.drop('year', axis=1))
                
                # Save intermediate files if accumulator gets too large (optional memory management)
                if len(yearly_accumulators[year]) >= 10:  # Every 10 chunks per year
                    temp_df = pd.concat(yearly_accumulators[year], ignore_index=True)
                    temp_file = os.path.join(output_dir, f"temp_{year}_{len(os.listdir(output_dir))}.parquet")
                    temp_df.to_parquet(temp_file, index=False)
                    yearly_accumulators[year] = []  # Reset accumulator
                    del temp_df
        
        del chunk, processed_chunk
    
    # Final save: combine any remaining data and temp files
    print("Combining and finalizing yearly files...")
    for year in tqdm(yearly_accumulators.keys()):
        combine_temp_files_for_year(year, output_dir, yearly_accumulators[year])
    
    print(f"Preprocessing complete. Data saved to {output_dir}")

def check_and_preprocess_if_needed(raw_data_path, postcode_district_lookup, processed_dir="data/processed_price_paid"):
    """
    Check if processed data exists, if not run preprocessing
    """
    if not os.path.exists(processed_dir) or len(os.listdir(processed_dir)) == 0:
        print("Processed data not found. Starting preprocessing...")
        preprocess_and_save_by_year(raw_data_path, postcode_district_lookup, processed_dir)
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


def load_and_process_pricepaid_data(file_path, postcode_district_lookup, years_needed):
    """
    New main function that uses the preprocessing approach
    """
    # Check and preprocess if needed
    check_and_preprocess_if_needed(file_path, postcode_district_lookup)
    
    # Load only the years needed
    return load_years_data(years_needed)