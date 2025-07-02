"""
OCOD Dataset Enhancement and Processing Pipeline - CSV Version

This script processes a single CSV file through the same comprehensive enhancement pipeline 
that includes address parsing, geolocation, and property classification.

The CSV file should contain the same columns as the OCOD dataset:
- property_address: The address to be parsed and enhanced
- Other OCOD columns as needed

Processing Pipeline:
1. **Address Parsing**: Uses a trained NLP model to parse and standardize property addresses
2. **Geolocation**: Enhances addresses with geographic information
3. **Address Matching**: Performs sophisticated address matching
4. **Property Classification**: Classifies properties based on business activity data

Usage:
    python ocod_csv_processing_pipeline.py

The CSV file path is hardcoded in the CSV_FILE_PATH variable below.
"""

from enhance_ocod.inference import parse_addresses_batch, convert_to_entity_dataframe
from enhance_ocod.address_parsing import (
    parsing_and_expansion_process, post_process_expanded_data
)
from enhance_ocod.locate_and_classify import (
    load_postcode_district_lookup, preprocess_expandaded_ocod_data, 
    add_missing_lads_ocod, load_voa_ratinglist, street_and_building_matching, substreet_matching,
    counts_of_businesses_per_oa_lsoa, voa_address_match_all_data, classification_type1, classification_type2,
    contract_ocod_after_classification
)
from enhance_ocod.price_paid_process import load_and_process_pricepaid_data
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

import torch

torch.set_float32_matmul_precision('medium')

SCRIPT_DIR = Path(__file__).parent.absolute()

# ====== HARDCODED CSV FILE PATH ======
# Replace this path with your test CSV file
CSV_FILE_PATH = SCRIPT_DIR.parent / "data" / "training_data" / "gt_addresses.csv"  # <-- CHANGE THIS PATH

# ====== CONSTANT PATHS AND SETTINGS ======
output_dir = SCRIPT_DIR.parent / "data" / "csv_processed"
model_path = SCRIPT_DIR.parent / "models" / "address_parser" / "checkpoint-750"
ONSPD_path = SCRIPT_DIR.parent / "data" / "ONSPD_FEB_2025.zip"
price_paid_path = SCRIPT_DIR.parent / "data" / "price_paid_data" / "price_paid_complete_may_2025.csv"
processed_price_paid_dir = SCRIPT_DIR.parent / "data" / "processed_price_paid"
voa_path = SCRIPT_DIR.parent / "data" / "2023_non_domestic_rating_list_entries.zip"
output_dir.mkdir(parents=True, exist_ok=True)

def load_csv_data(csv_path):
    """
    Load CSV data in the same format expected by the pipeline.
    Modify this function if your CSV has different column names or structure.
    """
    print(f"Loading CSV data from {csv_path}...")
    
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    required_columns = ['property_address']  # Add other required columns as needed
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {missing_columns}")
    
    print(f"Loaded {len(df)} rows from CSV file.")
    return df

def main():
    # Create output filename based on input CSV name
    csv_file = Path(CSV_FILE_PATH)
    out_name = csv_file.stem + "_processed.parquet"
    out_path = output_dir / out_name

    if out_path.exists():
        print(f"Output file already exists: {out_path}")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return

    print(f"Processing {csv_file.name}...")

    # Load CSV data instead of ZIP file
    ocod_data = load_csv_data(CSV_FILE_PATH)

    ###############
    # Parse addresses
    ###############
    print("Parsing addresses...")
    start_time = time.time()

    results = parse_addresses_batch(
        df=ocod_data,
        model_path=str(model_path),
        target_column="property_address",
        batch_size=512,
        use_fp16=True
    )

    end_time = time.time()
    print(f"Address parsing completed in {end_time - start_time:.2f} seconds")

    test = convert_to_entity_dataframe(results)
    test = parsing_and_expansion_process(all_entities=test)
    ocod_data = post_process_expanded_data(test, ocod_data)

    ###############
    # Geolocate
    ###############
    print("Geolocating...")
    postcode_district_lookup = load_postcode_district_lookup(str(ONSPD_path))

    ocod_data = preprocess_expandaded_ocod_data(ocod_data, postcode_district_lookup)

    price_paid_df = load_and_process_pricepaid_data(
        file_path=str(price_paid_path), 
        processed_dir=processed_price_paid_dir,
        postcode_district_lookup=postcode_district_lookup, 
        years_needed=[2017, 2018, 2019]
    )

    ocod_data = add_missing_lads_ocod(ocod_data, price_paid_df)

    voa_businesses = load_voa_ratinglist(str(voa_path), postcode_district_lookup)
    del postcode_district_lookup  # memory management

    ocod_data = street_and_building_matching(ocod_data, price_paid_df, voa_businesses)
    ocod_data = substreet_matching(ocod_data, price_paid_df, voa_businesses)
    del price_paid_df  # memory management

    ###########
    # Classify
    ###########
    print("Classifying...")
    ocod_data = counts_of_businesses_per_oa_lsoa(ocod_data, voa_businesses)
    ocod_data = voa_address_match_all_data(ocod_data, voa_businesses)
    del voa_businesses  # memory management

    ocod_data = classification_type1(ocod_data)
    ocod_data = classification_type2(ocod_data)
    
    ocod_data = contract_ocod_after_classification(ocod_data, class_type='class2', classes=['residential'])

    columns = ['title_number', 'within_title_id', 'within_larger_title', 'unique_id', 
              'unit_id', 'unit_type', 'building_name', 'street_number', 'street_name', 
              'postcode', 'city', 'district', 'region', 'property_address', 'oa11cd', 
              'lsoa11cd', 'msoa11cd', 'lad11cd', 'class', 'class2']

    ocod_data = ocod_data.loc[:, columns].rename(columns={
        'within_title_id': 'nested_id',
        'within_larger_title': 'nested_title'
    })

    # Save the processed data
    ocod_data.to_parquet(out_path)
    print(f"Processing complete! Saved processed data to {out_path}")
    print(f"Processed {len(ocod_data)} records.")

if __name__ == "__main__":
    main()