"""
OCOD Dataset Enhancement and Processing Pipeline.

This script processes the OCOD (Overseas Companies that Own Property in England 
and Wales) historical dataset. The process creates a tabular dataset where each 
row represents a single address and classifies each property as residential, 
business, airspace, parking, land, or other.

The first time this script runs can take significantly longer as support metadata 
is created and NER is performed on the OCOD datasets. However, subsequent runs 
will be much quicker as the pre-processed data such as NER outputs and gazetteers 
are re-used.

With re-use, one OCOD dataset takes less than half a minute to process and uses 
less than 10 GB of memory.

About OCOD Dataset:
    The OCOD dataset contains information about properties in England and Wales 
    that are owned by overseas companies. This data is published by HM Land 
    Registry and includes company details, property addresses, and ownership 
    information.

Input Files:
    - OCOD_FULL_*.zip: Historical OCOD dataset files from HM Land Registry
    - ONSPD_FEB_*.zip: Ordnance Survey National Statistics Postcode Directory
    - price_paid_complete_*.csv: HM Land Registry Price Paid dataset
    - 2023_non_domestic_rating_list_entries.zip: VOA non-domestic rating list

Output:
    Processed OCOD data saved as Parquet files with enhanced address information,
    geographic identifiers, and property classifications.

Usage:
    Run this script to process all OCOD historical files in the input directory.
    The script will skip files that have already been processed (existing output 
    files).

Example:
    python ocod_processing_pipeline.py

Requirements:
    - torch
    - pandas
    - pathlib
    - tqdm
    - pickle

Notes:
    - Requires GPU support for optimal performance
    - Creates intermediate files for caching parsed results
    - Automatically generates gazetteers if not present
"""

from enhance_ocod.inference import parse_addresses_pipeline
from enhance_ocod.address_parsing import (
    load_and_prep_OCOD_data,
    load_postcode_district_lookup,
    process_addresses,
    expand_dataframe_numbers,
    create_unique_id
)
from enhance_ocod.locate_and_classify import (
    load_voa_ratinglist,
    counts_of_businesses_per_oa_lsoa, # Not currently used
    add_geographic_metadata,
    enhance_ocod_with_gazetteers,
    add_business_matches,
    property_class,
    property_class_no_match
    )

from enhance_ocod.price_paid_process import check_and_preprocess_price_paid_data, gazetteer_generator
from pathlib import Path
from tqdm import tqdm
import time
import gc  # Add for memory management

import pickle
import pandas as pd


import torch

# There is a warning related to bfill and ffill which is basically internal to pandas so silencing here
import warnings

warnings.filterwarnings("ignore", message=".*Downcasting object dtype arrays.*")

torch.set_float32_matmul_precision("medium")

SCRIPT_DIR = Path(__file__).parent.absolute()

# ====== CONSTANT PATHS AND SETTINGS ======
input_dir = SCRIPT_DIR.parent / "data" / "ocod_history"
output_dir = SCRIPT_DIR.parent / "data" / "ocod_history_processed_new"
model_path = (
    SCRIPT_DIR.parent / "models" / "address_parser_original_fullset" / "final_model"
)

def get_first_file_in_data_dir(dirname):
    """Get the first file in a data subdirectory, or None if no files exist."""
    data_dir = SCRIPT_DIR.parent / "data" / dirname
    files = list(data_dir.glob("*"))
    return files[0] if files else None

# Usage
ONSPD_path = get_first_file_in_data_dir("onspd")
price_paid_path = get_first_file_in_data_dir("price_paid_data")
voa_path = get_first_file_in_data_dir("voa")


processed_price_paid_dir = SCRIPT_DIR.parent / "data" / "processed_price_paid"

output_dir.mkdir(parents=True, exist_ok=True)

parsed_results_dir = SCRIPT_DIR.parent / "data" / "parsed_ocod_dicts"
parsed_results_dir.mkdir(parents=True, exist_ok=True)

print("Loading common reference data...")
postcode_district_lookup = load_postcode_district_lookup(str(ONSPD_path))
voa_businesses = load_voa_ratinglist(str(voa_path), postcode_district_lookup)

check_and_preprocess_price_paid_data(str(price_paid_path), postcode_district_lookup, str(processed_price_paid_dir))


##########################
##
## Load and sort out the gazetteers
##
############################

# Define file paths
gazetteer_dir = SCRIPT_DIR.parent / 'data'/ 'gazetteer' 
building_file = gazetteer_dir / 'building_gazetteer.parquet'
district_file = gazetteer_dir / 'district_gazetteer.parquet'
street_file = gazetteer_dir / 'street_gazetteer.parquet'

# Check if all three files exist
if building_file.exists() and district_file.exists() and street_file.exists():
    print("Loading existing gazetteer files...")
    building_gazetteer = pd.read_parquet(building_file)
    building_gazetteer['fraction'] = 1
    district_gazetteer = pd.read_parquet(district_file)
    street_gazetteer = pd.read_parquet(street_file)
else:
    print("One or more gazetteer files missing. Running gazetteer_generator...")
    # Create directory if it doesn't exist
    gazetteer_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the function
    building_gazetteer, district_gazetteer, street_gazetteer = gazetteer_generator(
        price_paid_folder='../data/processed_price_paid'
    )
    
    # Add the fraction column
    building_gazetteer['fraction'] = 1
    
    # Save the results
    building_gazetteer.to_parquet(building_file)
    district_gazetteer.to_parquet(district_file)
    street_gazetteer.to_parquet(street_file)
    print("Gazetteer files saved successfully!")


# List of all zip files in input_dir
#
#
all_files = sorted([f for f in input_dir.glob("OCOD_FULL_*.zip")])


# test_indices = [0, 25, 50, 75]
# all_files = [all_files[i] for i in test_indices if i < len(all_files)]
print(f"Found {len(all_files)} OCOD history files.")

for zip_file in tqdm(all_files, desc="Processing OCOD files"):
    out_name = zip_file.stem + ".parquet"
    out_path = output_dir / out_name

    # Define parsed results file path
    parsed_results_file = parsed_results_dir / f"{zip_file.stem}_parsed_results.pkl"

    if out_path.exists():
        print(f"Skipping {zip_file.name}: already processed.")
        continue

    print(f"Processing {zip_file.name}...")

    # Load and process the OCOD data
    ocod_data = load_and_prep_OCOD_data(str(zip_file))

    ###############
    # Perform NER on addresses
    ###############
    if parsed_results_file.exists():
        print(f"Loading cached parsing results for {zip_file.name}...")
        with open(parsed_results_file, "rb") as f:
            results = pickle.load(f)
        print(
            f"Loaded cached results with success rate: {results['summary']['success_rate']:.1%}"
        )
    else:
        print(f"Parsing addresses for {zip_file.name}...")
        start_time = time.time()

        results = parse_addresses_pipeline(
            df=ocod_data,
            short_batch_size=128,  # The default seems really slow, might be to do with loading not sure
            model_path=str(model_path),
            target_column="property_address",
        )

        end_time = time.time()
        print(f"Address parsing took {end_time - start_time:.2f} seconds")
        print(f"Success rate: {results['summary']['success_rate']:.1%}")

        # Save parsing results
        print(f"Saving parsing results to {parsed_results_file}...")
        with open(parsed_results_file, "wb") as f:
            pickle.dump(results, f)


    #################v
    #
    # Process the NER dictionaries
    #
    ################
    processed_addresses_df = process_addresses(results['results'])

    post_processed_data = processed_addresses_df.merge(
        ocod_data, how="left", left_on="datapoint_id", right_index=True
    )[
            [
                "title_number",
                "tenure",
                "unit_id",
                "unit_type",
                "number_filter",
                "building_name",
                "street_number",
                "street_name",
                "postcode",
                "city",
                "district",
                "county",
                "region",
                "price_paid",
                "property_address",
                "country_incorporated",
            ]
        ]

    #################v
    #
    # Add geographic information
    #
    ################   

    post_processed_data["postcode"] = post_processed_data["postcode"].str.upper()

    pre_process_ocod = add_geographic_metadata(post_processed_data, postcode_district_lookup)

    # I should probably make a better way of doing this, having these changes here does not seem like a good idea
    pre_process_ocod['building_name'] = pre_process_ocod['building_name'].str.lower()
    pre_process_ocod['street_name2'] = pre_process_ocod['street_name2'].str.lower()
    enhanced  =  enhance_ocod_with_gazetteers(pre_process_ocod, building_gazetteer, district_gazetteer, street_gazetteer)

    with_matches = add_business_matches(enhanced, voa_businesses)

    #################v
    #
    # Classify and expand the rows
    #
    ################   

    classified = property_class(with_matches)

    classified = property_class_no_match(classified)

    expanded_df = expand_dataframe_numbers(classified, class_var = 'class', print_every=10000, min_count=1)

    expanded_df = create_unique_id(expanded_df)

    columns = [
        "title_number",
        "multi_id",
        "unique_id",
        "unit_id",
        "unit_type",
        "building_name",
        "street_number",
        "street_name",
        "postcode",
        "city",
        "district",
        "region",
        "property_address",
        "oa11cd",
        "lsoa11cd",
        "msoa11cd",
        "lad11cd",
        "country_incorporated",
        "class",
        "needs_expansion",
    ]


    ocod_data = expanded_df.loc[:, columns]
    # Save results
    ocod_data.to_parquet(out_path)
    print(f"Saved processed data to {out_path}")

    # Clean up for next iteration
    del ocod_data
    gc.collect()

    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Clean up common data
del postcode_district_lookup, voa_businesses
gc.collect()

print("All files processed.")
