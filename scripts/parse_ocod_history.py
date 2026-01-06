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
    python parse_ocod_history.py
    python parse_ocod_history.py --input-dir /path/to/input --output-dir /path/to/output
    python parse_ocod_history.py --model-path /path/to/local/model
    python parse_ocod_history.py --model-path Jonnob/OCOD_NER

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
    add_geographic_metadata,
    enhance_ocod_with_gazetteers,
    add_business_matches,
    property_class,
    get_default_property_rules,
    fill_unknown_classes_by_group,
    drop_non_residential_duplicates
    )

from enhance_ocod.price_paid_process import check_and_preprocess_price_paid_data, gazetteer_generator
from pathlib import Path
from tqdm import tqdm
import time
import gc  # Add for memory management
import argparse

import pickle
import pandas as pd


import torch

# There is a warning related to bfill and ffill which is basically internal to pandas so silencing here
import warnings

warnings.filterwarnings("ignore", message=".*Downcasting object dtype arrays.*")

torch.set_float32_matmul_precision("medium")

SCRIPT_DIR = Path(__file__).parent.absolute()

# ====== PARSE COMMAND LINE ARGUMENTS ======
def parse_arguments():
    """Parse command line arguments for input/output directories and model path."""
    parser = argparse.ArgumentParser(
        description="Process OCOD historical dataset with optional custom paths."
    )

    # Default paths
    default_input = SCRIPT_DIR.parent / "data" / "ocod_history"
    default_output = SCRIPT_DIR.parent / "data" / "ocod_history_processed"
    default_model = "Jonnob/OCOD_NER"

    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(default_input),
        help=f"Input directory containing OCOD_FULL_*.zip files (default: {default_input})"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output),
        help=f"Output directory for processed parquet files (default: {default_output})"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=default_model,
        help=f"Model path (HuggingFace model ID or local path) (default: {default_model})"
    )

    args = parser.parse_args()

    return Path(args.input_dir), Path(args.output_dir), args.model_path

# ====== CONSTANT PATHS AND SETTINGS ======
input_dir, output_dir, model_path = parse_arguments()

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
    
    # Run the function, it will take about 1.5 hours due to the time needed to create the building gazetteer.
    # This ibviously needs some form of improvement!
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

processed_count = 0
skipped_count = 0

for zip_file in tqdm(all_files, desc="Processing OCOD files"):
    out_name = zip_file.stem + ".parquet"
    out_path = output_dir / out_name

    # Define parsed results file path
    parsed_results_file = parsed_results_dir / f"{zip_file.stem}_parsed_results.pkl"

    if out_path.exists():
        skipped_count += 1
        continue
    
    processed_count += 1

    # Load and process the OCOD data
    ocod_data = load_and_prep_OCOD_data(str(zip_file))

    ###############
    # Perform NER on addresses
    ###############
    if parsed_results_file.exists():
        with open(parsed_results_file, "rb") as f:
            results = pickle.load(f)
    else:
        start_time = time.time()

        results = parse_addresses_pipeline(
            df=ocod_data,
            short_batch_size=128,  # The default seems really slow, might be to do with loading not sure
            model_path=str(model_path),
            target_column="property_address",
        )

        end_time = time.time()

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
    # just to minimise memory overhead
    del ocod_data

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

    #This created a temptorary ID before the real expansion phase, it allows
    # titles spread over multiple rows to have the same class.
    # It will be overwritten later
    with_matches = create_unique_id(with_matches)

    #################
    #
    # Classify and expand the rows
    #
    ################   

    rules = get_default_property_rules()
    classified  = property_class(with_matches.copy(), rules, include_rule_name=True)
    # fill the classes that are unknown but part of a larger title_number group 
    classified = fill_unknown_classes_by_group(classified)
    # Only residential properties should be expanded so drop all other multiple title_numbers
    classified = drop_non_residential_duplicates(classified, class_col='class')
    # Expand the residential class
    # large_expansion_threshold can be adjusted based on data analysis
    ocod_data = expand_dataframe_numbers(classified, class_var = 'class', print_every=10000, min_count=1, large_expansion_threshold=100)
    # Update the unique id
    ocod_data = create_unique_id(ocod_data)

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
        "matched_rule",
        "is_multi",
        "expansion_size",
        "large_expansion",
    ]


    ocod_data = ocod_data.loc[:, columns]
    # Save results
    ocod_data.to_parquet(out_path)

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

print(f"Processing complete: {processed_count} files processed, {skipped_count} files skipped.")
