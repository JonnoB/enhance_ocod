"""
OCOD Dataset Enhancement and Processing Pipeline

This script processes the OCOD (Overseas Companies that Own Property in England and Wales)
historical dataset files through a comprehensive enhancement pipeline that includes address
parsing, geolocation, and property classification.

About OCOD Dataset:
The OCOD dataset contains information about properties in England and Wales that are owned
by overseas companies. This data is published by HM Land Registry and includes company
details, property addresses, and ownership information.

Processing Pipeline:
1. **Address Parsing**: Uses a trained NLP model to parse and standardize property addresses
   from the raw address strings, extracting structured components like building numbers,
   street names, localities, and postcodes.

2. **Geolocation**: Enhances addresses with geographic information by:
   - Matching postcodes to administrative boundaries using ONSPD data
   - Adding Local Authority District (LAD) codes and other geographic identifiers
   - Cross-referencing with price paid data for additional location context

3. **Address Matching**: Performs sophisticated address matching against:
   - HM Land Registry Price Paid data for property transaction history
   - VOA (Valuation Office Agency) rating list for business classification
   - Street-level and sub-street level matching algorithms

4. **Property Classification**: Classifies properties based on:
   - Business activity data from VOA rating lists
   - Statistical analysis of business density per Output Area (OA) and Lower Super Output Area (LSOA)
   - Property type and usage patterns

Input Files:
- OCOD_FULL_*.zip: Historical OCOD dataset files from HM Land Registry
- ONSPD_FEB_2025.zip: Ordnance Survey National Statistics Postcode Directory
- price_paid_complete_may_2025.csv: HM Land Registry Price Paid dataset
- 2023_non_domestic_rating_list_entries.zip: VOA non-domestic rating list

Output:
- Processed OCOD data saved as Parquet files with enhanced address information,
  geographic identifiers, and property classifications

Usage:
Run this script to process all OCOD historical files in the input directory.
The script will skip files that have already been processed (existing output files).

Example:
    python ocod_processing_pipeline.py

Note:
This is a computationally intensive process that requires significant memory and processing
time. The script includes memory management strategies and progress tracking.
"""

from enhance_ocod.inference import parse_addresses_pipeline, convert_to_entity_dataframe
from enhance_ocod.address_parsing import (
    load_and_prep_OCOD_data,
    parsing_and_expansion_process,
    post_process_expanded_data,
    load_postcode_district_lookup,
)
from enhance_ocod.locate_and_classify import (
    preprocess_expandaded_ocod_data,
    add_missing_lads_ocod,
    load_voa_ratinglist,
    street_and_building_matching,
    substreet_matching,
    counts_of_businesses_per_oa_lsoa,
    voa_address_match_all_data,
    classification_type1,
    classification_type2,
    contract_ocod_after_classification,
)
from enhance_ocod.price_paid_process import load_and_process_pricepaid_data
from pathlib import Path
from tqdm import tqdm
import time
import gc  # Add for memory management

import pickle


import torch

# There is a warning related to bfill and ffill which is basically internal to pandas so silencing here
import warnings

warnings.filterwarnings("ignore", message=".*Downcasting object dtype arrays.*")

torch.set_float32_matmul_precision("medium")

SCRIPT_DIR = Path(__file__).parent.absolute()

# ====== CONSTANT PATHS AND SETTINGS ======
input_dir = SCRIPT_DIR.parent / "data" / "ocod_history"
output_dir = SCRIPT_DIR.parent / "data" / "ocod_history_processed"
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

# List of all zip files in input_dir
#
# TESTING!!! only 10 files!
#
all_files = sorted([f for f in input_dir.glob("OCOD_FULL_*.zip")])


# test_indices = [0, 25, 50, 75]
# all_files = [all_files[i] for i in test_indices if i < len(all_files)]
print(f"Found {len(all_files)} OCOD history files.")

# Load common data once (if these don't change between files)
print("Loading common reference data...")
postcode_district_lookup = load_postcode_district_lookup(str(ONSPD_path))
voa_businesses = load_voa_ratinglist(str(voa_path), postcode_district_lookup)

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
    # Parse addresses
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

    # Continue with post-parsing processing
    test = convert_to_entity_dataframe(results)
    test = parsing_and_expansion_process(all_entities=test)
    ocod_data = post_process_expanded_data(test, ocod_data)

    # Clean up
    del results, test
    gc.collect()

    ###############
    # Geolocate
    ###############
    print(f"Geolocating {zip_file.name}...")

    ocod_data = preprocess_expandaded_ocod_data(ocod_data, postcode_district_lookup)

    price_paid_df = load_and_process_pricepaid_data(
        file_path=str(price_paid_path),
        processed_dir=processed_price_paid_dir,
        postcode_district_lookup=postcode_district_lookup,
        years_needed=[2024, 2023, 2022],
    )

    ocod_data = add_missing_lads_ocod(ocod_data, price_paid_df)
    ocod_data = street_and_building_matching(ocod_data, price_paid_df, voa_businesses)
    ocod_data = substreet_matching(ocod_data, price_paid_df, voa_businesses)

    # Clean up price paid data
    del price_paid_df
    gc.collect()

    ###########
    # Classify
    ###########
    print(f"Classifying {zip_file.name}...")
    ocod_data = counts_of_businesses_per_oa_lsoa(ocod_data, voa_businesses)
    ocod_data = voa_address_match_all_data(ocod_data, voa_businesses)

    ocod_data = classification_type1(ocod_data)
    ocod_data = classification_type2(ocod_data)

    ocod_data = contract_ocod_after_classification(
        ocod_data, class_type="class2", classes=["residential"]
    )

    columns = [
        "title_number",
        "within_title_id",
        "within_larger_title",
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
        "class2",
    ]

    ocod_data = ocod_data.loc[:, columns].rename(
        columns={"within_title_id": "nested_id", "within_larger_title": "nested_title"}
    )
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
