from enhance_ocod.inference_utils import parse_addresses_batch, convert_to_entity_dataframe
from enhance_ocod.address_parsing_helper_functions import (
    load_and_prep_OCOD_data, parsing_and_expansion_process, post_process_expanded_data
)
from enhance_ocod.locate_and_classify_helper_functions import (
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
import pickle

import torch

torch.set_float32_matmul_precision('medium')

SCRIPT_DIR = Path(__file__).parent.absolute()

# ====== CONSTANT PATHS AND SETTINGS ======
input_dir = SCRIPT_DIR.parent / "data" / "ocod_history"
output_dir = SCRIPT_DIR.parent / "data" / "ocod_history_processed"
# Create a directory for cached parsed addresses
parsed_cache_dir = SCRIPT_DIR.parent / "data" / "parsed_addresses_cache"
parsed_cache_dir.mkdir(parents=True, exist_ok=True)

model_path = SCRIPT_DIR.parent / "models" / "address_parser_dev" / "final_model"
ONSPD_path = SCRIPT_DIR.parent / "data" / "ONSPD_FEB_2025.zip" 
price_paid_path = SCRIPT_DIR.parent / "data" / "price_paid_data" / "price_paid_complete_may_2025.csv"
processed_price_paid_dir = SCRIPT_DIR.parent / "data" / "processed_price_paid"
voa_path = SCRIPT_DIR.parent / "data" / "2023_non_domestic_rating_list_entries.zip"
output_dir.mkdir(parents=True, exist_ok=True)

# Dictionary to store timing results
timing_results = {}

# Function timing decorator
def time_function(func_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            timing_results[func_name] = execution_time
            return result
        return wrapper
    return decorator

def get_parsed_cache_path(zip_file):
    """Generate cache file path for parsed addresses"""
    cache_filename = f"{zip_file.stem}_parsed.pkl"
    return parsed_cache_dir / cache_filename

def save_parsed_addresses(ocod_data, cache_path):
    """Save the parsed OCOD data to cache"""
    print(f"Saving parsed addresses to cache: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(ocod_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Parsed addresses cached successfully")

def load_parsed_addresses(cache_path):
    """Load parsed addresses from cache"""
    print(f"Loading parsed addresses from cache: {cache_path}")
    with open(cache_path, 'rb') as f:
        ocod_data = pickle.load(f)
    print(f"✓ Parsed addresses loaded from cache")
    return ocod_data

# Get only the first file for single cycle processing
all_files = sorted([f for f in input_dir.glob("OCOD_FULL_*.zip")])
if not all_files:
    print("No OCOD files found!")
    exit()

zip_file = all_files[0]  # Process only the first file
print(f"Processing single file: {zip_file.name}")

out_name = zip_file.stem + ".parquet"
out_path = output_dir / out_name
cache_path = get_parsed_cache_path(zip_file)

print(f"Processing {zip_file.name}...")

# Wrap functions with timing decorator
@time_function("load_and_prep_OCOD_data")
def timed_load_and_prep_OCOD_data(*args, **kwargs):
    return load_and_prep_OCOD_data(*args, **kwargs)

@time_function("parse_addresses_batch")
def timed_parse_addresses_batch(*args, **kwargs):
    return parse_addresses_batch(*args, **kwargs)

@time_function("convert_to_entity_dataframe")
def timed_convert_to_entity_dataframe(*args, **kwargs):
    return convert_to_entity_dataframe(*args, **kwargs)

@time_function("parsing_and_expansion_process")
def timed_parsing_and_expansion_process(*args, **kwargs):
    return parsing_and_expansion_process(*args, **kwargs)

@time_function("post_process_expanded_data")
def timed_post_process_expanded_data(*args, **kwargs):
    return post_process_expanded_data(*args, **kwargs)

@time_function("load_parsed_addresses_from_cache")
def timed_load_parsed_addresses(*args, **kwargs):
    return load_parsed_addresses(*args, **kwargs)

@time_function("save_parsed_addresses_to_cache")
def timed_save_parsed_addresses(*args, **kwargs):
    return save_parsed_addresses(*args, **kwargs)

@time_function("load_postcode_district_lookup")
def timed_load_postcode_district_lookup(*args, **kwargs):
    return load_postcode_district_lookup(*args, **kwargs)

@time_function("preprocess_expandaded_ocod_data")
def timed_preprocess_expandaded_ocod_data(*args, **kwargs):
    return preprocess_expandaded_ocod_data(*args, **kwargs)

@time_function("load_and_process_pricepaid_data")
def timed_load_and_process_pricepaid_data(*args, **kwargs):
    return load_and_process_pricepaid_data(*args, **kwargs)

@time_function("add_missing_lads_ocod")
def timed_add_missing_lads_ocod(*args, **kwargs):
    return add_missing_lads_ocod(*args, **kwargs)

@time_function("load_voa_ratinglist")
def timed_load_voa_ratinglist(*args, **kwargs):
    return load_voa_ratinglist(*args, **kwargs)

@time_function("street_and_building_matching")
def timed_street_and_building_matching(*args, **kwargs):
    return street_and_building_matching(*args, **kwargs)

@time_function("substreet_matching")
def timed_substreet_matching(*args, **kwargs):
    return substreet_matching(*args, **kwargs)

@time_function("counts_of_businesses_per_oa_lsoa")
def timed_counts_of_businesses_per_oa_lsoa(*args, **kwargs):
    return counts_of_businesses_per_oa_lsoa(*args, **kwargs)

@time_function("voa_address_match_all_data")
def timed_voa_address_match_all_data(*args, **kwargs):
    return voa_address_match_all_data(*args, **kwargs)

@time_function("save_to_parquet")
def timed_save_to_parquet(df, path):
    return df.to_parquet(path)

###############
# Parse addresses (with caching)
###############
print("=" * 60)
print("CHECKING FOR CACHED PARSED ADDRESSES")
print("=" * 60)

if cache_path.exists():
    print(f"✓ Found cached parsed addresses for {zip_file.name}")
    print("Loading from cache...")
    ocod_data = timed_load_parsed_addresses(cache_path)
    print(f"✓ Loaded {len(ocod_data):,} records from cache")
else:
    print(f"✗ No cached parsed addresses found for {zip_file.name}")
    print("Performing full parsing process...")
    
    # Load and process the OCOD data as needed by your pipeline
    ocod_data = timed_load_and_prep_OCOD_data(str(zip_file))
    
    print(f"Parsing addresses for {zip_file.name}...")
    
    results = timed_parse_addresses_batch(
        df=ocod_data,
        model_path=str(model_path),
        target_column="property_address",
        batch_size=512,
        use_fp16=True
    )
    
    test = timed_convert_to_entity_dataframe(results)
    test = timed_parsing_and_expansion_process(all_entities=test)
    ocod_data = timed_post_process_expanded_data(test, ocod_data)
    
    # Save parsed addresses to cache
    timed_save_parsed_addresses(ocod_data, cache_path)

print("=" * 60)
print("PARSING PHASE COMPLETE")
print("=" * 60)

###############
# Geolocate
###############
print(f"Geolocating {zip_file.name}...")
postcode_district_lookup = timed_load_postcode_district_lookup(str(ONSPD_path))

ocod_data = timed_preprocess_expandaded_ocod_data(ocod_data, postcode_district_lookup)

price_paid_df = timed_load_and_process_pricepaid_data(
    file_path=str(price_paid_path), 
    processed_dir=processed_price_paid_dir,
    postcode_district_lookup=postcode_district_lookup, 
    years_needed=[2017, 2018, 2019]
)

ocod_data = timed_add_missing_lads_ocod(ocod_data, price_paid_df)

voa_businesses = timed_load_voa_ratinglist(str(voa_path), postcode_district_lookup)
del postcode_district_lookup  # memory management

ocod_data = timed_street_and_building_matching(ocod_data, price_paid_df, voa_businesses)
ocod_data = timed_substreet_matching(ocod_data, price_paid_df, voa_businesses)
del price_paid_df  # memory management

###########
# Classify
###########
print(f"Classifying {zip_file.name}...")
# Business processing
ocod_data = timed_counts_of_businesses_per_oa_lsoa(ocod_data, voa_businesses)
ocod_data = timed_voa_address_match_all_data(ocod_data, voa_businesses)
del voa_businesses  # memory management

timed_save_to_parquet(ocod_data, out_path)
print(f"Saved processed data to {out_path}")

# Print summary of all timings at the end
print("\n" + "=" * 80)
print("FUNCTION TIMING RESULTS (sorted by execution time)")
print("=" * 80)

# Sort functions by execution time (slowest first)
sorted_timings = sorted(timing_results.items(), key=lambda x: x[1], reverse=True)

total_time = sum(timing_results.values())
for i, (func_name, exec_time) in enumerate(sorted_timings, 1):
    percentage = (exec_time / total_time) * 100
    print(f"{i:2d}. {func_name:<45} {exec_time:>8.2f}s ({percentage:>5.1f}%)")

print("-" * 80)
print(f"    {'TOTAL EXECUTION TIME':<45} {total_time:>8.2f}s (100.0%)")
print("=" * 80)

# Also print top 5 slowest functions for quick reference
print(f"\nTOP 5 SLOWEST FUNCTIONS:")
print("-" * 40)
for i, (func_name, exec_time) in enumerate(sorted_timings[:5], 1):
    percentage = (exec_time / total_time) * 100
    print(f"{i}. {func_name}: {exec_time:.2f}s ({percentage:.1f}%)")

print(f"\nSingle cycle processing complete for {zip_file.name}")

# Print cache info
print(f"\n{'='*60}")
print("CACHE INFORMATION")
print(f"{'='*60}")
print(f"Cache directory: {parsed_cache_dir}")
print(f"Cache file: {cache_path.name}")
if cache_path.exists():
    cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"Cache size: {cache_size_mb:.1f} MB")
print(f"{'='*60}")