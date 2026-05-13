# run_price_paid_process.py

from enhance_ocod.price_paid_process import check_and_preprocess_price_paid_data
from enhance_ocod.address_parsing import (
    load_postcode_district_lookup,
)

# Define paths to your data files
ONSPD_DATA_PATH = 'data/onspd/ONSPD_MAY_2025.zip'
PRICE_PAID_DATA_PATH = 'data/price_paid_data/price_paid_complete.csv'

print(f"Loading postcode district lookup from: {ONSPD_DATA_PATH}")
postcode_district_lookup = load_postcode_district_lookup(ONSPD_DATA_PATH)
print("Postcode district lookup loaded successfully.")

print(f"Checking and preprocessing price paid data from: {PRICE_PAID_DATA_PATH}")
check_and_preprocess_price_paid_data(PRICE_PAID_DATA_PATH, 
                                       postcode_district_lookup)
print("Price paid data processing complete.")