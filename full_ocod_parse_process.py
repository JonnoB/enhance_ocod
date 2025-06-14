from address_parsing_helper_functions import (load_and_prep_OCOD_data, spacy_pred_fn, parsing_and_expansion_process, post_process_expanded_data)

from locate_and_classify_helper_functions import (load_postocde_district_lookup, preprocess_expandaded_ocod_data, load_and_process_pricepaid_data,
                                                  add_missing_lads_ocod, load_voa_ratinglist, street_and_building_matching, substreet_matching,
                                                  counts_of_businesses_per_oa_lsoa, voa_address_match_all_data, classification_type1, classification_type2,
                                                  contract_ocod_after_classification)
import re
import zipfile
import sys
from typing import List

"""
Enhanced OCOD (Overseas Companies Ownership Data) Parser

This script processes raw OCOD data through a comprehensive pipeline that creates a structured dataset and
enriches property addresses with additional metadata, classifies properties, and identifies 
business/residential usage.

Pipeline Steps:
1. Address Parsing: Uses Spacy NER to extract structured components from property addresses
2. Data Enhancement: Enriches addresses with:
   - Postcode data (via ONSPD)
   - Price paid information (from Land Registry)
   - Business ratings (from VOA)
3. Classification: Determines property types and identifies multi-unit properties

Required Data Files (must be in root_path):
- Input CSV file: Raw OCOD data with property addresses
- spacy_cpu_model/: Trained Spacy model for address component recognition
- ONSPD.zip: ONS Postcode Directory containing postcode to area mappings
- VOA_ratings.csv: VOA business ratings list
- price_paid_files/: Directory containing Land Registry price paid data

Usage:
    python full_ocod_parse_process.py <root_path> <input_file> <output_file>

Arguments:
    root_path: Path to directory containing all required data files
    input_file: Name of input CSV file in root_path
    output_file: Name for enhanced output CSV file

Example:
    python full_ocod_parse_process.py ./data/ input.csv output_enhanced.csv

Docker Usage:
    docker run --rm -it -v $(pwd):/app jonno/parse_process:test \
        ./app/enhance_ocod/full_ocod_parse_process.py \
        ./app/data input.csv output.csv

"""

def main(args: List[str]) -> None:
    root_path = str(args[1])
    data_file = str(args[2])
    output_file = str(args[3])
    # Load and prepare initial data
    ocod_data = load_and_prep_OCOD_data(root_path + data_file)

    # NLP processing
    all_entities = spacy_pred_fn(spacy_model_path=root_path+'spacy_cpu_model', ocod_data=ocod_data)

        # Known typo corrections
    TYPO_CORRECTIONS = {
        "stanley court ": "stanley court, ",
        "100-1124": "100-112",
        "40a, 40, 40¨, 42, 44": "40a, 40, 40, 42, 44",
        # Add more as discovered
    }

    # Apply typo corrections
    for typo, correction in TYPO_CORRECTIONS.items():
        address_series = address_series.str.replace(typo, correction, regex=False)

    full_expanded_data = parsing_and_expansion_process(all_entities, expand_addresses=True)
    del all_entities  # memory management
    
    ocod_data = post_process_expanded_data(full_expanded_data, ocod_data)
    del full_expanded_data  # memory management

    # Load and process ONSPD data
    print("Load ONSPD")
    zip_file = zipfile.ZipFile(root_path + 'ONSPD.zip')
    target_zipped_file = [i for i in zip_file.namelist() if re.search(r'^Data/ONSPD.+csv$',i)][0]
    postcode_district_lookup = load_postocde_district_lookup(root_path + "ONSPD.zip", target_zipped_file)

    # Pre-process and enhance data
    print("Pre-process expanded ocod data")
    ocod_data = preprocess_expandaded_ocod_data(ocod_data, postcode_district_lookup)
    
    print("Load and pre-process the Land Registry price paid dataset")
    price_paid_df = load_and_process_pricepaid_data(root_path+'price_paid_files/', postcode_district_lookup)
    
    print("Add in missing Local authority codes to the ocod dataset")
    ocod_data = add_missing_lads_ocod(ocod_data, price_paid_df)
    
    print("Load and pre-process the voa business ratings list dataset")
    voa_businesses = load_voa_ratinglist(root_path +'VOA_ratings.csv', postcode_district_lookup)
    del postcode_district_lookup  # memory management

    # Address matching
    print("Match street addresses and buildings")
    ocod_data = street_and_building_matching(ocod_data, price_paid_df, voa_businesses)

    print('Sub-street matching, this takes some time')
    ocod_data = substreet_matching(ocod_data, price_paid_df, voa_businesses)
    del price_paid_df  # memory management

    # Business processing
    print('Add in businesses per oa and lsoa')
    ocod_data = counts_of_businesses_per_oa_lsoa(ocod_data, voa_businesses)

    print('Identify businesses using address matching')
    ocod_data = voa_address_match_all_data(ocod_data, voa_businesses)
    del voa_businesses  # memory management

    # Classification
    print('Classification type 1')
    ocod_data = classification_type1(ocod_data)
    print('Classification type 2')
    ocod_data = classification_type2(ocod_data)

    print('Contract ocod dataset')
    ocod_data = contract_ocod_after_classification(ocod_data, class_type='class2', classes=['residential'])

    # Save results
    print('Process complete saving the enhanced ocod dataset to ' + root_path + output_file)
    columns = ['title_number', 'within_title_id', 'within_larger_title', 'unique_id', 
              'unit_id', 'unit_type', 'building_name', 'street_number', 'street_name', 
              'postcode', 'city', 'district', 'region', 'property_address', 'oa11cd', 
              'lsoa11cd', 'msoa11cd', 'lad11cd', 'class', 'class2']
              
    ocod_data.loc[:, columns].rename(columns={
        'within_title_id': 'nested_id',
        'within_larger_title': 'nested_title'
    }).to_csv(root_path + output_file)

if __name__ == '__main__':
    main(sys.argv)