from address_parsing_helper_functions import *
from locate_and_classify_helper_functions import *
import sys

"""
This script runs the full parsing pipeline.
It takes a single argument which is the root path of the empty_homes_data path folder.
It requires that all data is found in that folder.
I hope to update the script so that it is easier to swap out the data for new data when it is created
The script currently requires pre-labelled data. I hope to change that using a spacy model.

"""

args = sys.argv  

root_path = str(args[1])


ocod_data = load_and_prep_OCOD_data(root_path + 'OCOD_FULL_2022_02.csv')

all_entities = spacy_pred_fn(spacy_model_path = root_path+'spacy_data/cpu/model-best', ocod_data = ocod_data)

full_expanded_data = parsing_and_expansion_process(all_entities, expand_addresses = True)

del all_entities #memory management

ocod_data = post_process_expanded_data(full_expanded_data, ocod_data)

del full_expanded_data #memory management

print("Load ONSPD")
postcode_district_lookup = load_postocde_district_lookup(root_path + "ONSPD_NOV_2021_UK.zip", "Data/ONSPD_NOV_2021_UK.csv")

print("Pre-process expanded ocod data")
ocod_data = preprocess_expandaded_ocod_data(ocod_data, postcode_district_lookup)
print("Load and pre-process the Land Registry price paid dataset")
price_paid_df = load_and_process_pricepaid_data(root_path+'price_paid_files/', postcode_district_lookup)
print("Add in missing Local authority codes to the ocoda dataset")
ocod_data = add_missing_lads_ocod(ocod_data, price_paid_df)
print("Load and pre-process the voa business ratings list dataset")
voa_businesses = load_voa_ratinglist(root_path +'uk-englandwales-ndr-2017-listentries-compiled-epoch-0029-baseline-csv.csv', postcode_district_lookup)

del postcode_district_lookup #for memory purposes

print("Match street addresses and buildings")
ocod_data = street_and_building_matching(ocod_data, price_paid_df, voa_businesses)

#This takes some time
print('Sub-street matching, this takes some time')
ocod_data = substreet_matching(ocod_data, price_paid_df, voa_businesses)

del price_paid_df #for memory purposes
print('Add in businesses per oa and lsoa')
ocod_data = counts_of_businesses_per_oa_lsoa(ocod_data, voa_businesses)

print('Identify businesses using address matching')
ocod_data = voa_address_match_all_data(ocod_data, voa_businesses)

del voa_businesses #probably not necessary but still delete to save memory

print('Classification type 1')
ocod_data = classification_type1(ocod_data)
print('Classification type 2')
ocod_data = classification_type2(ocod_data)

print('Contract ocod dataset')
ocod_data = contract_ocod_after_classification(ocod_data, class_type = 'class2', classes = ['domestic'] )

print('Process complete saving the enchanced ocod dataset to ' + root_path + 'enhanced_ocod_dataset.csv')

ocod_data.to_csv(root_path+'enhanced_ocod_dataset.csv')

#FINISH!