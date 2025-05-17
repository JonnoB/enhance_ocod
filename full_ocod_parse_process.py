from address_parsing_helper_functions import (load_and_prep_OCOD_data, spacy_pred_fn, parsing_and_expansion_process, post_process_expanded_data)
from locate_and_classify_helper_functions import (load_postocde_district_lookup, preprocess_expandaded_ocod_data, load_and_process_pricepaid_data,
                                                  add_missing_lads_ocod, load_voa_ratinglist, street_and_building_matching, substreet_matching,
                                                  counts_of_businesses_per_oa_lsoa, voa_address_match_all_data, classification_type1, classification_type2,
                                                  contract_ocod_after_classification)
import re
import zipfile
import sys

"""
This script runs the full parsing pipeline.
It takes a single argument which is the root path of the empty_homes_data path folder.
It requires that all data is found in that folder.
I hope to update the script so that it is easier to swap out the data for new data when it is created


docker run --rm -it -v $(pwd):/app jonno/parse_process:test ./app/enhance_ocod/full_ocod_parse_process.py ./app/data Malcom_UK_Owners_Missing.csv Malcom_UK_Owners_Missing_enhanced.csv
"""

args = sys.argv  

root_path = str(args[1])

data_file = str(args[2])

ouput_file = str(args[3])


ocod_data = load_and_prep_OCOD_data(root_path + data_file)

all_entities = spacy_pred_fn(spacy_model_path = root_path+'spacy_cpu_model', ocod_data = ocod_data)
#all_entities = load_cleaned_labels(root_path + 'full_dataset_no_overlaps.json')
full_expanded_data = parsing_and_expansion_process(all_entities, expand_addresses = True)

del all_entities #memory management

ocod_data = post_process_expanded_data(full_expanded_data, ocod_data)

del full_expanded_data #memory management

print("Load ONSPD")
# zip file handler  
zip = zipfile.ZipFile(root_path + 'ONSPD.zip')
# looks in the data folder for a csv file that begins ONSPD
#This will obviously break if the ONS change the archive structure
target_zipped_file = [i for i in zip.namelist() if re.search(r'^Data\/ONSPD.+csv$',i) ][0]
postcode_district_lookup = load_postocde_district_lookup(root_path + "ONSPD.zip", target_zipped_file)

print("Pre-process expanded ocod data")
ocod_data = preprocess_expandaded_ocod_data(ocod_data, postcode_district_lookup)
print("Load and pre-process the Land Registry price paid dataset")
#loads from a folder of price paid files
price_paid_df = load_and_process_pricepaid_data(root_path+'price_paid_files/', postcode_district_lookup)
print("Add in missing Local authority codes to the ocoda dataset")
ocod_data = add_missing_lads_ocod(ocod_data, price_paid_df)
print("Load and pre-process the voa business ratings list dataset")
voa_businesses = load_voa_ratinglist(root_path +'VOA_ratings.csv', postcode_district_lookup)

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
ocod_data = contract_ocod_after_classification(ocod_data, class_type = 'class2', classes = ['residential'] )

print('Process complete saving the enchanced ocod dataset to ' + root_path + ouput_file)

#subset the dataframe to only the columns necessary for the dataset and save
ocod_data.loc[:, ['title_number', 'within_title_id', 'within_larger_title', 'unique_id', 'unit_id', 'unit_type',
       'building_name', 'street_number', 'street_name', 'postcode', 'city',
       'district',  'region', 'property_address', 'oa11cd', 'lsoa11cd',
       'msoa11cd',  'lad11cd', 'class', 'class2']].rename(columns={'within_title_id':'nested_id',
                                                                  'within_larger_title':'nested_title'}).to_csv(root_path + ouput_file)

#FINISH!