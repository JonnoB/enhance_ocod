
#the functions used by the empty homes project python notebooks
import io
import re
import time
import zipfile
import pandas as pd
import numpy as np

def expand_multi_id(multi_id_string):
    #the function takes a string that is in the form '\d+(\s)?(-|to)(\s)?\d+'
    #and outputs a continguous list of numbers between the two numbers in the string
    multi_id_list = [int(x) for x in re.findall(r'\d+', multi_id_string)]
    #min and max has to be used becuase somtimes the numbers are in descending order 4-3... I don't know why someone would do that
    out = list(range(min(multi_id_list), max(multi_id_list)+1))
    return(out)

def filter_contiguous_numbers(number_list, number_filter):
    #this function filters a list of contiguous house numbers/unit_id's to be even, odd, or unchanged
    #it takes as an argument a list of integers and a filter condition.
    #these values are contained in the label dictionary and reformated dataframe
    #The function ouputs the correct list of integers according to the filter condition

    if number_filter == 'odd':
        out = [ x for x in number_list if x%2==1]
    elif number_filter == 'even':
        out = [ x for x in number_list if x%2==0]
    else:
        out = number_list
    return out
#the loop iks quite fast considering the rest of the process so I am not sure printing is necessary anymore
#However, I want to keep it just in case so set the default to a very large numebr.
def expand_dataframe_numbers(df2, column_name, print_every = 1000, min_count = 1):
    #cycles through the dataframe and and expands xx-to-yy formats printing every ith iteration
    temp_list = []
    expand_time = 0
    filter_time = 0
    make_dataframe_time = 0
    
    for i in range(0, df2.shape[0]):
        
                
        start_expand_time = time.time()
        numbers_list = expand_multi_id(df2.loc[i][column_name])
        end_expand_time = time.time()

        numbers_list = filter_contiguous_numbers(numbers_list, df2.loc[i]['number_filter'])

        end_filter_time = time.time()
        
        dataframe_len = len(numbers_list)
        
        #This prevents large properties counting as several properties
        if dataframe_len>min_count:
            tmp = pd.concat([df2.iloc[i].to_frame().T]*dataframe_len, ignore_index=True)
            
            tmp[column_name] = numbers_list
        else:
            tmp = df2.iloc[i].to_frame().T
            
        temp_list.append(tmp)
        end_make_dataframe_time = time.time()
        
        expand_time = expand_time + (end_expand_time - start_expand_time)
        filter_time =filter_time + (end_filter_time - end_expand_time)
        make_dataframe_time = make_dataframe_time +(end_make_dataframe_time - end_filter_time)
        
        if (i>0) & (i%print_every==0): print("i=", i, " expand time,"+ str(round(expand_time, 3)) +
                           " filter time" + str(round(filter_time,3)) + 
                           " make_dataframe_time " + str(round(make_dataframe_time,3)))
    
    #once all the lines have been expanded concatenate them into a single dataframe
    
    out = pd.concat(temp_list)
    out = out.astype({column_name: 'string'})
    #The data type coming into the function is a string as it is in the form xx-yy
    #It needs to return a string as well otherwise there will be a pandas columns of mixed types
    #ehich causes problems later on
    out.loc[:, column_name] = out.loc[:, column_name].astype(str)

    return out


def load_postcode_district_lookup(file_path, target_post_area=None):
    """
    Load the ONS postcode district lookup table from a ZIP file.

    Loads the ONSPD (Office for National Statistics Postcode Directory) and reduces 
    it to relevant columns to save memory. Filters for English and Welsh postcodes.

    Args:
        file_path (str): Path to the ZIP file containing the ONSPD data.
        target_post_area (str, optional): Specific CSV file within the ZIP to load. 
            If None, automatically finds the appropriate file.

    Returns:
        pd.DataFrame: Processed postcode district lookup table with selected columns.
    """
    with zipfile.ZipFile(file_path) as zf:
        # If no target file specified, find it automatically
        if target_post_area is None:
            target_post_area = [
                i for i in zf.namelist() 
                if re.search(r'^Data/ONSPD.+csv$', i)
            ][0]

        with io.TextIOWrapper(zf.open(target_post_area), encoding='latin-1') as f:
            # Define dtypes for mixed-type columns to suppress warning and improve performance
            dtype_spec = {
                'pcds': 'string',
                'oslaua': 'string', 
                'oa11': 'string',
                'lsoa11': 'string',
                'msoa11': 'string',
                'ctry': 'string'
            }
            
            # Read only required columns with specified dtypes
            postcode_district_lookup = pd.read_csv(
                f, 
                usecols=['pcds', 'oslaua', 'oa11', 'lsoa11', 'msoa11', 'ctry'],
                dtype=dtype_spec,
                low_memory=False
            )
            
            # Filter for English and Welsh postcodes using isin() for better performance
            target_countries = ['E92000001', 'W92000004']
            postcode_district_lookup = postcode_district_lookup[
                postcode_district_lookup['ctry'].isin(target_countries)
            ]
            
            # Process postcode column more efficiently
            postcode_district_lookup['pcds'] = (
                postcode_district_lookup['pcds']
                .str.lower()
                .str.replace(' ', '', regex=False)  # Non-regex replacement is faster
            )
            
            # Rename columns
            postcode_district_lookup.rename(columns={
                'pcds': 'postcode2',
                'oslaua': 'lad11cd',
                'oa11': 'oa11cd',
                'lsoa11': 'lsoa11cd',
                'msoa11': 'msoa11cd'
            }, inplace=True)
            
            # Drop country column
            postcode_district_lookup.drop('ctry', axis=1, inplace=True)
    
    return postcode_district_lookup

def preprocess_expandaded_ocod_data(ocod_data, postcode_district_lookup):
    
    """
    Performs some light pre-processing on the previously created expanded ocod datast
    takes the expanded ocod dataframe as the only argument
    """
    ##add in the geographic area data like lsoa etc
    ocod_data['postcode2'] = ocod_data['postcode'].str.lower().str.replace("\s", "", regex = True)

    ocod_data = ocod_data.merge(postcode_district_lookup, 'left', left_on = "postcode2", right_on = "postcode2")

    ocod_data['street_name'] = ocod_data['street_name'].str.replace(r"^ +| +$", r"", regex=True)
    #this is to ensure that the street number includes only digits as it is used in the LSOA matching where the only thing
    #that matters is the street number not whether it is a or b or whatever.
    ocod_data['street_number2'] = ocod_data['street_number'].str.replace(r"^.*(?=\b[0-9]+$)", "", regex = True).str.replace(r"[^\d]", "", regex = True)

    #This stripped out versionof street name is used several times throughout the notebook
    ocod_data['street_name2'] = ocod_data.loc[:,'street_name'].str.replace(r"'", "", regex = True).\
    str.replace(r"s(s)?(?=\s)", "", regex = True).str.replace(r"\s", "", regex = True)
    
    return ocod_data


def add_missing_lads_ocod(ocod_data, price_paid_df):
    
    """
    Not all observations have a postcode and some postcodes are erroneous or otherwise cannot be found in the postcode database
    These entries need to be found and valid districts added in
    """

    #when there are multiples take the lad11cd with the largest number of counts
    lad_lookup = price_paid_df[['district', 'lad11cd']].dropna().groupby(['district', 'lad11cd']).size().reset_index()
    lad_lookup.rename(columns = {0:'counts'}, inplace = True)
    lad_lookup = lad_lookup.sort_values('counts', ascending=False).groupby('lad11cd').first().reset_index()
    lad_lookup.drop('counts', axis = 1, inplace = True)

    temp = ocod_data

    #only take data that is missing the lad code. If you include only is missing postcode, bad postcode entries also get excluded leaving a small number of
    ## blanks in the final dataset
    temp = temp[temp['lad11cd'].isna()]

    temp = temp.drop('lad11cd', axis = 1)

    temp = temp.merge(lad_lookup, left_on = "district", right_on = "district")

    temp['lad11cd'].isna().sum() #there are no na values showing all districts now have a lad code

    #join the ocod data back together again
    ocod_data = pd.concat( [temp, ocod_data[~ocod_data['lad11cd'].isna()]])
    
    return ocod_data


def clean_street_numbers(df, original_column='number_or_name'):
    """
    Optimized version with fewer pandas operations and better vectorization
    """
    
    # Create a copy to work with
    street_num = df[original_column].copy()
    
    # Chain multiple replacements in fewer operations
    street_num = (street_num
                  .str.replace(r"\(.+\)", "", regex=True, case=False)
                  .str.replace(r"\+", " ", regex=True, case=False)
                  .str.replace(r"@", " at ", regex=True, case=False)
                  .str.replace(r"&", " and ", regex=True, case=False)
                  .str.replace(r"\s*-\s*", "-", regex=True, case=False))
    
    # Handle unit/suite/room in one vectorized operation
    unit_mask = street_num.str.contains(r"unit|suite|room", regex=True, case=False, na=False)
    street_num = street_num.where(~unit_mask)
    
    # Continue with remaining operations in a chain
    street_num = (street_num
                  .str.extract(r"([^\s]+)$")[0]
                  .str.replace(r"[a-zA-Z]+", "", regex=True, case=False)
                  .str.replace(r"^-+|-+$", "", regex=True, case=False)
                  .str.replace(r"[\\\/]", " ", regex=True, case=False)
                  .str.replace(r"-{2,}", "-", regex=True, case=False))
    
    # Clean up empty strings and single hyphens in one operation
    cleanup_mask = (street_num.str.len() > 0) & (street_num != "-")
    street_num = street_num.where(cleanup_mask)
    
    df['street_number'] = street_num
    return df

    
##street matching

def create_lad_streetname2(df, target_lad, street_column_name):
    #
    # used to generalise the cleaning in the street matching process
    # Creates a column called streetname 2 for a single lad.
    #this function has been largely replaced as voa and ocod datasets have street name 2
    # created on loading.
    #filters to a single LAD
    temp = df.loc[(df['lad11cd']==target_lad)].copy(deep = True)
    
    temp.loc[:,'street_name2'] = temp[street_column_name].copy(deep=True)
    #clean street names of common matching errors
    #remove apostraphe's
    #remove trailing 's'
    #remove all spaces
    temp.loc[:,'street_name2'] = temp.loc[:,'street_name2'].str.replace(r"'", "", regex = True).\
    str.replace(r"s(s)?(?=\s)", "", regex = True).str.replace(r"\s", "", regex = True)
    
    return temp
    

def massaged_address_match(ocod_data, voa_data, target_lad):

	#Matches addresses betweeen the voa and ocod dataet


    ocod_data_road = ocod_data.loc[(ocod_data['lad11cd']==target_lad)].copy(deep = True)
    
    LAD_biz = voa_data.loc[(voa_data['lad11cd']==target_lad)].copy(deep = True)
    
    #replace nan values to prevent crash    
    ocod_data_road.loc[ocod_data_road.street_name.isna(),'street_name2'] ="xxxstreet name missingxxx"
    
    #the roads which match
    ocod_data_road['street_match'] = ocod_data_road['street_name2'].isin(LAD_biz.street_name2.unique())
    
    #remove irrelevant streets
    LAD_biz = LAD_biz[LAD_biz['street_name2'].isin(ocod_data_road['street_name2'].unique()) & LAD_biz['street_name2'].notna() ]
    #create the database table
    all_street_addresses = create_all_street_addresses(LAD_biz, target_lad).drop_duplicates(subset = ['street_name2', 'street_number']).rename(columns = {'street_number':'street_number2'})
    
    #prevents matching errors caused by some arcance and horrible thing
    all_street_addresses['street_number2'] = all_street_addresses['street_number2'].astype('str')
    all_street_addresses['street_number2'] = all_street_addresses['street_number2'].str.strip()

    ocod_data_road['street_number2'] = ocod_data_road['street_number2'].astype('str')
    ocod_data_road['street_number2'] = ocod_data_road['street_number2'].str.strip()

    #pre-make the new column and assign nan to all values. THis might make things a bit faster
    ocod_data_road['address_match'] = np.nan
    
    ocod_data_road = ocod_data_road.merge(all_street_addresses, how = "left", on = ['street_name2', 'street_number2'])
    
    ocod_data_road['address_match'] = ocod_data_road['business_address'].notna()
    
    ocod_data_road.loc[ocod_data_road['street_name2']== "xxxstreet name missingxxx",'street_name2'] = np.nan
    
     
    
    return(ocod_data_road)
    
    
def find_filter_type(street_num):
 #gets the highest street number and uses it to work out if the property is on the odd or even side of the street, or if that rule is ignore and it is all numbers
    values = [int(x) for x in street_num.replace("-", " ").split() if x.strip()]
    if (max(values)%2==0) & (min(values)%2==0):
        out = "even"
    elif (max(values)%2==1) & (min(values)%2==1):
        out = "odd"
    else:
        out = "all"
    
    return out
    
    


def street_number_to_lsoa(temp_road, target_number):
    #this is a helper function that finds the nearest building with known lsoa.
    #It is sensitive to odd/even as in the UK that determins the side of the road 
    #and in terms of boundaries can define a border.
    #this makes sure the data looks on the correct side of the road, as the boundary is liekly to fall down the middle of the road
    modulo_2 = target_number%2

    sub_road = temp_road[(pd.to_numeric(temp_road['street_number2'])%2 == modulo_2)]

    #It is not a given that there are odds and even's on a road. in that case just use the numbers you can find 
    if len(sub_road)>0:
        diff_array = np.abs(pd.to_numeric(sub_road['street_number2'])- target_number)

        #in the case there is more than 1 just take the first and give me a break
        out = sub_road.iloc[np.where(diff_array == diff_array.min())].iloc[0,2]
    else:
        diff_array = np.abs(pd.to_numeric(temp_road['street_number2'])- target_number)

        #in the case there is more than 1 just take the first and give me a break
        out = temp_road.iloc[np.where(diff_array == diff_array.min())].iloc[0,2]
    
    return out


def load_voa_ratinglist(file_path, postcode_district_lookup):
    """
    Performance-optimized version focusing on string operations
    """
    
    def find_largest_file(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            largest_file = max(zip_ref.infolist(), key=lambda x: x.file_size)
            return largest_file.filename

    VOA_headers_needed = [
        "Primary And Secondary Description Code",    # 4
        "Primary Description Text",                  # 5  
        "Unique Address Reference Number UARN",      # 6
        "Full Property Identifier",                  # 7
        "Firms Name",                               # 8
        "Number Or Name",                           # 9
        "Street",                                   # 10
        "Town",                                     # 11
        "Postal District",                          # 12
        "County",                                   # 13
        "Postcode",                                 # 14
    ]
    
    VOA_headers = [x.lower().replace(" ", "_") for x in VOA_headers_needed]
    usecols = list(range(4, 15))
    
    # Read data (same as before)
    if file_path.lower().endswith('.zip'):
        largest_file = find_largest_file(file_path)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            with zip_ref.open(largest_file) as csv_file:
                voa_businesses = pd.read_csv(csv_file,
                           sep="*",
                           encoding_errors='ignore',
                           header=None,
                           names=VOA_headers,
                           usecols=usecols,
                           index_col=False,
                           dtype=str,
                           )
    else:
        voa_businesses = pd.read_csv(file_path,
                       sep="*",
                       encoding_errors='ignore',
                       header=None,
                       names=VOA_headers,
                       usecols=usecols,
                       index_col=False,
                       dtype=str,
                       )

    # Early filtering BEFORE expensive string operations
    # Filter out unwanted data first to reduce processing volume
    print(f"Initial rows: {len(voa_businesses)}")
    
    # Use faster operations for filtering
    advertising_mask = voa_businesses['primary_description_text'].str.contains("ADVERTISING", na=False)
    voa_businesses = voa_businesses[~advertising_mask]
    
    excluded_codes = {'C0', 'CP', 'CP1', 'CX', 'MX'}  # Use set for faster lookup
    parking_mask = voa_businesses['primary_and_secondary_description_code'].isin(excluded_codes)
    voa_businesses = voa_businesses[~parking_mask]

    voa_businesses['postcode'] = voa_businesses['postcode'].str.lower()
    voa_businesses['street'] = voa_businesses['street'].str.lower()
    
    # Chain regex operations to reduce passes through data
    voa_businesses['postcode2'] = voa_businesses['postcode'].str.replace(r"\s", "", regex=True)
    
    voa_businesses['street_name2'] = (voa_businesses['street']
                                     .str.replace(r"'", "", regex=True)
                                     .str.replace(r"s(s)?(?=\s)", "", regex=True)  
                                     .str.replace(r"\s", "", regex=True))

    # First merge to reduce dataset size
    voa_businesses = voa_businesses.merge(postcode_district_lookup, left_on='postcode2', right_on="postcode2", how='inner')

    
    # Now do expensive operations on smaller dataset
    voa_businesses = clean_street_numbers(voa_businesses, original_column='number_or_name')

    westfield_mask = voa_businesses['full_property_identifier'].str.contains('WESTFIELD SHOPPING CENTRE', na=False)
    if westfield_mask.any():  # Only process if there are matches
        voa_businesses.loc[westfield_mask, 'street_number'] = np.nan
    
    if 'street_number' in voa_businesses.columns:
        phone_mask = voa_businesses['street_number'].str.contains(r'-\d+-', regex=True, na=False)
        if phone_mask.any():  # Only process if there are matches
            voa_businesses.loc[phone_mask, 'street_number'] = np.nan

    # OPTIMIZATION 6: Optimize building_name extraction
    voa_businesses['building_name'] = (voa_businesses['number_or_name']
                                     .str.lower()
                                     .str.extract(r'((?<!\S)(?:(?!\b(?:\)|\(|r\/o|floor|floors|pt|and|annexe|room|gf|south|north|east|west|at|on|in|of|adjoining|adj|basement|bsmt|fl|flr|flrs|wing)\b)[^\n\d])*? house\b)', expand=False))

    return voa_businesses


def street_and_building_matching(ocod_data, price_paid_df, voa_businesses):
    """
    Where there is no postcode properties are located using the building name or the street name
    
    This process is quite convoluted and there is certainly a more efficient and pythonic way
    however the order within each filling method is important to ensure that there are no duplicates
    as this causes the OCOD dataset to grow with duplicates
    """
    
    # Replace the missing lsoa using street matching
    temp_lsoa = pd.concat([
        price_paid_df[['street_name2', 'lad11cd', 'lsoa11cd']], 
        voa_businesses[['street_name2', 'lad11cd', 'lsoa11cd']]
    ]).dropna()
    
    # Optimize the groupby operations
    street_counts = (temp_lsoa.groupby(['street_name2', 'lad11cd', 'lsoa11cd'])
                    .size()
                    .reset_index(name='temp_count')
                    .groupby(['street_name2', 'lad11cd'])
                    .size()
                    .reset_index(name='counts'))
    
    temp = (street_counts[street_counts['counts'] == 1]
           .merge(temp_lsoa.drop_duplicates(), how="left", on=['street_name2', 'lad11cd'])
           .rename(columns={'lsoa11cd': 'lsoa_street'}))
    
    ocod_data = ocod_data.merge(temp[['lsoa_street', "street_name2", "lad11cd"]], 
                               how="left", on=['street_name2', 'lad11cd'])
    
    # Replace the missing lsoa using building matching
    # Optimize regex operations by compiling patterns
    import re
    digit_pattern = re.compile(r"\d+")
    special_chars_pattern = re.compile(r"and|\&|-|,")
    whitespace_pattern = re.compile(r"^ +| +$")
    
    # Create building name processing function to avoid repetition
    def clean_building_name(series):
        return (series.str.replace(digit_pattern, "", regex=True)
               .str.replace(special_chars_pattern, "", regex=True)
               .str.replace(whitespace_pattern, "", regex=True))
    
    # Process building matching more efficiently
    pp_temp = price_paid_df[['paon', 'lad11cd', 'oa11cd', 'lsoa11cd']].copy()
    pp_temp['paon_clean'] = clean_building_name(pp_temp['paon'])
    pp_temp = pp_temp[pp_temp['paon_clean'].str.len() > 0].drop_duplicates()
    
    building_counts = (pp_temp.groupby(['paon_clean', 'lad11cd'])
                      .size()
                      .reset_index(name='counts'))
    
    temp = (building_counts[building_counts['counts'] == 1]
           .merge(pp_temp, left_on=['paon_clean', 'lad11cd'], 
                  right_on=['paon_clean', 'lad11cd'], how="left")
           .rename(columns={'lsoa11cd': 'lsoa_building',
                           'oa11cd': 'oa_building',
                           'paon_clean': 'building_name'})
           .drop_duplicates(subset=['building_name', 'lsoa_building']))
    
    ocod_data = ocod_data.merge(temp[['lsoa_building', "oa_building", "building_name", "lad11cd"]], 
                               how="left", on=['building_name', 'lad11cd'])
    
    # VOA businesses merge - filter and process in one step
    voa_filtered = (voa_businesses.loc[~voa_businesses['building_name'].isna(), 
                                     ['building_name', 'oa11cd', 'lsoa11cd', 'lad11cd']]
                   .drop_duplicates(subset=['building_name', 'lsoa11cd'])
                   .rename(columns={'lsoa11cd': 'lsoa_busi_building',
                                   'oa11cd': 'oa_busi_building'}))
    
    ocod_data = ocod_data.merge(voa_filtered, how="left", on=['building_name', 'lad11cd'])
    
    # Fill in the lsoa11cd using the newly ID'd lsoa from the matching process
    # Optimize the filling process using fillna with method chaining
    lsoa_columns = ['lsoa_street', 'lsoa_building', 'lsoa_busi_building']
    oa_columns = ['oa_building', 'oa_busi_building']
    
    # More efficient filling using combine_first
    for col in lsoa_columns:
        ocod_data['lsoa11cd'] = ocod_data['lsoa11cd'].combine_first(ocod_data[col])
    
    for col in oa_columns:
        ocod_data['oa11cd'] = ocod_data['oa11cd'].combine_first(ocod_data[col])
    
    # Optimize nested properties processing - combine LSOA and OA processing
    def process_nested_properties(ocod_data, code_col, new_col_name):
        temp = (ocod_data.loc[(ocod_data[code_col].notnull()) & 
                             (ocod_data['within_larger_title']), 
                             [code_col, 'title_number']]
               .drop_duplicates()
               .groupby('title_number')[code_col]
               .first()
               .reset_index()
               .rename(columns={code_col: new_col_name}))
        
        ocod_data = ocod_data.merge(temp, how="left", on="title_number")
        ocod_data[code_col] = ocod_data[code_col].combine_first(ocod_data[new_col_name])
        return ocod_data
    
    # Process both LSOA and OA nested properties
    ocod_data = process_nested_properties(ocod_data, 'lsoa11cd', 'lsoa_nested')
    ocod_data = process_nested_properties(ocod_data, 'oa11cd', 'oa_nested')
    
    return ocod_data

def substreet_matching(ocod_data, price_paid_df, voa_businesses, print_lads=False, print_every=100):
    """
    Some streets are on the boundary of LSOA this section uses the street number to match to the nearest lsoa.
    """
    # Pre-filter the data once
    missing_lsoa_mask = (ocod_data['street_name'].notnull() & 
                        ocod_data['street_number'].notnull() & 
                        ocod_data['lsoa11cd'].isnull())
    
    if not missing_lsoa_mask.any():
        return ocod_data
    
    unique_lad_codes = ocod_data[missing_lsoa_mask]['lad11cd'].unique()
    
    # Pre-process street numbers for all datasets once
    ocod_subset = ocod_data[missing_lsoa_mask].copy()
    ocod_subset['street_number2'] = extract_numeric_fast(ocod_subset['street_number'])
    
    # Pre-process price_paid and voa data
    price_paid_df = price_paid_df.copy()
    price_paid_df['street_number2'] = extract_numeric_fast(price_paid_df['street_number'])
    
    voa_businesses = voa_businesses.copy()
    voa_businesses['street_number2'] = extract_numeric_fast(voa_businesses['street_number'])
    
    # Combine reference data once
    reference_data = pd.concat([
        price_paid_df[['street_name2', 'street_number', 'street_number2', 'lsoa11cd', 'lad11cd']],
        voa_businesses[['street_name2', 'street_number', 'street_number2', 'lsoa11cd', 'lad11cd']]
    ]).dropna()
    
    filled_lsoa_list = []
    
    for i, target_lad in enumerate(unique_lad_codes, 1):
        if print_lads: 
            print(target_lad)
        if i % print_every == 0: 
            print(f"lad {i} of {len(unique_lad_codes)}")
        
        # Vectorized LAD filtering
        lad_mask_ocod = ocod_subset['lad11cd'] == target_lad
        lad_mask_ref = reference_data['lad11cd'] == target_lad
        
        missing_lsoa_df = ocod_subset[lad_mask_ocod].copy()
        temp_lsoa_raw = reference_data[lad_mask_ref].copy()
        
        if len(missing_lsoa_df) == 0 or len(temp_lsoa_raw) == 0:
            continue
            
        target_street_names = missing_lsoa_df['street_name2'].unique()
        
        # Filter reference data to relevant streets
        temp_lsoa = temp_lsoa_raw[
            temp_lsoa_raw['street_name2'].isin(target_street_names) & 
            temp_lsoa_raw['street_number2'].notna()
        ]
        
        if len(temp_lsoa) == 0:
            continue
            
        temp_lsoa = create_all_street_addresses(
            temp_lsoa, target_lad, 
            ['street_name2', 'street_number2', 'lsoa11cd']
        )
        
        # Process all streets at once using vectorized operations
        result_df = process_streets_vectorized(missing_lsoa_df, temp_lsoa, target_street_names)
        
        if len(result_df) > 0:
            filled_lsoa_list.append(result_df)
    
    # Combine results
    if not filled_lsoa_list:
        temp_lsoa = ocod_data[0:0]
    else:
        temp_lsoa = pd.concat(filled_lsoa_list, ignore_index=True)
    
    # Rejoin data
    if len(temp_lsoa) > 0:
        ocod_data = pd.concat([
            ocod_data[~ocod_data['unique_id'].isin(temp_lsoa['unique_id'])], 
            temp_lsoa
        ], ignore_index=True)
    
    # Handle nested addresses (optimized)
    ocod_data = fill_nested_lsoa_vectorized(ocod_data)
    
    return ocod_data



def extract_numeric_fast(series):
    """Fast numeric extraction using vectorized operations"""
    # Use str.extract with regex to get the last number in the string
    numeric_str = series.str.extract(r'(\d+)$')[0]
    return pd.to_numeric(numeric_str, errors='coerce')

def process_streets_vectorized(missing_lsoa_df, temp_lsoa, target_street_names):
    """Vectorized processing of multiple streets"""
    results = []
    
    # Group reference data by street for faster lookup
    temp_grouped = temp_lsoa.groupby('street_name2')
    
    for target_road in target_street_names:
        missing_road = missing_lsoa_df[
            (missing_lsoa_df['street_name2'] == target_road) & 
            (missing_lsoa_df['street_number2'].notna())
        ].copy()
        
        if len(missing_road) == 0:
            continue
            
        if target_road in temp_grouped.groups:
            temp_road = temp_grouped.get_group(target_road)
            
            # Vectorized LSOA assignment
            lsoa_values = find_nearest_lsoa_vectorized(
                missing_road['street_number2'].values,
                temp_road
            )
            
            missing_road = missing_road.copy()
            missing_road['lsoa11cd'] = lsoa_values
            results.append(missing_road)
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def find_nearest_lsoa_vectorized(target_numbers, temp_road):
    """Vectorized version of street_number_to_lsoa logic"""
    temp_numbers = temp_road['street_number2'].values
    temp_lsoa = temp_road['lsoa11cd'].values
    
    results = []
    
    for target_num in target_numbers:
        modulo_2 = int(target_num) % 2
        
        # Find same parity numbers
        same_parity_mask = (temp_numbers % 2) == modulo_2
        
        if same_parity_mask.any():
            # Use same parity numbers
            candidates = temp_numbers[same_parity_mask]
            candidate_lsoa = temp_lsoa[same_parity_mask]
        else:
            # Use all numbers
            candidates = temp_numbers
            candidate_lsoa = temp_lsoa
        
        # Find nearest
        diff_array = np.abs(candidates - target_num)
        min_idx = np.argmin(diff_array)
        results.append(candidate_lsoa[min_idx])
    
    return results

def fill_nested_lsoa_vectorized(ocod_data):
    """Optimized nested LSOA filling"""
    # Find titles with known LSOA
    known_lsoa = ocod_data[
        (ocod_data['lsoa11cd'].notna()) & 
        (ocod_data['within_larger_title'])
    ][['lsoa11cd', 'title_number']].drop_duplicates()
    
    # Take first LSOA per title (handles multiple LSOA per title)
    lsoa_map = known_lsoa.groupby('title_number')['lsoa11cd'].first()
    
    # Map to missing values
    missing_mask = ocod_data['lsoa11cd'].isna()
    ocod_data.loc[missing_mask, 'lsoa11cd'] = ocod_data.loc[missing_mask, 'title_number'].map(lsoa_map)
    
    return ocod_data

def counts_of_businesses_per_oa_lsoa(ocod_data, voa_businesses):
    
    """
    This function allows areas with no  businesses to automatically exclude business from the classification
    """
    #Create a dataframe that contains the counts of businesses per OA
    postcode_counts_voa = voa_businesses.groupby('oa11cd').size().reset_index(name = 'business_counts')
    ocod_data = pd.merge(ocod_data, postcode_counts_voa, on = "oa11cd", how = "left")
    ocod_data["business_counts"] = ocod_data["business_counts"].fillna(0)

    #do the same for lsoa
    lsoa_counts_voa = voa_businesses.groupby('lsoa11cd').size().reset_index(name = 'lsoa_business_counts')
    ocod_data = pd.merge(ocod_data, lsoa_counts_voa, on = "lsoa11cd", how = "left")
    ocod_data["lsoa_business_counts"] = ocod_data["lsoa_business_counts"].fillna(0)

    return ocod_data

def voa_address_match_all_data(ocod_data, voa_businesses, print_lads=False, print_every=50):
    """
    Vectorized version - processes all LADs at once instead of looping DOES NOT USE MASSAGED_DATA FUNCTION!
    """
    # Filter out NaN LADs upfront
    valid_lads = ocod_data['lad11cd'].dropna().unique()
    print(f"Processing {len(valid_lads)} LADs")
    
    # Pre-filter both datasets to only valid LADs (avoids repeated filtering)
    ocod_filtered = ocod_data[ocod_data['lad11cd'].isin(valid_lads)].copy()
    voa_filtered = voa_businesses[voa_businesses['lad11cd'].isin(valid_lads)].copy()
    
    # OPTIMIZATION 1: Handle missing street names once for entire dataset
    ocod_filtered.loc[ocod_filtered['street_name'].isna(), 'street_name2'] = "xxxstreet name missingxxx"
    
    # OPTIMIZATION 2: Create street match mapping once for all LADs
    print("Creating street match lookup...")
    street_matches = create_street_match_lookup(ocod_filtered, voa_filtered)
    ocod_filtered['street_match'] = ocod_filtered['street_name2'].isin(street_matches)
    
    # OPTIMIZATION 3: Create address lookup once for all relevant streets
    print("Creating address lookup...")
    all_street_addresses = create_vectorized_address_lookup(voa_filtered, street_matches)
    
    # OPTIMIZATION 4: Single merge operation instead of per-LAD merges
    print("Performing address matching...")
    ocod_filtered = perform_vectorized_address_match(ocod_filtered, all_street_addresses)
    
    # Clean up the placeholder
    ocod_filtered.loc[ocod_filtered['street_name2'] == "xxxstreet name missingxxx", 'street_name2'] = np.nan
    
    return ocod_filtered


def create_street_match_lookup(ocod_data, voa_data):
    """Create set of streets that exist in both datasets"""
    ocod_streets = set(ocod_data['street_name2'].dropna())
    voa_streets = set(voa_data['street_name2'].dropna())
    return ocod_streets.intersection(voa_streets)


def create_vectorized_address_lookup(voa_data, valid_streets):
    """
    Create address lookup table for all valid streets at once
    """
    # Filter to only streets that match between datasets
    voa_relevant = voa_data[
        voa_data['street_name2'].isin(valid_streets) & 
        voa_data['street_name2'].notna()
    ].copy()
    
    if len(voa_relevant) == 0:
        return pd.DataFrame(columns=['street_name2', 'street_number2', 'business_address'])
    
    # Process address expansion vectorized by LAD groups
    lad_groups = []
    for lad in voa_relevant['lad11cd'].unique():
        lad_data = voa_relevant[voa_relevant['lad11cd'] == lad]
        lad_addresses = create_all_street_addresses(lad_data, lad)
        lad_groups.append(lad_addresses)
    
    # Combine all and deduplicate
    all_addresses = pd.concat(lad_groups, ignore_index=True)
    return all_addresses.drop_duplicates(subset=['street_name2', 'street_number2'])


def perform_vectorized_address_match(ocod_data, all_street_addresses):
    """
    Perform the address matching in a single vectorized operation
    """
    # Ensure string types and strip whitespace
    all_street_addresses['street_number2'] = (all_street_addresses['street_number2']
                                            .astype('str').str.strip())
    ocod_data['street_number2'] = ocod_data['street_number2'].astype('str').str.strip()
    
    # Single merge operation
    result = ocod_data.merge(
        all_street_addresses, 
        how='left', 
        on=['street_name2', 'street_number2']
    )
    
    # Create address match flag
    result['address_match'] = result['business_address'].notna()
    
    return result


def create_all_street_addresses(voa_businesses, target_lad, 
                                        return_columns=['street_name2', 'street_number', 'business_address']):
    """
    Optimized version with minimal copying and better vectorization
    """
    if len(voa_businesses) == 0:
        return pd.DataFrame(columns=['street_name2', 'street_number2', 'business_address'])
    
    # Work with the filtered data directly (no deep copy needed)
    temp = voa_businesses.copy()  # Shallow copy is sufficient
    
    # FIX: Proper boolean logic for filtering out problematic street numbers
    has_dot = temp['street_number'].str.contains(r"\.", regex=True, na=False)
    temp = temp[~has_dot]  # Remove rows with dots in street numbers
    
    # Vectorized multi-address detection
    temp['is_multi'] = temp['street_number'].str.contains(r"-", regex=True, na=False)
    
    temp.rename(columns={'full_property_identifier': 'business_address'}, inplace=True)
    
    # Split processing
    not_multi = temp[~temp['is_multi']]
    multi_addresses = temp[temp['is_multi']]
    
    results = []
    
    # Add non-multi addresses
    if len(not_multi) > 0:
        results.append(not_multi[return_columns].rename(columns={'street_number': 'street_number2'}))
    
    # Process multi addresses
    if len(multi_addresses) > 0:
        # Vectorized filter type detection
        multi_addresses = multi_addresses.reset_index(drop=True)
        filter_types = [find_filter_type(x) for x in multi_addresses['street_number']]
        multi_addresses['number_filter'] = filter_types
        
        # Expand addresses
        expanded = expand_dataframe_numbers(multi_addresses, 'street_number', print_every=10000, min_count=1)
        results.append(expanded[return_columns].rename(columns={'street_number': 'street_number2'}))
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['street_name2', 'street_number2', 'business_address'])


#
#Capture groups were used in this function previously. These have now been removed or replaced with non capture groups "(?:)"
#
def classification_type1(ocod_data):
    ocod_data['class'] = np.select(
        [
            ocod_data['property_address'].str.contains(r"^(?:land|plot)", case = False),
            ocod_data['property_address'].str.contains(r"^(?:[a-z\s]*)(?:garage|parking(?:\s)?space|parking space|car park(?:ing)?)", case = False),
            ocod_data['property_address'].str.contains(r"^(?:the airspace|airspace)", case = False),
            ocod_data['property_address'].str.contains(r"penthouse|flat|apartment", case = False),
            ocod_data['address_match'],
            ocod_data['property_address'].str.contains(r"cinema|hotel|office|centre|\bpub|holiday(?:\s)?inn|travel lodge|travelodge|medical|business|cafe|^shop| shop|service|logistics|building supplies|restaurant|home|^store(?:s)?\b|^storage\b|company|ltd|limited|plc|retail|leisure|industrial|hall of|trading|commercial|technology|works|club,|advertising|school|church|(?:^room)", case = False), 
            ocod_data['property_address'].str.contains(r"^[a-z\s']+\b(?:land(?:s)?|plot(?:s)?)\b", case = False), #land with words before it
            ocod_data['building_name'].str.contains(r'\binn$|public house|^the\s\w+\sand\s\w+|(?:tavern$)',case=False,  na=False), #pubs in various guises
            ocod_data['oa_busi_building'].notnull(),#a business building was matched
            ocod_data['business_address'].notnull()

        ], 
        [
            'land',
            'carpark',
            'airspace',
            'residential',
            'business',
            'business',
            'land',
            'business',
            'business',
            'business'
        ], 
        default='unknown'
    )
    
    #Fills out unknown class values using the known class values from the same title number

    temp_fill = ocod_data[~ocod_data['class'].isin(['unknown', 'airspace', 'carpark']) & (ocod_data['within_larger_title'])].groupby(['title_number', 'class']).\
    size().reset_index()[['title_number', 'class']].drop_duplicates()

    temp = ocod_data[ocod_data['title_number'].isin(temp_fill['title_number']) & (ocod_data['class']=='unknown') & (ocod_data['within_larger_title'])].copy()

    temp.drop('class', axis = 1, inplace = True)

    temp = temp.merge(temp_fill, how = "left", on = "title_number")

    ocod_data = pd.concat([temp, ocod_data[~ocod_data['unique_id'].isin(temp['unique_id'])]])

    
    return ocod_data

def classification_type2(ocod_data):
    
    ocod_data['class2'] = np.select(
        [   (ocod_data['class']=='unknown') & ((ocod_data.property_address.str.contains('^(?:the )?unit') & ocod_data.property_address.str.contains('park', regex = True))), #contains the word unit and park
            (ocod_data['class']=='unknown') & (ocod_data['business_counts']==0), #if there are no businesses in the oa then it is a residential
            (ocod_data['class']=='unknown') & (ocod_data['lsoa_business_counts']==0), #if there are no businesses in the lsoa then it is a residential
            (ocod_data['class']=='unknown') & (ocod_data['street_match']) & (ocod_data['street_name'].notnull()) & (ocod_data['street_number'].notnull()),
            (ocod_data['class']=='unknown') & (~ocod_data['street_match']).fillna(False) & (ocod_data['street_name'].notnull()),
            (ocod_data['class']=='unknown') & (ocod_data['building_name'].notnull())

        ], 
        [
            'business',
            'residential',
            'residential',
            'residential',
            'residential',
            'residential'
        ], 
        default= ocod_data['class']
    )


    #fillout larger titles that are now partially tagged

    temp_fill = ocod_data[~ocod_data['class2'].isin(['unknown', 'airspace', 'carpark']) & (ocod_data['within_larger_title'])].groupby(['title_number', 'class2']).\
    size().reset_index()[['title_number', 'class2']].drop_duplicates()

    temp = ocod_data[ocod_data['title_number'].isin(temp_fill['title_number']) & (ocod_data['class2']=='unknown') & (ocod_data['within_larger_title'])].copy()

    temp.drop('class2', axis = 1, inplace = True)

    temp = temp.merge(temp_fill, how = "left", on = "title_number")

    ocod_data = pd.concat([temp, ocod_data[~ocod_data['unique_id'].isin(temp['unique_id'])]])

    return ocod_data

def contract_ocod_after_classification(ocod_data, class_type = 'class2', classes = ['residential'] ):
    
    """
    This function removes mutli-addresses where they are not necessary. This is becuase only residential properties should be classed as 
    multiples. However, in some cases including unknown's may be of interest.
    
    The function takes as inputs
    ocod_data the ocod dataset as a pandas dataframe after classification by the classification functions
    class_type: A string either class or class2. The default is 'class2'
    classes: a list of strings naming the classes which WILL have multi-addresses, the default is 'doemstic' only
    """
    
    temp = ocod_data[~ocod_data[class_type].isin(classes)]

    ocod_data = pd.concat([ocod_data[ocod_data[class_type].isin(classes)], temp.drop_duplicates(subset ='title_number')]).sort_values(by = "unique_id")

    #when the unit type is a carparking space but the class is residential it means that the address is a residential property that explicitly mentions a car park
    ocod_data = ocod_data[~(ocod_data['unit_type'].str.contains(r"park|garage") & (ocod_data['class']=="residential"))]

    
    return ocod_data