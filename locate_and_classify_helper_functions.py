
#the functions used by the empty homes project python notebooks
import io
import os
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
            postcode_district_lookup = pd.read_csv(f)[['pcds', 'oslaua', 'oa11', 'lsoa11', 'msoa11', 'ctry']]
            
            # Filter for English and Welsh postcodes
            postcode_district_lookup = postcode_district_lookup[
                (postcode_district_lookup['ctry'] == 'E92000001') | 
                (postcode_district_lookup['ctry'] == 'W92000004')
            ]
            
            # Rename columns
            postcode_district_lookup.rename(columns={
                'pcds': 'postcode2',
                'oslaua': 'lad11cd',
                'oa11': 'oa11cd',
                'lsoa11': 'lsoa11cd',
                'msoa11': 'msoa11cd'
            }, inplace=True)
            
            # Remove spaces from postcodes
            postcode_district_lookup['postcode2'] = (
                postcode_district_lookup['postcode2']
                .str.lower()
                .str.replace(r"\s", r"", regex=True)
            )
            
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
    
    """

    # Remove anything in brackets
    df['street_number'] = df[original_column].str.replace(r"\(.+\)", "", regex=True, case=False)
    
    # Replace + symbols with space
    df['street_number'] = df['street_number'].str.replace(r"\+", " ", regex=True, case=False)
    
    # Remove units/suite/room entries
    df.loc[df['street_number'].str.contains(r"unit|suite|room", regex=True, case=False, na=False), 'street_number'] = np.nan
    
    # Replace @ and & with words
    df['street_number'] = df['street_number'].str.replace(r"@", " at ", regex=True, case=False)
    df['street_number'] = df['street_number'].str.replace(r"&", " and ", regex=True, case=False)
    
    # Replace hyphens with spaces around them with simple hyphen
    df['street_number'] = df['street_number'].str.replace(r"\s*-\s*", "-", regex=True, case=False)
    
    # Extract last word and remove letters
    df['street_number'] = df['street_number'].str.extract(r"([^\s]+)$")[0]
    df['street_number'] = df['street_number'].str.replace(r"[a-zA-Z]+", "", regex=True, case=False)
    
    # Remove dangling hyphens and slashes - FIXED REGEX
    df['street_number'] = df['street_number'].str.replace(r"^-+|-+$", "", regex=True, case=False)
    df['street_number'] = df['street_number'].str.replace(r"[\\\/]", " ", regex=True, case=False)
    
    # Replace double hyphens
    df['street_number'] = df['street_number'].str.replace(r"-{2,}", "-", regex=True, case=False)
    
    # Clean up empty strings and single hyphens
    df.loc[df['street_number'].str.len() == 0, 'street_number'] = np.nan
    df.loc[df['street_number'] == "-", 'street_number'] = np.nan
    
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
    
    ocod_data_road['address_match'] = ocod_data_road['business_address'].notna()==True
    
    ocod_data_road.loc[ocod_data_road['street_name2']== "xxxstreet name missingxxx",'street_name2'] = np.nan
    
     
    
    return(ocod_data_road)
    
    
def find_filter_type(street_num):
 #gets the highest street number and uses it to work out if the property is on the odd or even side of the street, or if that rule is ignore and it is all numbers
    values = [int(x) for x in street_num.split("-")]
    if (max(values)%2==0) & (min(values)%2==0):
        out = "even"
    elif (max(values)%2==1) & (min(values)%2==1):
        out = "odd"
    else:
        out = "all"
    
    return out



def create_all_street_addresses(voa_businesses, target_lad, return_columns = ['street_name2', 'street_number', 'business_address']):
    
    #creates a two column table where the first column is the street name and the second
    #column is the street number. The function expands address to get all numbers between for example 4-22
    #voa_businesses is a dataframe of the voa business listings and ratings dataset
    #target_lad is the ons code identifying which local authority will be used. 
    
    temp = voa_businesses[voa_businesses['lad11cd'] == target_lad].copy(deep = True)

    ##
    ## The below commented block has been replaces as street numbers are now created at dataloading
    ##

    
    # #remove anything in brackets
    # temp['street_number'] = temp['street_number'].str.replace(r"\(.+\)", "", regex = True, case = False)
    
    # #units often slip in as street numbers, this kills them off
    # temp.loc[temp['street_number'].str.contains(r"unit|suite", regex = True, case = False)==True, 'street_number'] = np.nan
    
    # #replace @ and & with words
    # temp['street_number'] = temp['street_number'].str.replace(r"@", " at ", regex = True, case = False).str.replace(r"&", " and ", regex = True, case = False)
    
    # #replace "-" with spaces with a simple "-"
    # temp['street_number'] = temp['street_number'].str.replace(r"(\s)?-(\s)?", "-", regex = True, case = False)
    
    # #take only things after the last space includes cases where there is no space. Then remove all letters
    # temp['street_number'] = temp['street_number'].str.extract(r"([^\s]+$)")[0].str.replace(r"([a-z]+)", "", regex = True, case = False)
    # #remove dangling hyphens and slashes
    # temp['street_number'] = temp['street_number'].str.replace(r"(-$)|(^-)|\\|\/", "", regex = True, case = False)
    # #replace double hyphen... yes it happens
    # temp['street_number'] = temp['street_number'].str.replace(r"--", r"-", regex = True, case = False)
    # temp.loc[temp['street_number'].str.len() == 0, 'street_number'] = np.nan
    temp.loc[temp['street_number'].str.contains(r"\.", regex = True)==True, 'street_number'] = np.nan

    temp['is_multi'] = temp['street_number'].str.contains(r"-", regex = True)
    
    temp.rename(columns = {'full_property_identifier': 'business_address'}, inplace = True)

    temp_multi = temp.loc[temp['is_multi']==True].copy()

    filter_types = [find_filter_type(x) for x in temp_multi['street_number']]
    
    temp_multi['number_filter'] = filter_types
    
    #occasionally one of the dataframes is empty, this causes a error with the concatenation
    #this if statement gets around that
    not_multi_address = temp.loc[temp['is_multi']==False]
    
    
    temp_multi = temp_multi.reset_index()

    if (temp_multi.shape[0]>0) & (not_multi_address.shape[0]>0):
        
        #expand the dataframe according to the correct rules
        temp_multi = expand_dataframe_numbers(temp_multi, 'street_number', print_every = 10000, min_count = 1)
        street_address_lookup = pd.concat([temp_multi, not_multi_address])[return_columns]
        
    elif (temp_multi.shape[0]==0) & (not_multi_address.shape[0]>0):
        street_address_lookup = not_multi_address[return_columns]
    
    elif (temp_multi.shape[0]>0) & (not_multi_address.shape[0]==0):
        temp_multi = expand_dataframe_numbers(temp_multi, 'street_number', print_every = 10000, min_count = 1)
        street_address_lookup =temp_multi[return_columns]
    else:
        #if there are no street addresses at all then
        
        street_address_lookup = temp_multi[return_columns]
        
    #i = 0
    #temp_list = []
    #for x in temp_multi['street_number'].unique():
    #   # print(i)
    #    i = i+1
    #    values = [int(x) for x in [x for x in x.split("-")]]
    #    temp_list = temp_list +[values]

    #street_address_lookup = pd.concat([temp_multi, temp.loc[temp['is_multi']==False]])[['street_name2', 'street_number']]
    return(street_address_lookup)
    
    
    
    
 ##
 ## The Fuzzy and massaged street match were an intermediary stage of development where I needed to check out street matching quality.
 ## they have both been superceded by the massaged_address_match function
 ##

def massaged_street_match(ocod_data, voa_data, target_lad):
##
## This exact match works pretty much as well as the fuzzy matcher but is much faster and clearer
##
    #filters to a single LAD
    #removes advertising hoardings which are irrelevant
    LAD_biz = voa_data.loc[(voa_data['lad11cd']==target_lad)].copy(deep = True)
    
    LAD_biz.loc[:,'street_name2'] = LAD_biz['street'].copy(deep=True)
    #remove apostraphe's
    LAD_biz.loc[:,'street_name2'] = LAD_biz.loc[:,'street_name2'].str.replace(r"'", "", regex = True).\
    str.replace(r"s(s)?(?=\s)", "", regex = True).str.replace(r"\s", "", regex = True)
    
    #subset to target LAD
    ocod_data_road = ocod_data[ocod_data['lad11cd']==target_lad].copy(deep = True)
    #replace nan values to prevent crash    
    
    #create second column
    ocod_data_road['street_name2'] = ocod_data_road['street_name'].copy(deep=True)
    
    #replace nan values to prevent crash    
    ocod_data_road.loc[ocod_data_road.street_name.isna(),'street_name2'] ="xxxstreet name missingxxx"
    #clean street names of common matching errors
    #remove apostraphe's
    #remove trailing 's'
    #remove all spaces
    ocod_data_road['street_name2'] = ocod_data_road['street_name2'].str.replace(r"'", "", regex = True).\
    str.replace(r"s(s)?(?=\s)", "", regex = True).str.replace(r"\s", "", regex = True)
    
    ocod_data_road['match'] = ocod_data_road['street_name2'].isin(LAD_biz.street_name2.unique())

    return(ocod_data_road)

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
    This allows businesses to be identified, and also missing lsoa to be added via the street name.
    Can handle both direct CSV files and ZIP files (will automatically find and use the largest file in the ZIP).
    """

    def find_largest_file(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            largest_file = max(zip_ref.infolist(), key=lambda x: x.file_size)
            return largest_file.filename

    VOA_headers_raw = ["Incrementing Entry Number", "Billing Authority Code", "NDR Community Code", 
     "BA Reference Number", "Primary And Secondary Description Code", "Primary Description Text",
    "Unique Address Reference Number UARN", "Full Property Identifier", "Firms Name", "Number Or Name",
    "Street", "Town", "Postal District", "County", "Postcode", "Effective Date", "Composite Indicator",
     "Rateable Value", "Appeal Settlement Code", "Assessment Reference", "List Alteration Date", "SCAT Code And Suffix",
     "Sub Street level 3", "Sub Street level 2", "Sub Street level 1", "Case Number", 
     "Current From Date", "Current To Date", 
    ]

    # Set to lower and replace spaces with underscore to turn the names into appropriate column names
    VOA_headers = [x.lower().replace(" ", "_") for x in VOA_headers_raw]
    
    # Check if the file is a ZIP file
    if file_path.lower().endswith('.zip'):
        # Handle ZIP file
        largest_file = find_largest_file(file_path)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            with zip_ref.open(largest_file) as csv_file:
                voa_businesses = pd.read_csv(csv_file,
                           sep="*",
                           encoding_errors='ignore',
                           header=None,
                           names=VOA_headers,
                           index_col=False,
                           )
    else:
        # Handle direct CSV file
        voa_businesses = pd.read_csv(file_path,
                       sep="*",
                       encoding_errors='ignore',
                       header=None,
                       names=VOA_headers,
                       index_col=False,
                       )

    voa_businesses['postcode'] = voa_businesses['postcode'].str.lower()
    voa_businesses['street'] = voa_businesses['street'].str.lower()

    voa_businesses['street_name2'] = voa_businesses.loc[:,'street'].str.replace(r"'", "", regex=True).\
        str.replace(r"s(s)?(?=\s)", "", regex=True).str.replace(r"\s", "", regex=True)

    # This removes advertising hordings which are irrelevant
    voa_businesses = voa_businesses.loc[voa_businesses['primary_description_text'].str.contains("ADVERTISING")==False,:]
    # Remove several kinds of car parking space
    voa_businesses = voa_businesses[~voa_businesses['primary_and_secondary_description_code'].isin(['C0', 'CP','CP1', 'CX', 'MX'])]
    
    ##
    ## Warning this removes a large amount of columns, these may be interesting for some people
    ##
    voa_businesses = voa_businesses.iloc[:,4:15]
    
    # Extract the street number
    # Replace unit numbers with nothing to avoid accidentally using them as street numbers
    voa_businesses = clean_street_numbers(voa_businesses, original_column='number_or_name')

    # Westfield has a ludicrous numbering system and is removed to avoid issues
    voa_businesses.loc[voa_businesses['full_property_identifier'].str.contains('WESTFIELD SHOPPING CENTRE')==True, 'street_number'] = np.nan
    # Sometimes a phone number ends up in the address causing havoc, these are few (145) and are ignored
    voa_businesses.loc[voa_businesses['street_number'].str.contains(r'-\d+-', regex=True)==True, 'street_number'] = np.nan

    voa_businesses['postcode2'] = voa_businesses['postcode'].str.lower().str.replace("\s", "", regex=True)
    # Extracts the name of any "house" buildings
    voa_businesses['building_name'] = voa_businesses['number_or_name'].str.lower().str.extract(r'((?<!\S)(?:(?!\b(?:\)|\(|r\/o|floor|floors|pt|and|annexe|room|gf|south|north|east|west|at|on|in|of|adjoining|adj|basement|bsmt|fl|flr|flrs|wing)\b)[^\n\d])*? house\b)')

    # Add in postcode data and LSOA etc data, this is useful for a range of tasks
    voa_businesses = voa_businesses.merge(postcode_district_lookup, left_on='postcode2', right_on="postcode2")

    voa_businesses['street_name2'] = voa_businesses.loc[:,'street'].str.replace(r"'", "", regex=True).\
        str.replace(r"s(s)?(?=\s)", "", regex=True).str.replace(r"\s", "", regex=True)

    return voa_businesses

def street_and_building_matching(ocod_data, price_paid_df, voa_businesses):
    
    """
    Where there is no postcode properties are located using the building name or the street name
    
    This process is quite convoluted and there is certainly a more efficient and pythonic way
    however the order within each filling method is important to ensure that there are no duplicates
    as this causes the OCOD dataset to grow with duplicates
    
    """
    
    #replce the missing lsoa using street matching
    print('replace the missing lsoa using street matching')

    temp_lsoa = pd.concat([
        price_paid_df[['street_name2', 'lad11cd', 'lsoa11cd']], voa_businesses[['street_name2', 'lad11cd', 'lsoa11cd']]
                        ]).dropna(axis = 0, how = 'any', inplace = False)

    temp = temp_lsoa.groupby(['street_name2', 'lad11cd', 'lsoa11cd']).size().reset_index().groupby(['street_name2', 'lad11cd']).size()\
    .reset_index().rename(columns = {0:'counts'})

    temp = temp[temp['counts']==1].merge(temp_lsoa.drop_duplicates(), 
                                        how = "left", on = ['street_name2', 'lad11cd']).rename(columns ={'lsoa11cd':'lsoa_street'})

    ocod_data = ocod_data.merge(temp[['lsoa_street', "street_name2", "lad11cd"]], how = "left", on = ['street_name2', 'lad11cd'])


    #replace the missing lsoa using building matching

    print('replace the missing lsoa using building matching')

    temp = price_paid_df.copy()

    temp['paon'] = temp['paon'].str.replace(r"\d+", "", regex = True).replace(r"and|\&|-|,", "", regex = True).replace(r"^ +| +$", r"", regex=True)

    temp = temp.groupby(['paon', 'lad11cd', 'lsoa11cd']).size().reset_index()

    temp = temp[temp['paon'].str.len()!=0].groupby(['paon', 'lad11cd']).size().reset_index().rename(columns = {0:'counts'})

    pp_temp = price_paid_df[['paon', 'lad11cd','oa11cd' , 'lsoa11cd']].copy()

    pp_temp['paon'] = pp_temp['paon'].str.replace(r"\d+", "", regex = True).replace(r"and|\&|-|,", "", regex = True).replace(r"^ +| +$", r"", regex=True)

    pp_temp = pp_temp.drop_duplicates()

    temp = temp[temp['counts']==1].merge(pp_temp, 
                                        how = "left", on = ['paon', 'lad11cd'])

    temp = temp.rename(columns ={'lsoa11cd':'lsoa_building',
                                'oa11cd':'oa_building',
                                'paon':'building_name'}).drop_duplicates(subset = ['building_name', 'lsoa_building'])

    ocod_data = ocod_data.merge(temp[['lsoa_building', "oa_building" ,"building_name", "lad11cd"]], 
                        how = "left", 
                        on =  ['building_name', 'lad11cd'])

    ocod_data = ocod_data.merge(voa_businesses.loc[~voa_businesses['building_name'].isna(), 
                    ['building_name','oa11cd', 'lsoa11cd', 'lad11cd']].drop_duplicates(subset = ['building_name', 'lsoa11cd']).rename(
        columns = {'lsoa11cd':'lsoa_busi_building',
                'oa11cd':'oa_busi_building'}),
                        how = "left", 
                        on =  ['building_name', 'lad11cd'])


    #fill in the lsoa11cd using the newly ID'd lsoa from the matching process
    #then repeat with the OA

    print("insert newly ID'd LSOA and OA")

    #I appreciate this is an almost artistically silly way of doing this. 
    for x in ['lsoa_street', 'lsoa_building', 'lsoa_busi_building']:#, 'lsoa_business']:
        ocod_data.loc[ocod_data['lsoa11cd'].isnull(), 'lsoa11cd'] = ocod_data[x][ocod_data['lsoa11cd'].isnull()] 
        
    #add in the OA for when building matches have occured
    for x in ['oa_building', 'oa_busi_building']:#, 'lsoa_business']:
        ocod_data.loc[ocod_data['oa11cd'].isnull(), 'oa11cd'] = ocod_data[x][ocod_data['oa11cd'].isnull()] 

    print("update missing LSOA and OA for nested properties where at least one nested property has an OA or LSOA")
    #after all other lsoa adding methods are completed
    #all nested properties with missing lsoa have the lsoa of the other properties within their group added

    temp = ocod_data.loc[(ocod_data['lsoa11cd'].notnull()) & (ocod_data['within_larger_title']==True) ,['lsoa11cd', 'title_number']].\
    groupby(['lsoa11cd', 'title_number']).size().reset_index()
    temp = temp[['lsoa11cd', 'title_number']].rename(columns = {'lsoa11cd':'lsoa_nested'})

    #there are a small number of nested addresses where there are multiple lsoa this prevents increasing the number of observations with these duplicates
    #I don't think it matters if a ver observations are in neighbouring lsoa, the general spatial coherence is maintained
    temp = temp.groupby('title_number')['lsoa_nested'].first().reset_index()


    ocod_data = ocod_data.merge(temp[['title_number', 'lsoa_nested']], 
                        how = "left",
                            on = "title_number")

    ocod_data.loc[ocod_data['lsoa11cd'].isnull(), 'lsoa11cd'] = ocod_data['lsoa_nested'][ocod_data['lsoa11cd'].isnull()] 

    ##
    ##repeats again but for oa instead of lsoa
    ##

    temp = ocod_data.loc[(ocod_data['oa11cd'].notnull()) & (ocod_data['within_larger_title']==True) ,['oa11cd', 'title_number']].\
    groupby(['oa11cd', 'title_number']).size().reset_index()
    temp = temp[['oa11cd', 'title_number']].rename(columns = {'oa11cd':'oa_nested'})

    #there are a small number of nested addresses where there are multiple lsoa this prevents increasing the number of observations with these duplicates
    #I don't think it matters if a ver observations are in neighbouring lsoa, the general spatial coherence is maintained
    temp = temp.groupby('title_number')['oa_nested'].first().reset_index()


    ocod_data = ocod_data.merge(temp[['title_number', 'oa_nested']], 
                        how = "left",
                            on = "title_number")

    ocod_data.loc[ocod_data['oa11cd'].isnull(), 'oa11cd'] = ocod_data['oa_nested'][ocod_data['oa11cd'].isnull()] 

    return ocod_data

def substreet_matching(ocod_data, price_paid_df, voa_businesses, print_lads = False, print_every = 100):
    """"
    Some streets are on the boundary of LSOA this section uses the street number to match to the nearest lsoa.
    """
    filled_lsoa_list = []
    i = 1
    unique_lad_codes = ocod_data[ocod_data['street_name'].notnull() & ocod_data['street_number'].notnull() & ocod_data['lsoa11cd'].isnull()]['lad11cd'].unique()

    for target_lad in unique_lad_codes:
        if print_lads: print(target_lad)
            
        if i%print_every==0: print("lad ", i, " of "+ str(round(len(unique_lad_codes), 3)))
        i = i+1
        
        #subset to the relevat rows within a single lad
        missing_lsoa_df = ocod_data[ocod_data['street_name'].notnull() & ocod_data['street_number'].notnull() & ocod_data['lsoa11cd'].isnull() & (ocod_data['lad11cd']==target_lad)].copy()
        missing_lsoa_df.loc[:,'street_number2'] = missing_lsoa_df.loc[:,'street_number'].str.replace(r"^.*(?=\b[0-9]+$)", "", regex = True).str.replace(r"[^\d]", "", regex = True)

        target_street_names = missing_lsoa_df['street_name2'].unique()

        temp_lsoa = pd.concat([
            #the price paid data with names changed
            price_paid_df[price_paid_df['street_name2'].isin(target_street_names )  & 
                                        (price_paid_df['lad11cd']==target_lad) ], 
        #voa data added in                
                        voa_businesses[(voa_businesses['lad11cd']==target_lad)]]
                                                        )[['street_name2', 'street_number', 'lsoa11cd', 'lad11cd']].dropna(axis = 0, how = 'any', inplace = False)

        temp_lsoa.loc[:,'street_number2'] = temp_lsoa.loc[:,'street_number'].str.replace(r"^.*(?=\b[0-9]+$)", "", regex = True).str.replace(r"[^\d]", "", regex = True)
        
        temp_lsoa  = create_all_street_addresses(temp_lsoa[temp_lsoa['street_name2'].isin(target_street_names ) & 
                                                temp_lsoa['street_number2'].notnull()], 
                                        target_lad, 
                                        ['street_name2', 'street_number2', 'lsoa11cd'])
        

        for target_road in target_street_names:
            #print(target_road)
            missing_lsoa_road = missing_lsoa_df[missing_lsoa_df['street_name2']== target_road ].copy()
            temp_road = temp_lsoa[temp_lsoa['street_name2'] ==target_road ]

            if len(temp_road)>0:
                missing_lsoa_road['lsoa11cd'] = [street_number_to_lsoa(temp_road, int(missing_lsoa_road.iloc[missing_lsoa_row]['street_number2'])) 
                                                for missing_lsoa_row 
                                                in range(0, len(missing_lsoa_road))]
                filled_lsoa_list = filled_lsoa_list + [missing_lsoa_road]
    
    #if the temp_lso is returned as empty then
    #then pd.concat crashes. to avoid this the following if statement is used
    #the if statment checks if the list is empty
    if not filled_lsoa_list:
        #join the list back together
        temp_lsoa = ocod_data[0:0]
    else:
        temp_lsoa = pd.concat(filled_lsoa_list)
  

    #join the ocod dataset backtogether
    ocod_data = pd.concat([ocod_data[~ocod_data['unique_id'].isin(temp_lsoa['unique_id'])], temp_lsoa ] )

    ##
    ##Fill in the missing data in the nested addresses again, where at least one address in the nested group has an lsoa/os
    ##
    #Doing this grouped nested business a second time pushes lsoa ID over 90% which seems good enough for me

    #after all other lsoa adding methods are completed
    #all nested properties with missing lsoa have the lsoa of the other properties within their group added

    temp = ocod_data.loc[(ocod_data['lsoa11cd'].notnull()) & (ocod_data['within_larger_title']==True) ,['lsoa11cd', 'title_number']].\
    groupby(['lsoa11cd', 'title_number']).size().reset_index()
    temp = temp[['lsoa11cd', 'title_number']].rename(columns = {'lsoa11cd':'lsoa_nested2'})

    #there are a small number of nested addresses where there are multiple lsoa this prevents increasing the number of observations with these duplicates
    #I don't think it matters if a ver observations are in neighbouring lsoa, the general spatial coherence is maintained
    temp = temp.groupby('title_number')['lsoa_nested2'].first().reset_index()


    ocod_data = ocod_data.merge(temp[['title_number', 'lsoa_nested2']], 
                        how = "left",
                            on = "title_number")

    ocod_data.loc[ocod_data['lsoa11cd'].isnull(), 'lsoa11cd'] = ocod_data['lsoa_nested2'][ocod_data['lsoa11cd'].isnull()] 

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

def voa_address_match_all_data(ocod_data, voa_businesses, print_lads = False, print_every =50):
    
    """
    Cycles through all addresses and attempts to match to the VOA database
    """
    all_lads = ocod_data.lad11cd.unique()

    matched_lads_list = []
    i = 0
    all_lads = [x for x in all_lads if str(x) != 'nan']
    #see which roads match a road in voa data set for each local authority
    for target_lad in all_lads:
        if print_lads: print(target_lad)
            
        if (i>0) & (i%print_every==0): print("address matched ", i, "lads of "+ str(round(len(all_lads), 3)))
        i = i+1
        #temp['matches_business_address'] = business_address_matcher(temp['street_name'], temp['street_number'], voa_businesses, target_lad)
        matched_lads_list = matched_lads_list + [massaged_address_match(ocod_data, voa_businesses, target_lad)]

        #matched_lads_list = [massaged_address_match(ocod_data, voa_businesses, target_lad) for target_lad in all_lads]
 
    ocod_data = pd.concat(matched_lads_list)
    
    return ocod_data


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
            ocod_data['address_match']==True,
            ocod_data['property_address'].str.contains(r"cinema|hotel|office|centre|\bpub|holiday(?:\s)?inn|travel lodge|travelodge|medical|business|cafe|^shop| shop|service|logistics|building supplies|restaurant|home|^store(?:s)?\b|^storage\b|company|ltd|limited|plc|retail|leisure|industrial|hall of|trading|commercial|technology|works|club,|advertising|school|church|(?:^room)", case = False), 
            ocod_data['property_address'].str.contains(r"^[a-z\s']+\b(?:land(?:s)?|plot(?:s)?)\b", case = False)==True, #land with words before it
            ocod_data['building_name'].str.contains(r'\binn$|public house|^the\s\w+\sand\s\w+|(?:tavern$)')==True, #pubs in various guises
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

    temp_fill = ocod_data[~ocod_data['class'].isin(['unknown', 'airspace', 'carpark']) & (ocod_data['within_larger_title']==True)].groupby(['title_number', 'class']).\
    size().reset_index()[['title_number', 'class']].drop_duplicates()

    temp = ocod_data[ocod_data['title_number'].isin(temp_fill['title_number']) & (ocod_data['class']=='unknown') & (ocod_data['within_larger_title']==True)].copy()

    temp.drop('class', axis = 1, inplace = True)

    temp = temp.merge(temp_fill, how = "left", on = "title_number")

    ocod_data = pd.concat([temp, ocod_data[~ocod_data['unique_id'].isin(temp['unique_id'])]])

    
    return ocod_data

def classification_type2(ocod_data):
    
    ocod_data['class2'] = np.select(
        [   (ocod_data['class']=='unknown') & ((ocod_data.property_address.str.contains('^(?:the )?unit') & ocod_data.property_address.str.contains('park', regex = True))==True), #contains the word unit and park
            (ocod_data['class']=='unknown') & (ocod_data['business_counts']==0), #if there are no businesses in the oa then it is a residential
            (ocod_data['class']=='unknown') & (ocod_data['lsoa_business_counts']==0), #if there are no businesses in the lsoa then it is a residential
            (ocod_data['class']=='unknown') & (ocod_data['street_match']==True) & (ocod_data['street_name'].notnull()==True) & (ocod_data['street_number'].notnull()==True),
            (ocod_data['class']=='unknown') & (ocod_data['street_match']==False) & (ocod_data['street_name'].notnull()==True),
            (ocod_data['class']=='unknown') & (ocod_data['building_name'].notnull()==True)

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

    temp_fill = ocod_data[~ocod_data['class2'].isin(['unknown', 'airspace', 'carpark']) & (ocod_data['within_larger_title']==True)].groupby(['title_number', 'class2']).\
    size().reset_index()[['title_number', 'class2']].drop_duplicates()

    temp = ocod_data[ocod_data['title_number'].isin(temp_fill['title_number']) & (ocod_data['class2']=='unknown') & (ocod_data['within_larger_title']==True)].copy()

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
