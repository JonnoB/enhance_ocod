
#the functions used by the empty homes project python notebooks
import json
import re
import time
import pandas as pd
import numpy as np

##
## Removing overlaps section
##
def is_overlap_function_tuples(a,b):
    ##This version uses tuples and is adapted to create a spacy format

    #identifies any overlapping tags.
    #takes two dictionaries and outputs a logical value identifying if the tags overlap
     start_overlap = (b[0] >= a[0]) & (b[0] <= a[1]) 
     end_overlap = (b[1] >= a[0]) & (b[1] <= a[1]) 
     return (start_overlap | end_overlap)

    
def remove_overlapping_spans_tuples(list_of_labels_dict):
    ##This version uses tuples and is adapted to create a spacy format

    #This function iterates through the list pairwise checking to see if there are overlaps
    #this function builds on the previous version which checked for overlaps in one go
    #but didn't consider that a single entity might overlap several subsequent entities
    #example A (start = 0, end = 20), B (start = 3, end = 10), C (start = 12, end = 20)
    #A overlaps B and C even though B and C do not overlap, as a result only A should remain
    
    i = 0
    j = i+1
    while j < len(list_of_labels_dict):
        
        
        #check if there is overlap
        pair_overlaps = is_overlap_function_tuples(list_of_labels_dict[i],list_of_labels_dict[j])
        if pair_overlaps:
            #get index of smallest span
            span_a = list_of_labels_dict[i][1] - list_of_labels_dict[i][0]
            span_b = list_of_labels_dict[j][1] - list_of_labels_dict[j][0]

            if span_a > span_b:
                list_of_labels_dict.pop(j)
            else:
                list_of_labels_dict.pop(i)
        else:
            #i and j are only updated here as if either is removed the indexing changed such that the original i and j values now represent differnt elements
            #only when there is no overlap should the indexes advance
            i = i +1
            j = i +1
        
    return list_of_labels_dict


def is_overlap_function(a,b):
    #identifies any overlapping tags.
    #takes two dictionaries and outputs a logical value identifying if the tags overlap
     start_overlap = (b['start'] >= a['start']) & (b['start'] <= a['end']) 
     end_overlap = (b['end'] >= a['start']) & (b['end'] <= a['end']) 
     return (start_overlap | end_overlap)

    
def remove_overlapping_spans2(list_of_labels_dict):
    #This function iterates through the list pairwise checking to see if there are overlaps
    #this function builds on the previous version which checked for overlaps in one go
    #but didn't consider that a single entity might overlap several subsequent entities
    #example A (start = 0, end = 20), B (start = 3, end = 10), C (start = 12, end = 20)
    #A overlaps B and C even though B and C do not overlap, as a result only A should remain
    
    i = 0
    j = i+1
    while j < len(list_of_labels_dict):
        
        
        #check if there is overlap
        pair_overlaps = is_overlap_function(list_of_labels_dict[i],list_of_labels_dict[j])
        if pair_overlaps:
            #get index of smallest span
            span_a = list_of_labels_dict[i]['end'] - list_of_labels_dict[i]['start']
            span_b = list_of_labels_dict[j]['end'] - list_of_labels_dict[j]['start']

            if span_a > span_b:
                list_of_labels_dict.pop(j)
            else:
                list_of_labels_dict.pop(i)
        else:
            #i and j are only updated here as if either is removed the indexing changed such that the original i and j values now represent differnt elements
            #only when there is no overlap should the indexes advance
            i = i +1
            j = i +1
        
    return list_of_labels_dict

##
##
## Expanding the addresses
##

def load_cleaned_labels(file_path):
    """
    Used when the json file being loaded has already had overlaps removed using the json overlap removing
    code from the unit tagging and span cleaning notebook
    
    N.B.
    This function is clear and intemediary function and I would like to remove it when possible
    This function is not neccessary when the denoising process has taken place

    """
    

    with open(file_path, "r") as read_file:
          all_entities_json = json.load(read_file)

    all_entities = pd.json_normalize(all_entities_json, record_path = "labels", meta= ['datapoint_id', 'text'], meta_prefix = "meta.")

    all_entities = all_entities.rename(columns = {'text':'label_text',
    'meta.datapoint_id':'datapoint_id',
    'meta.text':'text'})

    all_entities = all_entities.sort_values(['datapoint_id', 'start'])

    all_entities['label_id_count'] = all_entities.groupby(['datapoint_id', 'label']).cumcount()
    
    return all_entities

def remove_overlaps_jonno(x):
    
    """
    This is my version of the overlap removal. It works but is quite slow.
    Harry from Humanloop made a different version. Mine was at one point faster, not sure now.
    This function only needs to be used if denoising process has not been used
    """
    
    #this functions is modified from 
    #https://stackoverflow.com/questions/57804145/combining-rows-with-overlapping-time-periods-in-a-pandas-dataframe
    x = x.copy()
    #create a unique label, this is used for joining the data back on 
    #and removes a reliance on the data being pre-sorted
    x['unique_label'] = [*range(0,x.shape[0])]
    #get the size of the spans
    x['diff'] = (x['end']-x['start'])

    
    startdf = pd.DataFrame({'position':x['start'], 'unique_label':x['unique_label'], 'what':1})
    enddf = pd.DataFrame({'position':x['end'], 'unique_label':x['unique_label'], 'what':-1})
    mergdf = pd.concat([startdf, enddf]).sort_values('position')
    mergdf['running'] = mergdf['what'].cumsum()
    mergdf['newwin'] = mergdf['running'].eq(1) & mergdf['what'].eq(1)
    mergdf['group'] = mergdf['newwin'].cumsum()
    
    #merge back on using uniqe label to ensure correct ordering
    x = x.merge(mergdf.loc[mergdf['what'].eq(1),['unique_label','group']], how = 'left', on = 'unique_label')
    #sort within group and keep only the largest
    x = x.sort_values('diff', ascending=False).groupby(['group', 'datapoint_id'], as_index=False).first()

    x.drop(['diff', 'unique_label', 'group'], axis = 1, inplace = True)
    
    x.reset_index(drop = True, inplace = True)

    return(x)


def load_data_with_overlaps_jonno(file_path):
    
    """
    This function is used to load a json file that contains overlapping labels
    It uses my version of the overlap remover
    
    """
    
    print('loading json')
    with open(file_path, "r") as read_file:
        all_entities_json = json.load(read_file)
    
    print('pre-processing data')

    all_entities = pd.json_normalize(
        all_entities_json["datapoints"],
        record_path=["programmatic", "results"],
        meta=["data", "id"],
        #record_prefix="result_stuff.",
        meta_prefix="data_stuff.",
        errors="ignore",
    )
    all_entities = all_entities.rename(columns = {'data_stuff.id':'datapoint_id',
                                                 'text':'label_text'})

    all_entities["text"] = all_entities["data_stuff.data"].map(lambda x: x["text"])

    #all_entities.drop(['data_stuff.data'], axis = 1, inplace = True)

    all_entities = all_entities.sort_values(['datapoint_id', 'start'])

    all_entities['label_id_count'] = all_entities.groupby(['datapoint_id', 'label']).cumcount()

    all_entities.drop(columns = ['data_stuff.data'], inplace = True)
    
    print('renmoving overlaps this takes time')
    all_entities = all_entities.groupby(['datapoint_id']).apply(remove_overlaps_jonno)

    all_entities.reset_index(drop = True, inplace = True)
    
    return all_entities

def remove_overlaps_harry(
    spans: pd.DataFrame, groupby="datapoint_id", start="start", end="end"
):
    """
    This function was made by Harry from Humanloop to deal with the overlaps problem
    Removes rows from a DataFrame where start:end overlap.

    Attempts to keep the longer of the overlapping spans.
    """
    spans_to_remove = []
    for datapoint_id, datapoint_spans in spans.groupby(groupby):
        intervals = datapoint_spans.apply(
            lambda x: pd.Interval(left=getattr(x, start), right=getattr(x, end)), axis=1
        )
        for i, (index_a, interval_a) in enumerate(intervals.iteritems()):
            for j, (index_b, interval_b) in enumerate(
                intervals.iloc[i + 1 :].iteritems()
            ):
                if interval_a.overlaps(interval_b):
                    # print(i, j, index_a, index_b, interval_a, interval_b)
                    # Overlapping ground truths at index_a and index_b.
                    # Keep only longer of the two.
                    if interval_a.length >= interval_b.length:
                        spans_to_remove.append(index_b)
                    else:
                        spans_to_remove.append(index_a)

    return spans[~spans.index.isin(spans_to_remove)]


def load_data_with_overlaps_harry(file_path):
    
    """
    This is Harry from HUmanloop's of the overlap removal.
    This function only needs to be used if denoising process has not been used
    """

    with open(file_path, "r") as f:
        data = json.load(f)

    df = pd.json_normalize(
        data["datapoints"],
        # record_path=["programmatic", "aggregateResults"],
        record_path=["programmatic", "results"],
        meta=["data", "id"],
        record_prefix="result_stuff.",
        meta_prefix="data_stuff.",
        errors="ignore",
    )
    df["full_text"] = df["data_stuff.data"].map(lambda x: x["text"])

    df = all_entities.copy()



    new_df = remove_overlaps_harry(
        df, groupby="data_stuff.id", start="result_stuff.start", end="result_stuff.end"
    )

    all_entities = all_entities.sort_values(['datapoint_id', 'start'])

    all_entities.reset_index(drop = True, inplace = True)
    
    return new_df

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
        
        if i%print_every==0: print("i=", i, " expand time,"+ str(round(expand_time, 3)) +
                           " filter time" + str(round(filter_time,3)) + 
                           " make_dataframe_time " + str(round(make_dataframe_time,3)))
    
    #once all the lines have been expanded concatenate them into a single dataframe
    start_concat_time = time.time()
    out = pd.concat(temp_list)
    end_concat_time = time.time

    return out


def clean_street_numbers(df, original_column = 'number_or_name'):
    #This function is primarily designed to clean the street numbers of the voa dataset

    temp = df
        #remove anything in brackets
    temp['street_number'] = temp[original_column].str.replace(r"\(.+\)", "", regex = True, case = False)
    #there are a few + sy,bols used mostly in relation to westfield shopping centeres these are replaced with space
    temp['street_number'] = temp['street_number'].str.replace("+", r" ", regex = True, case = False)
    
    #     #units often slip in as street numbers, this kills them off
    temp.loc[temp['street_number'].str.contains(r"unit|suite|room", regex = True, case = False)==True, 'street_number'] = np.nan
    
    #     #replace @ and & with words
    temp['street_number'] = temp['street_number'].str.replace(r"@", " at ", regex = True, case = False).str.replace(r"&", " and ", regex = True, case = False)
    
    #     #replace "-" with spaces with a simple "-"
    temp['street_number'] = temp['street_number'].str.replace(r"(\s)?-(\s)?", "-", regex = True, case = False)
    
    #     #take only things after the last space includes cases where there is no space. Then remove all letters
    temp['street_number'] = temp['street_number'].str.extract(r"([^\s]+$)")[0].str.replace(r"([a-z]+)", "", regex = True, case = False)
    #     #remove dangling hyphens and slashes
    temp['street_number'] = temp['street_number'].str.replace(r"(-$)|(^-)|\\|\/", "", regex = True, case = False)
    temp['street_number'] = temp['street_number'].str.replace(r"\\|\/", " ", regex = True, case = False)
    #     #replace double hyphen... yes it happens
    temp['street_number'] = temp['street_number'].str.replace(r"--", r"-", regex = True, case = False)

    #
    temp.loc[temp['street_number'].str.len() == 0, 'street_number'] = np.nan
    #some names are simply a string of hyphentated words, this would leave a single - which would cause a crash, this removes those
    temp.loc[temp['street_number']=="-", 'street_number'] = np.nan
    
    return(temp)
    
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

    temp.loc[temp['street_number'].str.contains(r"\.", regex = True)==True, 'street_number'] = np.nan

    temp['is_multi'] = temp['street_number'].str.contains(r"-", regex = True)
    
    temp.rename(columns = {'full_property_identifier': 'business_address'}, inplace = True)

    temp_multi = temp.loc[temp['is_multi']==True]#, ['street_number', 'is_multi']]

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

def fuzzy_street_match(ocod_data, voa_data, target_lad):
##
## It's possible that this entire thing can be replaced with a simple isin() type command.
## I will be able to do the replacement if matching is basically bianry
##
    #filters to a single LAD
    #removes advertising hoardings which are irrelevant
    LAD_biz = voa_businesses.loc[(voa_data['lad11cd']==target_lad)].copy(deep = True)
    
    LAD_biz.loc[:,'street_name2'] = LAD_biz['street'].copy(deep=True)
    #remove apostraphe's
    LAD_biz.loc[:,'street_name2'] = LAD_biz.loc[:,'street_name2'].str.replace(r"'", "", regex = True).str.replace(r"s(?=\s)", "", regex = True).str.replace(r"\s(?=way|gate)", "", regex = True)
    
    #subset to target LAD
    ocod_data_road = ocod_data[ocod_data['lad11cd']==target_lad].copy(deep = True)
    #replace nan values to prevent crash
    #ocod_data_road['street_name'] = 
    #ocod_data_road['street_name'].fillna("xxxstreet name missingxxx")
    
    ocod_data_road.loc[ocod_data_road.street_name.isna(),'street_name'] ="xxxstreet name missingxxx"
    
    #create second column
    ocod_data_road['street_name2'] = ocod_data_road['street_name'].copy(deep=True)
    #remove apostraphe's
    ocod_data_road['street_name2'] = ocod_data_road['street_name2'].str.replace(r"'", "", regex = True).str.replace(r"s(?=\s)", "", regex = True).str.replace(r"\s(?=way|gate)", "", regex = True)
    #remove trailing 's'
    #ocod_data_road['street_name2'] = ocod_data_road['street_name2'].str.remove(r"s(?=\s)")
    #remove space preceeding 'way' or 'gate'
    #ocod_data_road['street_name2'] = ocod_data_road['street_name2'].str.remove(r"\s(?=way|gate)")
    
    
    #using the below line cause a copy warning
    # ocod_data_road.loc[:,'street_name'].fillna("xxxstreet name missingxxx", inplace = False)

    #extract unique names
    unique_ocod_names = ocod_data_road.street_name2.unique()
    unique_voa_names = LAD_biz.street_name2.unique()

    #fuzzy match unique names
    street_matches_df = pd.DataFrame([process.extractOne(x, unique_voa_names) for x in unique_ocod_names], 
                                 columns = ['matched_road_name', 'similarity'])
    street_matches_df['street_name2'] = unique_ocod_names

    out = ocod_data_road.merge(street_matches_df, left_on = "street_name2", right_on = "street_name2")
    #out.loc[ocod_data_road.street_name == "xxxstreet name missingxxx",'street_name'] = None
    #remove the modified street name
    #out = out.drop('street_name2', axis = 1)
    return(out)


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

