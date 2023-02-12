import json
import pandas as pd
import re
import numpy as np
import time
import spacy
#These are to modify the tokenization process
from spacy.lang.char_classes import LIST_ELLIPSES
from spacy.lang.char_classes import LIST_ICONS, HYPHENS
from spacy.lang.char_classes import CONCAT_QUOTES, ALPHA_LOWER, ALPHA_UPPER, ALPHA
from spacy.util import compile_infix_regex


#This  module is supposed to contain all the relevant functions for parsing the LabeledS json file 

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


##
##
##
##
##
##

def identify_multi_addresses(all_entities):
    
    """
    An important part of the parsing process is knowing which addresses represent multiple properties.
    This function takes the all entities dataframe and returns three lists containing the indexes of 
    the nested properties.
    multi_unit_id: The multi properties are units not entire properties aka flats
    multi_property: The properties are nested multiproperties but are not flats
    all_multi_ids: the combination of both.
    """
    
    xx_to_yy_regex = r'^\d+\s?(?:-|to)\s?\d+$'

    multi_check_df = all_entities[['datapoint_id', 'text', ]].drop_duplicates()
    multi_check_df['comma_count'] = multi_check_df['text'].str.count(',')
    multi_check_df['land'] = multi_check_df['text'].str.contains(r"^(?:land|plot|airspace|car|parking)", case = False)

    multi_check_df['business'] = multi_check_df['text'].str.contains(r"cinema|hotel|office|centre|\bpub|holiday\s?inn|travel\s?lodge|business|cafe|^shop| shop|restaurant|home|^stores?\b|^storage\b|company|ltd|limited|plc|retail|leisure|industrial|hall of|trading|commercial|works", case = False)

    temp_df = all_entities[['datapoint_id', 'label']].groupby(['datapoint_id', 'label']).value_counts().to_frame(name = "counts").reset_index().pivot(index = 'datapoint_id', columns = 'label', values = 'counts').fillna(0)

    xx_to_yy_street_counts = all_entities['datapoint_id'][all_entities['label_text'].str.contains(
        xx_to_yy_regex)& (all_entities['label']=="street_number")
                            ].to_frame(name = 'datapoint_id').groupby('datapoint_id').size().to_frame(name = 'xx_to_yy_street_counts')

    xx_to_yy_unit_counts = all_entities['datapoint_id'][all_entities['label_text'].str.contains(
        xx_to_yy_regex)& (all_entities['label']=="unit_id")
                            ].to_frame(name = 'datapoint_id').groupby('datapoint_id').size().to_frame(name = 'xx_to_yy_unit_counts')

    multi_check_df = multi_check_df.merge(temp_df, how = 'left', left_on = "datapoint_id", right_index = True).\
    merge(xx_to_yy_street_counts, how = 'left', left_on = "datapoint_id", right_index = True).\
    merge(xx_to_yy_unit_counts, how = 'left', left_on = "datapoint_id", right_index = True).fillna(0)


    del xx_to_yy_street_counts
    del xx_to_yy_unit_counts

    #separate the classes using logical rules
    multi_check_df['class'] = np.select(
        [
            multi_check_df['land']== True,
            multi_check_df['business']== True,
            (multi_check_df['building_name']==1) & (multi_check_df['unit_id'] == 0), #this has to go infront of 'multi_check_df['xx_to_yy_unit_counts']>0'
            multi_check_df['xx_to_yy_unit_counts']>0,
            multi_check_df['street_number']>1,
            multi_check_df['unit_id']>1,
            (multi_check_df['street_number']<=1) & (multi_check_df['xx_to_yy_street_counts']<=1) & (multi_check_df['unit_id']<=1) ##This does most of the heavy lifting
        ], 
        [
        'single',
        'single',
        'single',
        'multi',
        'multi',
        'multi',
        'single',
        
        ], 
        default='unknown'
    )
   #With the multiaddress dataframe created the required vectors can now be produced

    multi_unit_id = set(multi_check_df['datapoint_id'][(multi_check_df['class']=='multi') &( multi_check_df['unit_id']>0)].tolist())
    multi_property = set(multi_check_df['datapoint_id'][(multi_check_df['class']=='multi') &( multi_check_df['unit_id']==0)].tolist())
    all_multi_ids = list(multi_unit_id) +list(multi_property)

    return multi_unit_id, multi_property, all_multi_ids
    

def spread_address_labels(df, all_multi_ids):
    """
    This function spreads the address dataframe  so that each
    label class is it's own column
    """
    #pivot the columns so that each label class is it's own column and the value in the column is the text

    temp_df = df[df.datapoint_id.isin(all_multi_ids)].copy()

    temp_df['index'] = temp_df.index
    df = temp_df[['index', 'label', 'label_text']].pivot(index='index',columns='label',values='label_text')
    #add the datapoint_id back in for each of joining
    df = pd.concat([temp_df['datapoint_id'], df], axis=1).merge(temp_df[['datapoint_id' ,'text']].drop_duplicates(), 
          how = "left",
          left_on = "datapoint_id", right_on = "datapoint_id")
    
    return df


def add_backfill_blockers(df):
    """
    This places blockers in the spread address dataframe to prevent labels
    being propergates back or forword when not logical.
    As an example if a building is going     to back fill up previous addresses it should not 
    back fill past another street as this is highly unlikely to be the same building

    """
    
    #This indexing should be done using loc
    df.loc[df['street_name'].notnull(),'building_name'] = 'block'
    df.loc[df['street_name'].notnull(),'street_number'] = 'block' #for multi-flats inside a common building

    #returns true if current number filter is null and the next row has street_number or unit id is not null
    #prevents number filters propergsating back across roads and unit ids
    number_filter_block = df['number_filter'].isnull() & (df['street_number'].shift().notnull() |df['unit_id'].shift().notnull())
    df.loc[number_filter_block,'number_filter'] = 'block'
    
    return df

def backfill_address_labels(df):
    
    """
    Backfilling adds address information in. However, street address should only be back filled for multi addresses.
    I need to work out how to do flat, which may be before or after the unit ID
    Also I don't think this is a very good way of doing it at all.
    """
    
    df['number_filter'] = df[['datapoint_id','number_filter']].groupby('datapoint_id').fillna(method ='bfill')
    df['building_name'] = df[['datapoint_id','building_name']].groupby('datapoint_id').fillna(method ='bfill')
    df['street_number'] = df[['datapoint_id','street_number']].groupby('datapoint_id').fillna(method ='bfill')
    df['postcode'] = df[['datapoint_id','postcode']].groupby('datapoint_id').fillna(method ='bfill')
    df['street_name'] = df[['datapoint_id','street_name']].groupby('datapoint_id').fillna(method ='bfill')
    df['number_filter'] = df[['datapoint_id','number_filter']].groupby('datapoint_id').fillna(method ='bfill')
    df['city'] = df[['datapoint_id','city']].groupby('datapoint_id').fillna(method ='bfill')
    #should this will backwards or forwards? as mostly it is flat xx not xx flat?
    df['unit_type'] = df[['datapoint_id','unit_type']].groupby('datapoint_id').fillna(method ='ffill') 
    
    return df

def final_parsed_addresses(df,all_entities ,multi_property, multi_unit_id, all_multi_ids, expand_addresses = True):

    """
    This function creates the final parsed address dataframe.
    It can either expand the multi-addresses in the format xx to yy or not.
    This is because other address parsers are not designed to perform such and expansion
    and so would make such a comparison unfair.
    """
    xx_to_yy_regex = r'^\d+\s?(?:-|to|\/)\s?\d+$'

    expanded_street = df[df.datapoint_id.isin(multi_property) & df.street_number.str.contains(xx_to_yy_regex)].reset_index()
    expanded_unit_id = df[df.datapoint_id.isin(multi_unit_id) & df.unit_id.str.contains(xx_to_yy_regex)].reset_index()

    #Generally expansion is required as it changes the format to 1 address per row
    #N.B. not all expanded addresses are valid. Office blocks are 1 property but can cover multiple street addresses
    #A matching and cleaning process is required to identify what should be expanded and what not
    if expand_addresses==True:
        expanded_street = expand_dataframe_numbers(expanded_street, column_name = "street_number" )
        expanded_unit_id = expand_dataframe_numbers(expanded_unit_id, column_name = "unit_id" )
        
    #unit id and street number that does does not have the xx to yy format and so has already been expanded by spreading and backfilling
    expanded_street_simple = df[df.datapoint_id.isin(multi_property) & 
                            (df.street_number.str.contains(xx_to_yy_regex)==False) & (df.street_number!='block')].reset_index()
    expanded_unit_id_simple = df[df.datapoint_id.isin(multi_unit_id) & 
                             (df.unit_id.str.contains(xx_to_yy_regex)==False) & (df.unit_id!='block')].reset_index()

    #remove the multi-addresses
    single_address_only =all_entities[~all_entities['datapoint_id'].isin(all_multi_ids)]
    #remove all but the first instance of a label in the remaining instances
    #this is because for single addresses there should be only a single label for each class
    single_address_only =single_address_only[single_address_only['label_id_count']==0]
    df2 = single_address_only.pivot(index='datapoint_id',columns='label',values='label_text')
    #add the datapoint_id back in for each of joining
    df2 = df2.merge(single_address_only[['datapoint_id', 'text']].drop_duplicates(), 
          how = "left",
          left_on = "datapoint_id", right_on = "datapoint_id")

    full_expanded_data = pd.concat([expanded_street, 
           expanded_unit_id, 
           expanded_street_simple, 
           expanded_unit_id_simple, 
           df2 ])

   
    return full_expanded_data

def parsing_and_expansion_process(all_entities, expand_addresses = False ):
    
    """
    This function is more syntactic sugar to provide a simple interface for the expansion and parsing pipeline
    It takes a pandas dataframe that has been produced by one of the loading functions that is "load_cleaned_labels" or "load_data_with_overlaps_jonno"
    It also takes a logical depending on whether expanding the addresses is desired or not
    """
    #This regex is used in several places and is kept here as it was originally used in the function below.
    #xx_to_yy_regex = r'^\d+\s?(?:-|to)\s?\d+$'
    multi_unit_id, multi_property, all_multi_ids = identify_multi_addresses(all_entities)
    df = spread_address_labels(all_entities, all_multi_ids)

    #Blockers prevent the filling of wrong information. As an example if a building is going to back fill up 
    #previous addresses it should not back fill past another street as this is highly unlikely to be the same building
    df = add_backfill_blockers(df)
    df = backfill_address_labels(df)

    out = final_parsed_addresses(df,all_entities ,multi_property, multi_unit_id, all_multi_ids, expand_addresses = expand_addresses)
    
    return out

def load_and_prep_OCOD_data(file_path):
    
    """
    This function is used to provide a simplified interface for the loading and minor processing required for the OCOD dataset
    before the key meta data it contains is added to the expanded dataset
    """
    ocod_data =  pd.read_csv(file_path,
                   encoding_errors= 'ignore').rename(columns = lambda x: x.lower().replace(" ", "_"))
    #empty addresses cannot be used. however there are only three so not a problem
    ocod_data = ocod_data.dropna(subset = 'property_address')
    ocod_data.reset_index(inplace = True, drop = True)
    ocod_data = ocod_data[['title_number', 'tenure', 'district', 'county',
       'region', 'price_paid', 'property_address']]

    ocod_data['property_address'] = ocod_data['property_address'].str.lower()

    #ensure there is a space after commas
    #This is because some numbers are are written as 1,2,3,4,5 which causes issues during tokenisation
    ocod_data.property_address = ocod_data.property_address.str.replace(',', r', ', regex = True)
    #remove multiple spaces
    ocod_data.property_address = ocod_data.property_address.str.replace('\s{2,}', r' ', regex = True)

    #typo in the data leads to a large number of fake flats
    ocod_data.loc[:, 'property_address'] = ocod_data['property_address'].str.replace("stanley court ", "stanley court, ")
    #This typo leads to some rather silly addresses
    ocod_data.loc[:, 'property_address'] = ocod_data['property_address'].str.replace("100-1124", "100-112")
    ocod_data.loc[:, 'property_address'] = ocod_data['property_address'].str.replace("40a, 40, 40¨, 42, 44", "40a, 40, 40, 42, 44")
    
    return ocod_data

def post_process_expanded_data(expanded_data, ocod_data):
    """
    This function adds in additional meta-data from the ocod dataset and prepares the final expanded dataset to be 
    exported for geo-location and classification
    it takes two arguements
    expanded_data is a pandas dataframe produced by the 'final_parsed_addresses' function
    """
    full_expanded_data = expanded_data.merge(ocod_data, how = "left", left_on = "datapoint_id", right_index = True)
    
    full_expanded_data['within_title_id'] = full_expanded_data.groupby('title_number').cumcount()+1
    full_expanded_data['unique_id'] = [str(x) + '-' + str(y) for x, y in zip(full_expanded_data['title_number'], full_expanded_data['within_title_id'])]

    tmp_df =((full_expanded_data[['title_number', 'within_title_id']].groupby('title_number').max('within_title_id'))>1)
    tmp_df.columns = tmp_df.columns.str.replace('within_title_id', 'within_larger_title') #could also be called nested_address
    full_expanded_data = full_expanded_data.merge(tmp_df, how = "left", left_on = "title_number", right_index = True)


    full_expanded_data['postcode'] =full_expanded_data['postcode'].str.upper()
    del tmp_df

    #re-order the columns and drop columns that are not needed

    full_expanded_data =full_expanded_data[['title_number', 'within_title_id', 'unique_id', 'within_larger_title',  'tenure','unit_id', 'unit_type','building_name','street_number', 'street_name', 'postcode','city',  'district', 'county', 'region',
       'price_paid' ,'property_address']].replace('block', np.NaN)
    
    return full_expanded_data


def spacy_pred_fn(spacy_model_path, ocod_data):

    """
    This function predicts over the OCOD dataframe using a spacy model from the path arguement.
    THe function is designed to be used with the CPU model but can be used with GPU. 
    However, currently due to issues with the CUDA drivers I am only using CPU
    """
    print('Loading the spaCy model')
    nlp1 = spacy.load(spacy_model_path) 
    #Tokenization needs to be customised to take account of the foibles of the data. I have added a couple of additional splitting criteria
    infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\-\,*^\(\)](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
        r"(?<=[{a}0-9])[:<>=/\(\)](?=[{a}])".format(a=ALPHA),
        r"(?<=[{a}])[:<>=/\(\)](?=[{a}0-9])".format(a=ALPHA), #I added this one in to try and break things like "(odd)33-45"
    ]   
    )

    infix_re = compile_infix_regex(infixes)
    nlp1.tokenizer.infix_finditer = infix_re.finditer

    print('Adding the datapoint id and title number meta data to the property address')
    ocod_context = [(ocod_data.loc[x,'property_address'], {'datapoint_id':x, 'title_number':str(ocod_data.title_number[x])}) for x in range(0,ocod_data.shape[0])]
    i = 0
    all_entities_json = []    
    print('predicting over the OCOD dataset using the pre-trained spaCy model')    
    for doc, context in list(nlp1.pipe(ocod_context, as_tuples = True)):

        #This doesn't print as it is a stream not a conventional loop
        #if i%print_every==0: print("doc ", i, " of "+ str(ocod_data.shape[0]))
        #i = i+1

        temp = doc.to_json()
        temp.pop('tokens')
        """
        i = 0
        #add the actual entity text to the json
        for entity in doc.ents:
            temp['ents'][i]['label_text'] = entity
            i = i+1
        """

        temp.update({'datapoint_id':context['datapoint_id']})
        all_entities_json = all_entities_json + [temp]

    all_entities = pd.json_normalize(all_entities_json, record_path = "ents", meta= ['text', 'datapoint_id'])
    print('extracting entity label text')
    all_entities['label_text'] = [all_entities.text[x][all_entities.start[x]:all_entities.end[x]] for x in range(0,all_entities.shape[0])]
    
    all_entities['label_id_count'] = all_entities.groupby(['datapoint_id', 'label']).cumcount()
    print('Names Entity Recognition labelling complete')
    return all_entities