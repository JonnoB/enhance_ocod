
#the functions used by the empty homes project python notebooks

import json
import operator
import os
import requests
from pathlib import Path
import spacy
from spacy.tokens import DocBin
from spacy.lang.char_classes import LIST_PUNCT, LIST_ELLIPSES, LIST_QUOTES, LIST_CURRENCY
from spacy.lang.char_classes import LIST_ICONS, HYPHENS, CURRENCY, UNITS
from spacy.lang.char_classes import CONCAT_QUOTES, ALPHA_LOWER, ALPHA_UPPER, ALPHA, PUNCT
from spacy.util import compile_infix_regex





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



def create_spacy_ready_json(data, type = "results"):

    """
    
    This function takes the labelled data from programmatic and puts it in a format that can be imported to spacy

    The function has two arguments

    data: A jsonlike list the output of programmatic
    type: A string. Whether the raw results should be used and overlapping spans cleaned or "aggregateResults" where the denoised
    results are used and overlapping spans are already removed.

    """

    datapoint_id_list = [*range(0,len(data['datapoints']))]
    data_and_labels = []

    #type is either results or aggregate results
    #This also changes the save file name accordingly
    #type ='results' #'aggregateResults' #'results'

    count_it = 0
    for i in set(datapoint_id_list):
        count_it += 1
        if count_it % 5000 == 0: 
            print('count = {}'.format(count_it))
            
        ##these labels are in tuple form
        
        results_list = data['datapoints'][i]['programmatic'][type ]
        list_of_labels =[(x['start'],x['end'],x['label'] ) for x in results_list]
        
        list_of_labels.sort(key=lambda y: y[0])
        #list_of_labels_dict.sort(key=operator.itemgetter('start'))
        #if type =="results":
        list_of_labels = remove_overlapping_spans_tuples(list_of_labels)

        #create the NER dataset structure shown on the spacy website
        data_and_labels = data_and_labels + [ {
            'datapoint_id': i,
            'text':data['datapoints'][i]['data']['text'],
                                            'entities':list_of_labels}  ]
    
    return data_and_labels


def create_spacy_ready_json_from_gt(data):

    """
    creates a spacy ready json from the ground truth produced by humanloop cloud labelling

    """

    data_and_labels = []

    count_it = 0
    for i in range(0, len(data)):
        count_it += 1
        if count_it % 5000 == 0: 
            print('count = {}'.format(count_it))
            
        inputs = data[i]['inputs']

        labels = data[i]['data']['labels']

        list_of_labels =[(x['start'],x['end'],x['label'] ) for x in labels]
        
        list_of_labels.sort(key=lambda y: y[0])
        
        
        #create the NER dataset structure shown on the spacy website
        data_and_labels = data_and_labels + [ {
            'datapoint_id': inputs['datapoint_id'],
            'text':inputs['text'], 
                                            'entities':list_of_labels}  ]
        
        data_and_labels.sort(key=lambda y: y.get('datapoint_id'))

    return data_and_labels


def create_spacy_docbin(training_data, alignment_mode_type = "expand", print_iteration =False):

    """
    creates the actual data that will be used by the spacy training process

    print_iteration: Logical used for debugging if the spans cannot be tokenized
    """


    nlp = spacy.blank("en")


    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\,*^\(\)\/](?=[0-9-])", #added in / to break up 34/45 etc. added in , to break up 34,35 although this should now be removed in the cleaning stage
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
    nlp.tokenizer.infix_finditer = infix_re.finditer

    # the DocBin will store the example documents
    db = DocBin()
    for i in range(0, len(training_data)):
        current_set = training_data[i]
        if print_iteration: print(i) #printing is used for debugging
        #print(training_data[i])
        doc = nlp(current_set['text'])
        ents = []
        for start, end, label in current_set['entities']:
            span = doc.char_span(start, end, label=label, alignment_mode = alignment_mode_type )
            ents.append(span)
        doc.ents = ents
        db.add(doc)

    return db


def upload_to_human_loop(jason_data, config, project_owner ):
    
    """
    jason_data: list the data to upload overlapping spans removed

    config: the config file holding your key
    project_owner: string your humanloop email
    """

    """
    Step 1: Specify URL and headers for your API requests and some helper methods 
    Reference: https://api.humanloop.com/docs#section/Authentication
    Notes: 
        - If you don't already have a Humanloop account,
          signup @ https://app.humanloop.com/signup
        - Replace <INSERT YOUR API KEY HERE> with your users X-API-KEY 
          @ https://app.humanloop.com/profile
    """
    base_url = "https://api.humanloop.com"
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY":  config.api_key,#the api key is hidden in a config file
    }
    # use the email associated to your Humanloop account
    project_owner = project_owner


    def get_field_id_by_name(name: str, fields):
        """Helper method for parsing field_id from dataset.fields given the name"""
        return [field for field in fields if field["name"] == name][0]["id"]


    """ 
    Step 2: Create a dataset
    Reference: https://api.humanloop.com/docs#operation/upload_data_datasets_post
    Notes:
        - It can be helpful to include your own unique identifiers for your data-points
          if available so that you can easily correlate any annotations and predictions 
          created by Humanloop back to your system.
        - If using large datasets (> 10k rows), you will have to upload it in multiple 
          batches using the API. Starting with the POST as shown below, then adding 
          subsequent batches using the PUT against the newly created dataset 
          (https://api.humanloop.com/docs#operation/update_data_datasets__id__put.)
    """

    dataset_fields = requests.post(
        url=f"{base_url}/datasets", data=json.dumps(jason_data), headers=headers ######################## CHANGE this depending on whether dev or test
    ).json()["fields"]


    """
    Step 3: Create a project
    Reference: https://api.humanloop.com/docs#operation/create_project_projects_post
    Notes:
        - A Humanloop project is made up of one or more datasets, a team of annotators 
          and a model. As your team begin to annotate the data, a model is trained in real 
          time and used to prioritise what data your annotators should focus on next 
          (see https://humanloop.com/blog/why-you-should-be-using-active-learning).
        - The project inputs specify those dataset fields you wish to show to 
          your annotators and the model. 
        - The project output specifies the type of model you wish to train and the 
          corresponding label taxonomy. 
        - If your dataset has a field with existing annotations, you can use this to 
          warm start your project as shown in the following examples. 
          If you want your team to first review these existing annotations in Humanloop, 
          set "review_existing_annotations" to True, otherwise they will be used 
          automatically to train an initial model.
        - Both classification (single and multi-label) and extraction
          projects are supported.
        - You can update your project with more data by either connecting another dataset 
          or simply adding additional data-points to your existing dataset. 
          Alternatively, you can submit tasks for your model and/or team to complete
          (see our Human-in-the-loop tutorial for more information on this!).
    """


    """
    Step 3b: Span extraction project 
    """
    extraction_project_request = {
        "name": "Ground truth for offshore empties dev set for spacy training",
        "inputs": [
            {
                "name": "text",
                "data_type": "text",
                "description": "unparsed addresses",
                "data_sources": [
                    {"field_id": get_field_id_by_name("text", dataset_fields)}
                ],
            },
            {
                "name": "datapoint_id",
                "data_type": "text",
                "description": "The original row the data is on",
                "display_only": True,
                "data_sources": [
                    {"field_id": get_field_id_by_name("datapoint_id", dataset_fields)}
                ],
            },
        ],
        "outputs": [
            {
                "name": "labels",
                "description": "entities address parts",
                "task_type": "sequence_tagging",
                "data_sources": [
                    {"field_id": get_field_id_by_name("labels", dataset_fields)}
                ],
                # which input you wish your model to extract from
                "input": "text",
            }
        ],
        "users": [project_owner],
        "guidelines": "Insert your markdown annotator guidelines here",
        "review_existing_annotations": True,
    }

    extraction_project_id = requests.post(
        url=f"{base_url}/projects",
        data=json.dumps(extraction_project_request),
        headers=headers,
    ).json()["id"]

    print(f"Navigate to https://app.humanloop.com/projects/{extraction_project_id}")


def clean_programmatic_for_humanloop(data, ocod_data):
    
    """
    prepares the data for upload to the humanloop cloud for hand labelling a ground truth
    """

    datapoint_id_list = [*range(0,len(data['datapoints']))]

    data_and_labels = []
    data_labels_dict = []

    count_it = 0
    for i in set(datapoint_id_list):

        count_it += 1
        if count_it % 5000 == 0: 
            print('count = {}'.format(count_it))

        #single_id_index = np.where(np.array(datapoint_id_list)==i)
        ##these labels are in tuple form
       # list_of_labels = [(data[x]['start'], data[x]['end'], data[x]['label']) for x in single_id_index[0].tolist()]
        list_of_labels_dict = results_list = data['datapoints'][i]['programmatic']['results']
        ##these labels are in dictionary form
    #     list_of_labels_dict = [{'start': x['start'], 
    #                             'end':x['end'], 
    #                             'label': x['label'], 
    #                             'label_text': x['text'] } for x in results_list]

        #this inplace sorting using operator orders the dictionary by the start point. ties are automatically broken
        #it required the operator library
        list_of_labels_dict.sort(key=operator.itemgetter('start'))

        list_of_labels_dict = remove_overlapping_spans2(list_of_labels_dict)

        #create the NER dataset structure shown on the spacy website
       # data_and_labels = data_and_labels + [ ( ocod_data['property_address'][i], list_of_labels ) ]
        #create a list of dictionaries using a similar structure to save as a json
        data_labels_dict = data_labels_dict + [
            {
                'text' : data['datapoints'][i]['data']['text'],
                'labels' : list_of_labels_dict,
                'datapoint_id': i,
                'title_number':ocod_data['title_number'][i]
            }
        ]
    return data_labels_dict

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

def is_overlap_function(a,b):
    #identifies any overlapping tags.
    #takes two dictionaries and outputs a logical value identifying if the tags overlap
     start_overlap = (b['start'] >= a['start']) & (b['start'] <= a['end']) 
     end_overlap = (b['end'] >= a['start']) & (b['end'] <= a['end']) 
     return (start_overlap | end_overlap)


def create_spacy_training_set(programmatic_data_path = "./data/test.json",
                              dev_set_path = './data/ground_truth_dev_set_labels.json',
                              hmm_denoising = False,
                              alignment_mode_type = "expand",
                              save_folder = "./data/spacy_data"):
    
    """
    goes through the process of creating the spacy training set
    
    
    """

    print('loading data')
    f =open(programmatic_data_path)  #aggregate and download button
    data = json.load(f)

    f =open(dev_set_path)
    data_gt = json.load(f)

    if hmm_denoising:
        type_denoiser = "aggregateResults"
    else:
        type_denoiser = "results"

    print(type_denoiser)

    print('removing overlapping spans from training data')
    spacy_data = create_spacy_ready_json(data, type_denoiser)

    print('removing overlapping spans from dev data')
    dev_set = create_spacy_ready_json_from_gt(data_gt)

    del data
    del data_gt

    #need to create the no dev set alternativee

    #get the indices of the dev set
    dev_set_indices = list(map(operator.itemgetter('datapoint_id'), dev_set))
    dev_set_indices = list(map(int, dev_set_indices))

    #create trainset by removing devset indices

    #random.seed(2017)
    #dev_set_indices = random.sample([*range(0, len(spacy_data))], 9400) 
    #dev_set = [spacy_data[x] for x in dev_set.loc[:, 'datapoint_id'].to_list()]
    train_set = [spacy_data[x] for x in [*range(0, len(spacy_data))] if x not in dev_set_indices ]

    print('creating training DocBin')
    training_data = create_spacy_docbin(train_set, alignment_mode_type = alignment_mode_type)
    print('creating dev DocBin')
    dev_data = create_spacy_docbin(dev_set, alignment_mode_type = alignment_mode_type)

    path = Path(save_folder)

    path.mkdir(parents=True, exist_ok=True)

    training_data.to_disk(save_folder + "/train.spacy")
    dev_data.to_disk(save_folder + "/dev.spacy")


    return training_data, dev_data