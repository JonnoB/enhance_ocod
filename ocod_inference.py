#!/usr/bin/env python3

import json
import spacy

from address_parsing_helper_functions import load_and_prep_OCOD_data

import os
print(os.getcwd())

#spacy.require_gpu()
spacy.prefer_gpu()

nlp1 = spacy.load("/home/jonno/empty_homes_data/spacy_data/model-best") 

ocod_data = load_and_prep_OCOD_data('/home/jonno/empty_homes_data/' +'OCOD_FULL_2022_02.csv')

spacy_labels = []
for x in  range(0,ocod_data.shape[0]):

    if x % 100 == 0: 
        print('x = {}'.format(x))

    temp = nlp1(ocod_data.loc[x,'property_address']).to_json()
    temp.pop('tokens')
    #temp['datapoint_id'] = x
    temp.update({'datapoint_id':x})
    spacy_labels = spacy_labels + [temp]

with open('/home/jonno/empty_homes_dataspacy_pred_labels.json', 'w') as f:
    json.dump(spacy_labels, f)