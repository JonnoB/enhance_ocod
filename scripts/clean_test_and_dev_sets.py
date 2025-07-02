"""
# Correcting the Dev and Test sets

# This script does not need to be run and has been kept only for completness


During training the ModernBERT models an error was discovered in the DEV and TEST sets. 
This was caused by a download from the original weaklabelling platform not being correctly filtered to remove 
labels predicted by the platform. As a result the DEV TEST sets contained a mixture of handlabeled data and 
poorly predicted data. This had no impact on the model or results in the original paper but would affect 
subsequent work and reproducibility. 

This notebook remedies the situation by remaking the DEV and TEST sets from the original data and removing 
the predicted labels. This is done by ensuring that only data marked as "annotation" is included, whilst 
"prediction" is excluded.

In addition the csv columns will be renamed to be consistent with the rest of the training and inference pipeline.

The results of this change cause a large reduction in the apparent number of entities in the DEV and TEST sets, 
however, these entities are at best duplicates of the annotations and at worse incorrectly parsed entities which 
are duplicated several times. This means that the updated DEV and TEST sets are much smaller than the original 
publibly available sets.

"""

import pandas as pd
def clean_raw_data(path):

    """
    This funcction simply keeps only the rows that are marked as "annotation" i.e. are hand labelled.
    It then selects the needed columns and renames as appropriate 
    """

    gt_data_raw = pd.read_csv(path)


    gt_data =  gt_data_raw.loc[gt_data_raw['type'] == 'annotation',
    ['input:datapoint_id', 'start', 'end', 'label', 'text', 'input:text',]]
    
    gt_data = gt_data.rename(columns = {'input:text':'property_address', 
    'input:datapoint_id':'datapoint_id'})    

    return gt_data


gt_test_data = clean_raw_data(path = "data/ground_truth_test_set_labels.csv")
gt_test_data.to_csv("data/enhanced_ocod_data_and_gt/ground_truth_test_set_labels.csv", index = False)
print(gt_test_data.head())


gt_dev_data = clean_raw_data(path = "data/ground_truth_dev_set_labels.csv")
gt_dev_data.to_csv("data/enhanced_ocod_data_and_gt/ground_truth_dev_set_labels.csv", index = False)
print(gt_dev_data.head())