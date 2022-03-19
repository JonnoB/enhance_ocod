
#the functions used by the empty homes project python notebooks
import re
import time
import pandas as pd
import numpy as np

##
## Removing overlaps section
##

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
