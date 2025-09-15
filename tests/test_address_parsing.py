import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from enhance_ocod.address_parsing import process_addresses
from typing import List
import numpy as np


"""
This set of tests ensures that address entities are being parsed as expected.
It checks basic parse order as well as more complex edge cases. 
New edge cases can be added and fixes created (if necessary) as they are found.
"""

def create_test_entity(text: str, target_substring: str, entity_type: str) -> dict:
    """
    Find a substring in text and return its position with the specified type.
    
    Args:
        text: The input text to search in
        target_substring: The exact substring to find
        entity_type: The entity type to label it as
    
    Returns:
        Dictionary with type, text, start, and end positions
    """
    start_pos = text.find(target_substring)
    
    if start_pos == -1:
        raise ValueError(f"Substring '{target_substring}' not found in text '{text}'")
    
    end_pos = start_pos + len(target_substring)
    
    return {
        'type': entity_type,
        'text': target_substring,
        'start': start_pos,
        'end': end_pos
    }

def build_test_case(original_address: str, annotations: List[tuple]) -> dict:
    """
    Build a complete test case from address text and annotations.
    
    Args:
        original_address: The full address string
        annotations: List of (substring, entity_type) tuples
    
    Returns:
        Dictionary containing original_address, entities
    """
    entities = []
    
    for substring, entity_type in annotations:
        entity = create_test_entity(original_address, substring, entity_type)
        entities.append(entity)
    
    return {
        'original_address': original_address,
        'entities': entities
    }

# Example usage - let me create test cases for your addresses:

def create_all_test_cases():
    """Create test cases for all your test addresses"""
    
    test_cases = []
    # simplest address
    test_cases.append(build_test_case(
        "33 high street, London",
        [
            ("33", "street_number"),
            ("high street", "street_name"), 
            ("London", "city")
        ]
    ))
    # one single multi street level address
    test_cases.append(build_test_case(
        "33-35, high street, London",
        [
            ("33-35", "street_number"),
            ("high street", "street_name"),
            ("London", "city")
        ]
    ))
    # several addresses
    test_cases.append(build_test_case(
        "33, 34,35 high street, London",
        [
            ("33", "street_number"),
            ("34", "street_number"),
            ("35", "street_number"),
            ("high street", "street_name"),
            ("London", "city")
        ]
    ))
    #multiple unit types
    test_cases.append(build_test_case(
        "units 33-35, high street, London", 
        [
            ("units", "unit_type"),
            ("33-35", "unit_id"),
            ("high street", "street_name"),
            ("London", "city")
        ]
    ))
    # simply different unit type
    test_cases.append(build_test_case(
        "flat 33-35, high street, London",
        [
            ("flat", "unit_type"),
            ("33-35", "unit_id"), 
            ("high street", "street_name"),
            ("London", "city")
        ]
    ))

    # Multiple numbers but only 1 filter
    test_cases.append(build_test_case(
        "1, 7-10, 11-17 (odds), Broadhall House, 34-37 river road, London (sw15 8pt)",
        [
            ("1", "unit_id"),
            ("7-10", "unit_id"),
            ("11-17", "unit_id"), 
            ("odd", "number_filter"), 
            ("Broadhall House", "building_name"),
            ("34-37", "street_number"),
            ("river road", "street_name"),
            ("London", "city"),
            ("sw15 8pt", "postcode")
        ]
    ))
    # Multiple numbers and multiple filters
    test_cases.append(build_test_case(
        "1, 7-10, 11-17 (odds), 22-28 (evens), Broadhall House, 34-37 river road, London (sw15 8pt)",
        [
            ("1", "unit_id"),
            ("7-10", "unit_id"),
            ("11-17", "unit_id"), 
            ("odd", "number_filter"), 
            ("22-28", "unit_id"),
            ("even", "number_filter"),
            ("Broadhall House", "building_name"),
            ("34-37", "street_number"),
            ("river road", "street_name"),
            ("London", "city"),
            ("sw15 8pt", "postcode")
        ]
    ))
    
    # Test case Multiple buildings
    test_cases.append(build_test_case(
        "Flats 36 - 40 (even), 42-44, climb house, Flats 1-5 down buildings, ascent street, London, se45 6pq",
        [
            ("Flats", "unit_type"),
            ("36 - 40", "unit_id"),
            ("even", "number_filter"),
            ("42-44", "unit_id"),
            ("climb house", "building_name"),
            ("Flats", "unit_type"),  
            ("1-5", "unit_id"),
            ("down buildings", "building_name"),
            ("ascent street", "street_name"),
            ("London", "city"),
            ("se45 6pq", "postcode")
        ]
    ))
    
    # Two buildings no city, postcode,or street. This should result in two separate trees
    test_cases.append(build_test_case(
        "Flats 36 - 40 (even), 42-44, climb house, Flats 1-5 down buildings",
        [
            ("Flats", "unit_type"),
            ("36 - 40", "unit_id"),
            ("even", "number_filter"),
            ("42-44", "unit_id"),
            ("climb house", "building_name"),
            ("Flats", "unit_type"),  
            ("1-5", "unit_id"),
            ("down buildings", "building_name"),
        ]
    ))

    return [
    {**test_case, 'row_index': i+1, 'datapoint_id': i+1} 
    for i, test_case in enumerate(test_cases)
    ]

# --- Test Configuration ---

# Define the columns that represent the parsed address parts
ENTITY_COLUMNS = [
    'unit_type','unit_id','number_filter','building_name','street_number','street_name','postcode','city', 'datapoint_id'
]

def load_ground_truth():
    """Loads and prepares the ground truth data from the CSV file."""
    try:
        df = pd.read_csv('parse_tree_unit_tests.csv')

        # Replace NaN with None for cleaner comparison (pandas default is float NaN)
        return df.where(pd.notna(df), None)
    except FileNotFoundError:
        pytest.fail("ground_truth.csv not found. Please create it in the test directory.")

# Load the ground truth data once
ground_truth_df = load_ground_truth()

# Create a list of unique datapoint_ids to parameterize the tests
# THIS IS THE CORRECTED LINE
datapoint_ids = ground_truth_df['datapoint_id'].unique().tolist() 


# --- Pytest Test Function ---

@pytest.mark.parametrize("datapoint_id", datapoint_ids)
def test_address_parsing(datapoint_id):
    """
    Tests the address parsing for a single datapoint ID from the CSV,
    handling cases where one input address results in multiple parsed addresses.

    Args:
        datapoint_id: The ID of the original address to test, provided by pytest.
    """
    # 1. ARRANGE: Get the input address and the expected output for the current datapoint_id
    test_case_df = ground_truth_df[ground_truth_df['datapoint_id'] == datapoint_id]

    # Prepare the expected DataFrame for comparison
    expected_df = test_case_df[ENTITY_COLUMNS].reset_index(drop=True)
    
    input_df = create_all_test_cases()

    # 2. ACT: Run the address processing function on the input address
    actual_df = process_addresses(input_df)
    
    # 3. CONVERT NA VALUES IN THE ACTUAL DATAFRAME TO NONE
    actual_df = actual_df.replace({np.nan: None})

    # Prepare the actual DataFrame for comparison
    actual_df_filtered = actual_df[actual_df['datapoint_id'] == datapoint_id]
    actual_df_filtered = actual_df_filtered[ENTITY_COLUMNS].reset_index(drop=True)

    # 4. ASSERT
    assert_frame_equal(actual_df_filtered, expected_df)