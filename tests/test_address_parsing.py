import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from enhance_ocod.address_parsing import (
    process_addresses,
    expand_dataframe_numbers,
    expand_dataframe_numbers_core,
    expand_multi_id,
    needs_expansion
)
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


# --- Tests for Expansion Size Tracking and Large Expansion Flagging ---

class TestExpansionTracking:
    """Tests for the expansion size tracking and large expansion flagging functionality."""

    def test_expand_dataframe_numbers_core_adds_metadata_columns(self):
        """Test that expand_dataframe_numbers_core adds expansion_size and large_expansion columns."""
        # Create a simple test dataframe with a range to expand
        df = pd.DataFrame({
            'unit_id': ['1-5'],
            'number_filter': ['none'],
            'class': ['residential']
        })

        result = expand_dataframe_numbers_core(
            df,
            column_name='unit_id',
            large_expansion_threshold=10
        )

        # Check that metadata columns exist
        assert 'expansion_size' in result.columns
        assert 'large_expansion' in result.columns

        # Check correct values
        assert len(result) == 5  # 1-5 expands to 5 rows
        assert all(result['expansion_size'] == 5)
        assert all(result['large_expansion'] == False)  # 5 < 10

    def test_large_expansion_flag_triggered(self):
        """Test that large_expansion flag is set to True when threshold is exceeded."""
        # Create a dataframe with a large range
        df = pd.DataFrame({
            'unit_id': ['1-150'],
            'number_filter': ['none'],
            'class': ['residential']
        })

        result = expand_dataframe_numbers_core(
            df,
            column_name='unit_id',
            large_expansion_threshold=100
        )

        # Check that flag is True since 150 > 100
        assert len(result) == 150
        assert all(result['expansion_size'] == 150)
        assert all(result['large_expansion'] == True)

    def test_custom_threshold_parameter(self):
        """Test that custom threshold values work correctly."""
        df = pd.DataFrame({
            'unit_id': ['1-50'],
            'number_filter': ['none'],
            'class': ['residential']
        })

        # Test with threshold of 30
        result_30 = expand_dataframe_numbers_core(
            df,
            column_name='unit_id',
            large_expansion_threshold=30
        )
        assert all(result_30['large_expansion'] == True)  # 50 > 30

        # Test with threshold of 60
        result_60 = expand_dataframe_numbers_core(
            df,
            column_name='unit_id',
            large_expansion_threshold=60
        )
        assert all(result_60['large_expansion'] == False)  # 50 < 60

    def test_expansion_size_for_small_range(self):
        """Test expansion_size is correct for small ranges."""
        df = pd.DataFrame({
            'unit_id': ['10-12'],
            'number_filter': ['none'],
            'class': ['residential']
        })

        result = expand_dataframe_numbers_core(
            df,
            column_name='unit_id',
            large_expansion_threshold=100
        )

        assert len(result) == 3  # 10, 11, 12
        assert all(result['expansion_size'] == 3)
        assert all(result['large_expansion'] == False)

    def test_erroneous_ner_example(self):
        """Test the specific case mentioned: NER error creating 0-2009 instead of 2000-2009."""
        # Simulate the NER error case
        df = pd.DataFrame({
            'unit_id': ['0-2009'],
            'number_filter': ['none'],
            'class': ['residential']
        })

        result = expand_dataframe_numbers_core(
            df,
            column_name='unit_id',
            large_expansion_threshold=100
        )

        # This should create 2010 rows and be flagged
        assert len(result) == 2010
        assert all(result['expansion_size'] == 2010)
        assert all(result['large_expansion'] == True)

    def test_legitimate_large_apartment_building(self):
        """Test a legitimate large apartment building (e.g., 2000-2099)."""
        df = pd.DataFrame({
            'unit_id': ['2000-2099'],
            'number_filter': ['none'],
            'class': ['residential']
        })

        result = expand_dataframe_numbers_core(
            df,
            column_name='unit_id',
            large_expansion_threshold=150  # Set threshold above 100
        )

        # This creates 100 units, which may or may not be flagged depending on threshold
        assert len(result) == 100
        assert all(result['expansion_size'] == 100)
        assert all(result['large_expansion'] == False)  # 100 < 150

    def test_expand_dataframe_numbers_adds_metadata_to_all_rows(self):
        """Test that expand_dataframe_numbers adds metadata columns to both expanded and non-expanded rows."""
        # Mix of rows that need expansion and rows that don't
        df = pd.DataFrame({
            'unit_id': ['1-5', '10', None],
            'street_number': [None, None, '25'],
            'number_filter': ['none', 'none', 'none'],
            'class': ['residential', 'residential', 'residential']
        })

        result = expand_dataframe_numbers(
            df,
            class_var='class',
            large_expansion_threshold=10
        )

        # Check that all rows have metadata columns
        assert 'expansion_size' in result.columns
        assert 'large_expansion' in result.columns

        # Non-expanded rows should have expansion_size=1 and large_expansion=False
        non_expanded = result[result['expansion_size'] == 1]
        assert all(non_expanded['large_expansion'] == False)

    def test_expansion_with_even_filter(self):
        """Test expansion size tracking works correctly with number filters."""
        df = pd.DataFrame({
            'unit_id': ['1-20'],
            'number_filter': ['even'],
            'class': ['residential']
        })

        result = expand_dataframe_numbers_core(
            df,
            column_name='unit_id',
            large_expansion_threshold=15
        )

        # Even numbers from 1-20: 2,4,6,8,10,12,14,16,18,20 = 10 numbers
        assert len(result) == 10
        assert all(result['expansion_size'] == 10)
        assert all(result['large_expansion'] == False)  # 10 < 15

    def test_expansion_with_odd_filter(self):
        """Test expansion size tracking works correctly with odd number filter."""
        df = pd.DataFrame({
            'unit_id': ['1-20'],
            'number_filter': ['odd'],
            'class': ['residential']
        })

        result = expand_dataframe_numbers_core(
            df,
            column_name='unit_id',
            large_expansion_threshold=5
        )

        # Odd numbers from 1-20: 1,3,5,7,9,11,13,15,17,19 = 10 numbers
        assert len(result) == 10
        assert all(result['expansion_size'] == 10)
        assert all(result['large_expansion'] == True)  # 10 > 5

    def test_multiple_expansions_in_same_dataframe(self):
        """Test tracking works when multiple ranges are expanded in the same dataframe."""
        df = pd.DataFrame({
            'unit_id': ['1-10', '50-200', '5-8'],
            'number_filter': ['none', 'none', 'none'],
            'class': ['residential', 'residential', 'residential']
        })

        result = expand_dataframe_numbers_core(
            df,
            column_name='unit_id',
            large_expansion_threshold=50
        )

        # Check that each expansion has the correct size
        # First expansion: 1-10 = 10 rows
        first_expansion = result[result['unit_id'] == '1']
        assert first_expansion['expansion_size'].iloc[0] == 10
        assert first_expansion['large_expansion'].iloc[0] == False

        # Second expansion: 50-200 = 151 rows
        second_expansion = result[result['unit_id'] == '50']
        assert second_expansion['expansion_size'].iloc[0] == 151
        assert second_expansion['large_expansion'].iloc[0] == True  # 151 > 50

        # Third expansion: 5-8 = 4 rows
        third_expansion = result[result['unit_id'] == '5']
        assert third_expansion['expansion_size'].iloc[0] == 4
        assert third_expansion['large_expansion'].iloc[0] == False

    def test_street_number_expansion_tracking(self):
        """Test that expansion tracking works for street_number column as well."""
        df = pd.DataFrame({
            'unit_id': [None],
            'street_number': ['1-100'],
            'number_filter': ['none'],
            'class': ['residential']
        })

        # Mark as needing expansion first
        df = needs_expansion(df, class_var='class')

        result = expand_dataframe_numbers_core(
            df[df['needs_expansion']].reset_index(drop=True),
            column_name='street_number',
            large_expansion_threshold=50
        )

        assert len(result) == 100
        assert all(result['expansion_size'] == 100)
        assert all(result['large_expansion'] == True)  # 100 > 50

    def test_default_threshold_value(self):
        """Test that the default threshold of 100 is used when not specified."""
        df = pd.DataFrame({
            'unit_id': ['1-99', '1-101'],
            'number_filter': ['none', 'none'],
            'class': ['residential', 'residential']
        })

        # Use default threshold (should be 100)
        result = expand_dataframe_numbers_core(
            df,
            column_name='unit_id'
            # Not specifying large_expansion_threshold
        )

        # 99 should not be flagged (99 < 100)
        result_99 = result[result['unit_id'] == '1'].head(1)
        assert result_99['expansion_size'].iloc[0] == 99
        assert result_99['large_expansion'].iloc[0] == False

        # 101 should be flagged (101 > 100)
        result_101 = result[result['unit_id'].astype(int) > 99].head(1)
        assert result_101['expansion_size'].iloc[0] == 101
        assert result_101['large_expansion'].iloc[0] == True

    def test_empty_dataframe_handling(self):
        """Test that empty dataframes are handled correctly."""
        df = pd.DataFrame({
            'unit_id': [],
            'number_filter': [],
            'class': []
        })

        result = expand_dataframe_numbers_core(
            df,
            column_name='unit_id',
            large_expansion_threshold=100
        )

        # Should return empty dataframe unchanged
        assert len(result) == 0