import pytest
import pandas as pd
import numpy as np
from enhance_ocod.locate_and_classify import property_class  


""" 
This set of tests is to ensure that the property classification function logic works as expected. 
The tests are not for the rules themselves

"""

class TestPropertyClassification:
    
    def test_single_rule_land_identification(self):
        """Test that a single rule correctly identifies land properties."""
        # Arrange
        df = pd.DataFrame({
            'property_address': ['land at the edge', 'apartment 1', 'office building']
        })
        
        rules = [
            {
                'rule_name': 'Land identification',
                'condition': lambda df: df["property_address"].str.contains("land", case=False),
                'class': 'land',
                'comments': 'Identifies properties containing "land"'
            }
        ]
        
        # Act
        result_df = property_class(df, rules)
        
        # Assert
        assert result_df.loc[0, 'class'] == 'land'
        assert result_df.loc[1, 'class'] == 'unknown'  # default value
        assert result_df.loc[2, 'class'] == 'unknown'  # default value

    def test_single_rule_business_identification(self):
        """Test that a single rule correctly identifies business properties."""
        # Arrange
        df = pd.DataFrame({
            'property_address': ['shopping centre main hall', 'apartment 1', 'residential flat']
        })
        
        rules = [
            {
                'rule_name': 'Business identification',
                'condition': lambda df: df["property_address"].str.contains("centre", case=False),
                'class': 'business',
                'comments': 'Identifies properties containing "centre"'
            }
        ]
        
        # Act
        result_df = property_class(df, rules)
        
        # Assert
        assert result_df.loc[0, 'class'] == 'business'
        assert result_df.loc[1, 'class'] == 'unknown'
        assert result_df.loc[2, 'class'] == 'unknown'

    def test_rule_hierarchy_land_first(self):
        """Test that when land rule comes first, it takes precedence over business rule."""
        # Arrange
        df = pd.DataFrame({
            'property_address': ['land at the edge of the shopping centre', 'apartment 1']
        })
        
        rules = [
            {
                'rule_name': 'Land identification',
                'condition': lambda df: df["property_address"].str.contains("land", case=False),
                'class': 'land',
                'comments': 'Identifies properties containing "land"'
            },
            {
                'rule_name': 'Business identification',
                'condition': lambda df: df["property_address"].str.contains("centre", case=False),
                'class': 'business',
                'comments': 'Identifies properties containing "centre"'
            }
        ]
        
        # Act
        result_df = property_class(df, rules)
        
        # Assert
        assert result_df.loc[0, 'class'] == 'land'  # Land rule wins due to order
        assert result_df.loc[1, 'class'] == 'unknown'

    def test_rule_hierarchy_business_first(self):
        """Test that when business rule comes first, it takes precedence over land rule."""
        # Arrange
        df = pd.DataFrame({
            'property_address': ['land at the edge of the shopping centre', 'apartment 1']
        })
        
        rules = [
            {
                'rule_name': 'Business identification',
                'condition': lambda df: df["property_address"].str.contains("centre", case=False),
                'class': 'business',
                'comments': 'Identifies properties containing "centre"'
            },
            {
                'rule_name': 'Land identification',
                'condition': lambda df: df["property_address"].str.contains("land", case=False),
                'class': 'land',
                'comments': 'Identifies properties containing "land"'
            }
        ]
        
        # Act
        result_df = property_class(df, rules)
        
        # Assert
        assert result_df.loc[0, 'class'] == 'business'  # Business rule wins due to order
        assert result_df.loc[1, 'class'] == 'unknown'

    def test_multiple_properties_mixed_rules(self):
        """Test classification of multiple properties with mixed rule matches."""
        # Arrange
        df = pd.DataFrame({
            'property_address': [
                'land at the edge of the shopping centre',  # matches both
                'apartment with shopping centre view',       # matches business only
                'land plot for sale',                       # matches land only
                'residential house'                         # matches neither
            ]
        })
        
        rules = [
            {
                'rule_name': 'Land identification',
                'condition': lambda df: df["property_address"].str.contains("land", case=False),
                'class': 'land',
                'comments': 'Identifies properties containing "land"'
            },
            {
                'rule_name': 'Business identification',
                'condition': lambda df: df["property_address"].str.contains("centre", case=False),
                'class': 'business',
                'comments': 'Identifies properties containing "centre"'
            }
        ]
        
        # Act
        result_df = property_class(df, rules)
        
        # Assert
        assert result_df.loc[0, 'class'] == 'land'      # Land rule comes first
        assert result_df.loc[1, 'class'] == 'business'  # Only business rule matches
        assert result_df.loc[2, 'class'] == 'land'      # Only land rule matches
        assert result_df.loc[3, 'class'] == 'unknown'   # No rules match

    def test_empty_dataframe(self):
        """Test that function handles empty dataframes gracefully."""
        # Arrange
        df = pd.DataFrame({'property_address': []})
        
        rules = [
            {
                'rule_name': 'Land identification',
                'condition': lambda df: df["property_address"].str.contains("land", case=False),
                'class': 'land',
                'comments': 'Identifies properties containing "land"'
            }
        ]
        
        # Act
        result_df = property_class(df, rules)
        
        # Assert
        assert len(result_df) == 0
        assert 'class' in result_df.columns

    def test_missing_property_address_column(self):
        """Test that function handles missing columns gracefully."""
        # Arrange
        df = pd.DataFrame({'other_column': ['some value']})
        
        rules = [
            {
                'rule_name': 'Land identification',
                'condition': lambda df: df["property_address"].str.contains("land", case=False),
                'class': 'land',
                'comments': 'Identifies properties containing "land"'
            }
        ]
        
        # Act
        result_df = property_class(df, rules)
        
        # Assert - Now all rules fail, so all entries get "unknown"
        assert all(result_df['class'] == 'unknown')

    def test_rule_with_exception_continues_processing(self, capsys):
        """Test that a rule with an exception doesn't stop processing of other rules."""
        # Arrange
        df = pd.DataFrame({
            'property_address': ['land plot', 'shopping centre']
        })
        
        rules = [
            {
                'rule_name': 'Broken rule',
                'condition': lambda df: df["non_existent_column"].str.contains("test"),
                'class': 'broken',
                'comments': 'This rule should fail'
            },
            {
                'rule_name': 'Working rule',
                'condition': lambda df: df["property_address"].str.contains("centre", case=False),
                'class': 'business',
                'comments': 'This rule should work'
            }
        ]
        
        # Act
        result_df = property_class(df, rules)
        
        # Assert
        captured = capsys.readouterr()
        assert "Warning: Rule 'Broken rule' failed:" in captured.out
        assert result_df.loc[0, 'class'] == 'unknown'  # No working rule matched
        assert result_df.loc[1, 'class'] == 'business'  # Working rule matched

    def test_no_rules_provided(self):
        """Test behavior when no rules are provided."""
        # Arrange
        df = pd.DataFrame({
            'property_address': ['land plot', 'shopping centre']
        })
        
        rules = []
        
        # Act
        result_df = property_class(df, rules)
        
        # Assert
        assert all(result_df['class'] == 'unknown')
        assert len(result_df) == 2

    def test_all_rules_fail(self, capsys):
        """Test behavior when all rules fail due to exceptions."""
        # Arrange
        df = pd.DataFrame({
            'property_address': ['land plot', 'shopping centre']
        })
        
        rules = [
            {
                'rule_name': 'Broken rule 1',
                'condition': lambda df: df["non_existent_column"].str.contains("test"),
                'class': 'broken1',
                'comments': 'This rule should fail'
            },
            {
                'rule_name': 'Broken rule 2',
                'condition': lambda df: df["another_missing_column"].str.contains("test"),
                'class': 'broken2',
                'comments': 'This rule should also fail'
            }
        ]
        
        # Act
        result_df = property_class(df, rules)
        
        # Assert
        captured = capsys.readouterr()
        assert "Warning: Rule 'Broken rule 1' failed:" in captured.out
        assert "Warning: Rule 'Broken rule 2' failed:" in captured.out
        assert all(result_df['class'] == 'unknown')