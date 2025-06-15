"""
Text Preprocessing Module for NLP Training Data

This module provides utilities for preprocessing text data and maintaining entity span
alignment for Named Entity Recognition (NER) and other NLP tasks. It addresses common
tokenization issues by strategically adding spaces around punctuation and operators,
while ensuring that entity annotations remain correctly positioned after preprocessing.

Key Features:
    - Text preprocessing with strategic space insertion around punctuation/operators
    - Entity span position recalculation after text transformation
    - Validation of span-text alignment
    - Support for training data loading and preprocessing

Main Components:
    - preprocess_text_for_tokenization(): Core text preprocessing function
    - find_span_positions(): Utility for finding span positions in transformed text
    - preprocess_training_data(): Batch processing of training examples
    - validate_data(): Validation of span-text alignment
    - load_and_preprocess_data(): Complete pipeline for loading and processing data

Typical Usage:
    >>> # Load and preprocess training data
    >>> processed_data = load_and_preprocess_data('train.json')
    
    >>> # Or preprocess individual text
    >>> clean_text = preprocess_text_for_tokenization("apartment(A)123")
    >>> print(clean_text)  # "apartment ( A ) 123"

Data Format:
    The module expects training data in the following format:
    [
        {
            "text": "input text string",
            "spans": [
                {
                    "text": "entity text",
                    "start": int,  # character start position
                    "end": int,    # character end position (exclusive)
                    "label": "entity_label"
                }
            ]
        }
    ]


"""

import re
import json
from typing import List, Dict, Any

def preprocess_text_for_tokenization(text: str) -> str:
    """
    Preprocesses text by inserting spaces at specific patterns to ensure proper tokenization.
    
    This function addresses common tokenization issues in text data by adding
    spaces around punctuation and operators that would otherwise be incorrectly tokenized.
    This is particularly useful for address data where punctuation often appears without
    proper spacing.
    
    Args:
        text (str): The input text to be preprocessed.
    
    Returns:
        str: The preprocessed text with added spaces around specific patterns and 
             normalized whitespace.
    
    Examples:
        >>> preprocess_text_for_tokenization("123+456")
        '123 + 456'
        >>> preprocess_text_for_tokenization("semi-detached")
        'semi - detached'
        >>> preprocess_text_for_tokenization("london.Manchester")
        'london. Manchester'
        >>> preprocess_text_for_tokenization("(odd)33-45")
        '( odd ) 33-45'
    
    Note:
        The function applies transformations in a specific order and cleans up
        multiple spaces at the end. The transformations are designed to be
        idempotent - running the function multiple times on the same text
        should produce the same result.
    """
    # Pattern 1: Add spaces around operators/punctuation between numbers
    # e.g., "123+456" -> "123 + 456"
    text = re.sub(r'(?<=[0-9])([+\-\,*^\(\)])(?=[0-9-])', r' \1 ', text)
    
    # Pattern 2: Add space after period between lowercase/quotes and uppercase/quotes
    # e.g., "end.Start" -> "end. Start"
    text = re.sub(r'(?<=[a-z"\'])\.(?=[a-zA-Z"\'])', r'. ', text)
    
    # Pattern 3: Add space after commas between alphabetic characters
    # e.g., "apple,banana" -> "apple, banana"
    text = re.sub(r'(?<=[a-zA-Z]),(?=[a-zA-Z])', r', ', text)
    
    # Pattern 4: Add spaces around hyphens between alphabetic characters
    # e.g., "semi-detached" -> "semi - detached"
    text = re.sub(r'(?<=[a-zA-Z])[-–—](?=[a-zA-Z])', r' - ', text)
    
    # Pattern 5 & 6: Add spaces around operators between alphanumeric and alphabetic
    # e.g., "(odd)33-45" -> " ( odd ) 33-45"
    # e.g., "value:123" -> "value : 123"
    text = re.sub(r'(?<=[a-zA-Z0-9])([:<>=/\(\)])(?=[a-zA-Z])', r' \1 ', text)
    text = re.sub(r'(?<=[a-zA-Z])([:<>=/\(\)])(?=[a-zA-Z0-9])', r' \1 ', text)
    
    # Clean up any multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()




def find_span_positions(text: str, span_text: str) -> tuple:
    """
    Find the start and end positions of a span within a text.
    
    Searches for an exact match of span_text within text. If no exact match is found,
    falls back to case-insensitive search. If still not found, returns dummy positions
    and prints a warning.
    
    Args:
        text (str): The text to search within.
        span_text (str): The span text to find.
    
    Returns:
        tuple: A tuple of (start, end) positions where:
            - start (int): The starting character index of the span in text
            - end (int): The ending character index (exclusive) of the span in text
            
    Examples:
        >>> find_span_positions("Hello world", "world")
        (6, 11)
        >>> find_span_positions("Hello World", "world")  # Case-insensitive fallback
        (6, 11)
        >>> find_span_positions("Hello", "goodbye")  # Not found
        Warning: Could not find 'goodbye' in text
        (0, 7)
        
    Warning:
        When span_text is not found, the function returns (0, len(span_text)) as
        dummy positions. Calling code should validate the results.
    """
    start = text.find(span_text)
    if start == -1:
        # If exact match not found, it might be due to case or whitespace differences
        # Try case-insensitive search as fallback
        start = text.lower().find(span_text.lower())
        if start == -1:
            print(f"Warning: Could not find '{span_text}' in text")
            return 0, len(span_text)  # Return dummy positions
    
    end = start + len(span_text)
    return start, end


def preprocess_training_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Preprocess json training data by transforming text and updating entity span positions.
    
    Applies text preprocessing to both the main text and entity spans, then recalculates
    span positions in the transformed text. This ensures that entity annotations remain
    correctly aligned after text preprocessing.
    
    Args:
        data (List[Dict[str, Any]]): A list of training examples, where each example
            is a dictionary containing:
            - 'text' (str): The input text
            - 'spans' (List[Dict]): List of entity spans, each containing:
                - 'text' (str): The entity text
                - 'start' (int): Original start position
                - 'end' (int): Original end position  
                - 'label' (str): The entity label
    
    Returns:
        List[Dict[str, Any]]: A list of preprocessed training examples with the same
            structure as input, but with:
            - Preprocessed text
            - Updated span positions reflecting the preprocessed text
            - Preprocessed span text
    
    Example:
        >>> data = [{
        ...     "text": "apartment(A)123",
        ...     "spans": [{
        ...         "text": "apartment",
        ...         "start": 0,
        ...         "end": 9,
        ...         "label": "unit_type"
        ...     }]
        ... }]
        >>> processed = preprocess_training_data(data)
        >>> processed[0]['text']
        'apartment ( A ) 123'
        >>> processed[0]['spans'][0]['start'], processed[0]['spans'][0]['end']
        (0, 9)
    
    Note:
        - Both the main text and span texts are preprocessed using the same function
          to ensure consistency
        - If a span cannot be found in the preprocessed text, dummy positions are used
          and a warning is printed
        - The function preserves all original span attributes (like 'label')
    """
    processed_data = []
    
    for example in data:
        # Transform the main text
        processed_text = preprocess_text_for_tokenization(example['text'])
        
        # Process each span
        processed_spans = []
        for span in example['spans']:
            # Transform the span text the same way
            processed_span_text = preprocess_text_for_tokenization(span['text'])
            
            # Find where this span appears in the processed text
            new_start, new_end = find_span_positions(processed_text, processed_span_text)
            
            processed_spans.append({
                'text': processed_span_text,
                'start': new_start,
                'end': new_end,
                'label': span['label']
            })
        
        processed_data.append({
            'text': processed_text,
            'spans': processed_spans
        })
    
    return processed_data


def validate_data(data: List[Dict[str, Any]]) -> bool:
    """Quick validation that spans match their positions."""
    is_valid = True
    for i, example in enumerate(data):
        for span in example['spans']:
            extracted = example['text'][span['start']:span['end']]
            if extracted != span['text']:
                print(f"Mismatch in example {i}: expected '{span['text']}', got '{extracted}'")
                is_valid = False
    return is_valid


# Usage:
def load_and_preprocess_data(train_file: str):
    """Load and preprocess the training data."""
    # Load data
    with open(train_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples")
    
    # Preprocess
    processed_data = preprocess_training_data(data)
    
    # Validate
    if validate_data(processed_data):
        print("✓ All spans validated successfully")
    else:
        print("✗ Some spans failed validation")
    
    return processed_data