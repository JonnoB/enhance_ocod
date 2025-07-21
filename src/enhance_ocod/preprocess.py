""" """

import re
import pandas as pd
from typing import Union


def preprocess_text_for_tokenization(
    text: Union[str, pd.Series],
) -> Union[str, pd.Series]:
    """
    Preprocesses text by inserting spaces at specific patterns to ensure proper tokenization.

    This function addresses common tokenization issues in text data by adding
    spaces around punctuation and operators that would otherwise be incorrectly tokenized.
    This is particularly useful for address data where punctuation often appears without
    proper spacing.

    Args:
        text (Union[str, pd.Series]): The input text to be preprocessed. Can be a single
                                    string or a pandas Series for vectorized processing.

    Returns:
        Union[str, pd.Series]: The preprocessed text with added spaces around specific
                             patterns and normalized whitespace. Returns same type as input.

    Examples:
        >>> preprocess_text_for_tokenization("123+456")
        '123 + 456'
        >>> preprocess_text_for_tokenization("semi-detached")
        'semi - detached'
        >>> preprocess_text_for_tokenization("london.Manchester")
        'london. Manchester'
        >>> preprocess_text_for_tokenization("(odd)33-45")
        '( odd ) 33-45'

        # Vectorized usage:
        >>> df['processed'] = preprocess_text_for_tokenization(df['text_column'])

    Note:
        The function applies transformations in a specific order and cleans up
        multiple spaces at the end. The transformations are designed to be
        idempotent - running the function multiple times on the same text
        should produce the same result.
    """

    # Define all regex patterns in order
    patterns_replacements = [
        # Pattern 1: Add spaces around operators/punctuation between numbers
        # e.g., "123+456" -> "123 + 456"
        (r"(?<=[0-9])([+\-\,*^\(\)])(?=[0-9-])", r" \1 "),
        # Pattern 2: Add space after period between lowercase/quotes and uppercase/quotes
        # e.g., "end.Start" -> "end. Start"
        (r'(?<=[a-z"\'])\.(?=[a-zA-Z"\'])', r". "),
        # Pattern 3: Add space after commas between alphabetic characters
        # e.g., "apple,banana" -> "apple, banana"
        (r"(?<=[a-zA-Z]),(?=[a-zA-Z])", r", "),
        # Pattern 4: Add spaces around hyphens between alphabetic characters
        # e.g., "semi-detached" -> "semi - detached"
        (r"(?<=[a-zA-Z])[-–—](?=[a-zA-Z])", r" - "),
        # Pattern 5 & 6: Add spaces around operators between alphanumeric and alphabetic
        # e.g., "(odd)33-45" -> " ( odd ) 33-45"
        # e.g., "value:123" -> "value : 123"
        (r"(?<=[a-zA-Z0-9])([:<>=/\(\)])(?=[a-zA-Z])", r" \1 "),
        (r"(?<=[a-zA-Z])([:<>=/\(\)])(?=[a-zA-Z0-9])", r" \1 "),
        # Additional patterns for production consistency
        # Separate parentheses from digits
        (r"(\))(\d)", r"\1 \2"),
        (r"(\d)(\()", r"\1 \2"),
        # Ensure space after punctuation (enhanced version)
        (r"([,;:])(?=\S)", r"\1 "),
        # Normalize hyphen spacing for ranges (preserve alphabetic hyphens handled above)
        (r"(\d+)\s*-\s*(\d+)", r"\1-\2"),
        # Clean up any multiple spaces
        (r"\s+", " "),
    ]

    # Handle both string and Series inputs
    if isinstance(text, pd.Series):
        # Vectorized processing for pandas Series
        processed_series = text.astype(str)

        # Apply all patterns sequentially using vectorized operations
        for pattern, replacement in patterns_replacements:
            processed_series = processed_series.str.replace(
                pattern, replacement, regex=True
            )

        return processed_series.str.strip()

    else:
        # Single string processing (original behavior)
        processed_text = str(text)

        # Apply all regex patterns
        for pattern, replacement in patterns_replacements:
            processed_text = re.sub(pattern, replacement, processed_text)

        return processed_text.strip()


def find_spans(df):
    """
    Calculate and start and end positions for text spans within property_address.

    Parameters:
    df (pandas.DataFrame): DataFrame containing 'text' and 'property_address' columns

    Returns:
    pandas.DataFrame: DataFrame with updated 'start' and 'end' columns
    """
    df = df.copy()  # Work with a copy to avoid modifying original

    for idx, row in df.iterrows():
        text = str(row["text"]).strip()
        property_address = str(row["property_address"]).strip()

        # Find the position of text in property_address
        start_pos = property_address.lower().find(text.lower())

        if start_pos != -1:
            # Text found - calculate start and end positions
            df.loc[idx, "start"] = start_pos
            df.loc[idx, "end"] = start_pos + len(text)
        else:
            # Text not found - try to find partial matches or handle edge cases
            # This could happen after preprocessing if text was modified
            df.loc[idx, "start"] = -1  # Indicate not found
            df.loc[idx, "end"] = -1

    return df


def validate_spans(df):
    """
    Validate that the calculated spans correctly extract the text from property_address.

    Parameters:
    df (pandas.DataFrame): DataFrame with 'text', 'property_address', 'start', 'end' columns

    Returns:
    pandas.DataFrame: DataFrame with additional 'span_valid' column indicating validation results
    """
    df = df.copy()

    def check_span(row):
        if row["start"] == -1 or row["end"] == -1:
            return False

        extracted_text = row["property_address"][row["start"] : row["end"]]
        return extracted_text.lower() == row["text"].lower()

    df["span_valid"] = df.apply(check_span, axis=1)
    return df


def dataframe_to_ner_format(df):
    """
    Convert a pandas DataFrame with NER annotations to the required format for training.

    Args:
        df: pandas DataFrame with columns: datapoint_id, start, end, label, text, property_address

    Returns:
        list: List of dictionaries in the format required for NER training
    """
    result = []

    # Group by datapoint_id to process each unique text
    grouped = df.groupby("datapoint_id")

    for datapoint_id, group in grouped:
        # Get the full text (should be the same for all rows in the group)
        full_text = group["property_address"].iloc[0]

        # Create spans list for this datapoint
        spans = []
        for _, row in group.iterrows():
            span = {
                "text": row["text"].strip(),  # Remove any leading/trailing whitespace
                "start": row["start"],
                "end": row["end"],
                "label": row["label"],
            }
            spans.append(span)

        # Create the entry for this datapoint
        entry = {"text": full_text, "spans": spans}

        result.append(entry)

    return result
