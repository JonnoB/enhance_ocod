"""
Data Processing Script for NER Training Data Preparation

This script processes CSV files containing labeled text data and converts them into 
formats suitable for Named Entity Recognition (NER) model training and inference.

The script performs the following operations for each input file:
1. Loads CSV data containing text and property address labels
2. Creates original NER-ready JSON format data
3. Preprocesses text data for tokenization
4. Finds and validates text spans for entity extraction
5. Creates preprocessed NER-ready JSON format data
6. Adds placeholder columns and saves inference-ready CSV data

Input Files:
    - ground_truth_test_set_labels.csv
    - ground_truth_dev_set_labels.csv  
    - weakly_labelled.csv

Output Structure:
    data/training_data/
    ├── ner_ready/                    # Original NER training data (JSON)
    ├── ner_ready_preprocessed/       # Preprocessed NER training data (JSON)
    └── inference_ready/              # CSV files with additional columns for inference

Dependencies:
    - pandas: For CSV data manipulation
    - pathlib: For cross-platform file path handling
    - json: For JSON output formatting
    - enhance_ocod.preprocess: Custom preprocessing functions
        - find_spans: Locates entity spans in text
        - preprocess_text_for_tokenization: Cleans text for NER tokenization
        - validate_spans: Validates entity span accuracy
        - dataframe_to_ner_format: Converts DataFrame to NER JSON format

Expected CSV Input Columns:
    - text: Main text content for entity extraction
    - property_address: Property address text
    - datapoint_id: Unique identifier for each data point

Output CSV Additional Columns:
    - title_number: Set to datapoint_id value
    - tenure: Set to "freehold" (placeholder)
    - district: Set to "westminster" (placeholder) 
    - county: Set to "london" (placeholder)
    - region: Set to "london" (placeholder)
    - price_paid: Set to 100 (placeholder)

Usage:
    Run this script from the command line or import and execute.
    Ensure the input CSV files exist in PROJECT_ROOT/data/training_data/
    
Example:
    python data_processing_script.py

Notes:
    - Creates output directories automatically if they don't exist
    - Prints span validity statistics for quality assessment
    - Additional columns added to inference data are placeholder values
    - Script assumes it's run from within the project structure
"""

from pathlib import Path
import pandas as pd
import json
from enhance_ocod.preprocess import (
    find_spans,
    preprocess_text_for_tokenization,
    validate_spans,
    dataframe_to_ner_format,
)

# Get the script directory and go up one level to the project root
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent  # This gets you to the enhance_ocod directory

# Define the files to process
files_to_process = [
    "ground_truth_test_set_labels.csv",
    "ground_truth_dev_set_labels.csv",
    "weakly_labelled.csv"
]

for filename in files_to_process:
    print(f"Processing {filename}...")

    # Read the CSV file - now using PROJECT_ROOT instead of SCRIPT_DIR
    df = pd.read_csv(PROJECT_ROOT / "data/training_data" / filename)

    # Extract base name without extension for output files
    base_name = Path(filename).stem

    # Create data for training ner training (original)
    output_path = PROJECT_ROOT / "data/training_data/ner_ready" / f"{base_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(dataframe_to_ner_format(df), f, indent=2)

    # Process preprocessed version
    df_preproc = df.copy()
    df_preproc["text"] = preprocess_text_for_tokenization(df_preproc["text"])
    df_preproc["property_address"] = preprocess_text_for_tokenization(
        df_preproc["property_address"]
    )
    df_preproc = find_spans(df_preproc)
    df_preproc2 = validate_spans(df_preproc)
    print(f"{filename} - Span validity counts:")
    print(df_preproc2.groupby("span_valid").size())
    print()

    # Create data for training ner training (preprocessed)
    output_path_preproc = (
        PROJECT_ROOT / "data/training_data/ner_ready_preprocessed" / f"{base_name}.json"
    )
    output_path_preproc.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path_preproc, "w") as f:
        json.dump(dataframe_to_ner_format(df_preproc), f, indent=2)

    # Add additional columns this is just filler so the code runs it does not impact the outcome.
    df["title_number"] = df["datapoint_id"]
    df["tenure"] = "freehold"
    df["district"] = "westminster"
    df["county"] = "london"
    df["region"] = "london"
    df["price_paid"] = 100

    # Save inference ready data
    output_csv_path = PROJECT_ROOT / "data/training_data/inference_ready" / filename
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    print(f"Completed processing {filename}")
    print("-" * 50)
