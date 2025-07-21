"""
Weak Labelling Data Processing Script for Property Address OCR/NER Training Data

This script processes property address data from the OCOD  dataset to create weakly labelled training data for OCR/NER
tasks. The script applies automated labelling functions to identify and annotate
entities within property addresses, then cleans and formats the results for
machine learning model training.

Workflow:
1. Load OCOD property address data from compressed CSV
2. Apply weak labelling functions in batches to extract entity spans
3. Clean labelled data by removing overlapping and zero-length spans
4. Convert processed annotations to structured DataFrame format
5. Save final training dataset as CSV

Input:
    - OCOD_FULL_2022_02.zip: Compressed CSV containing property address data

Output:
    - weakly_labelled.csv: Processed training data with entity annotations

Key Features:
    - Batch processing (5000 records per batch) for memory efficiency
    - Automatic span conflict resolution (overlapping/invalid spans)
    - Progress tracking with verbose output
    - Memory management with garbage collection
    - Warning suppression for pandas dtype downcasting

Dependencies:
    - pandas: Data manipulation and CSV I/O
    - pathlib: Cross-platform file path handling
    - tqdm: Progress bar visualization
    - enhance_ocod.labelling.weak_labelling: Core labelling functions

Configuration:
    - batch_size: 5000 (adjustable for memory constraints)
    - text_column: 'Property Address' (target column for labelling)
    - include_function_name: False (exclude labelling function metadata)
    - save_intermediate: False (no intermediate file saves)
    - verbose: True (enable progress reporting)

Note:
    Pandas downcasting warnings are suppressed as they are internal to the
    library and do not affect functionality.
"""

import pandas as pd
from pathlib import Path


# There is a warning related to bfill and ffill which is basically internal to pandas so silencing here
import warnings

warnings.filterwarnings("ignore", message=".*Downcasting object dtype arrays.*")

from enhance_ocod.labelling.weak_labelling import (
    process_dataframe_batch,
    convert_weakly_labelled_list_to_dataframe,
    remove_overlapping_spans,
    remove_zero_length_spans,
)

SCRIPT_DIR = Path("..").parent.absolute()

data_dir = SCRIPT_DIR.parent / "data"
zip_file = data_dir / "ocod_history" / "OCOD_FULL_2022_02.zip"
weakly_labelled_csv_save_path = data_dir / "training_data" / "weakly_labelled.csv"

df = pd.read_csv(zip_file)

weakly_labelled_dict = process_dataframe_batch(
    df,
    batch_size=5000,
    text_column="Property Address",
    include_function_name=False,
    save_intermediate=False,
    verbose=True,
)

remove_overlapping_spans(weakly_labelled_dict)

remove_zero_length_spans(weakly_labelled_dict)

processed_df = convert_weakly_labelled_list_to_dataframe(weakly_labelled_dict)

processed_df.to_csv(weakly_labelled_csv_save_path, index=False)
