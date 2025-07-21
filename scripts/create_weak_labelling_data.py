import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import gc  # Add for memory management

import pickle
import json
from pathlib import Path

# There is a warning related to bfill and ffill which is basically internal to pandas so silencing here
import warnings
warnings.filterwarnings('ignore', message='.*Downcasting object dtype arrays.*')

from enhance_ocod.labelling.weak_labelling import (process_dataframe_batch, convert_weakly_labelled_list_to_dataframe,
 remove_overlapping_spans, remove_zero_length_spans)

SCRIPT_DIR = Path('..').parent.absolute()

data_dir =  SCRIPT_DIR.parent / "data"
zip_file =  data_dir / "ocod_history" / "OCOD_FULL_2022_02.zip"
weakly_labelled_csv_save_path = data_dir / "training_data" / "weakly_labelled.csv"

df = pd.read_csv(zip_file)

weakly_labelled_dict = process_dataframe_batch(df, 
                           batch_size = 5000,
                           text_column  = 'Property Address',
                           include_function_name  = False,
                           save_intermediate  = False,
                           verbose  = True)
                           
remove_overlapping_spans(weakly_labelled_dict)

remove_zero_length_spans(weakly_labelled_dict)

processed_df = convert_weakly_labelled_list_to_dataframe(weakly_labelled_dict)

processed_df.to_csv(weakly_labelled_csv_save_path, index = False)

