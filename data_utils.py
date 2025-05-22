import json
from pathlib import Path
from typing import List, Dict, Any, Iterator
import random
import pandas as pd

def load_data(file_path: str, max_samples: int = None) -> List[Dict]:
    """Load data from JSON file in GLiNER format."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    # If max_samples is specified, limit the number of samples
    if max_samples:
        data = data[:max_samples]
        
    return data

def save_data(data: List[Dict], file_path: str) -> None:
    """Save data to JSONL file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def train_val_split(data: List[Dict], val_ratio: float = 0.1) -> tuple:
    """Split data into training and validation sets."""
    random.shuffle(data)
    split_idx = int(len(data) * (1 - val_ratio))

    return data[:split_idx], data[split_idx:]



def convert_df_to_gliner_format(df: pd.DataFrame) -> List[Dict]:
    """
    Convert DataFrame data in the given format to GLiNER training format.
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        List of dictionaries in GLiNER format
    """
    
    # Group by the input text and datapoint_id, but deduplicate spans
    grouped_data = {}
    for _, row in df.iterrows():
        key = (row['input:text'], row['input:datapoint_id'])
        
        if key not in grouped_data:
            grouped_data[key] = {
                'text': row['input:text'],
                'spans': set()  # Using a set to prevent duplicates
            }
        
        # Convert span to a hashable tuple to use in a set
        span_tuple = (row['text'], row['start'], row['end'], row['label'])
        grouped_data[key]['spans'].add(span_tuple)
    
    # Convert back to the GLiNER format
    gliner_data = []
    for data in grouped_data.values():
        # Convert set of tuples back to list of dictionaries
        spans = [
            {'text': text, 'start': start, 'end': end, 'label': label}
            for text, start, end, label in data['spans']
        ]
        
        gliner_data.append({
            'text': data['text'],
            'spans': spans
        })
    
    return gliner_data
