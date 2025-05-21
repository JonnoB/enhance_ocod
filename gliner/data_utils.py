import json
from pathlib import Path
from typing import List, Dict, Any, Iterator
import random

def load_data(file_path: str, max_samples: int = None) -> List[Dict]:
    """Load data from JSONL file in GLiNER format."""
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line))
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