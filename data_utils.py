import json
from pathlib import Path
from typing import List, Dict, Any, Iterator
import random
import pandas as pd
from torch.utils.data import Dataset
from nervaluate import Evaluator
import re
from pre_processing_funcs import preprocess_training_data
import os

def load_data(file_path: str, max_samples: int = None) -> List[Dict]:
    """
    Load data with preprocessing support.
    
    Automatically generates preprocessed path by adding 'preprocessed_data' folder.
    If preprocessed file exists, load it.
    Otherwise, load raw data, preprocess it, save it, then return it.
    """
    
    # Generate preprocessed path
    path = Path(file_path)
    preprocessed_path = path.parent / "preprocessed_data" / path.name
    
    if preprocessed_path.exists():
        print(preprocessed_path)
        print('Loading pre-processed data')
        with open(preprocessed_path, 'r') as f:
            data = json.load(f)
    else:
        # Load raw data
        with open(file_path, 'r') as f:
            data = json.load(f)
        print("pre-processing data")
        # Preprocess the data
        data = preprocess_training_data(data)
        print("pre-processing data complete")
        # Create directory if it doesn't exist
        preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessed data
        with open(preprocessed_path, 'w') as f:
            json.dump(data, f)
    
    # Apply max_samples limit if specified
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


class GLiNERDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Convert from {'text': ..., 'spans': [...]} to {'text': ..., 'ner': [...]}
        ner_spans = []
        for span in item.get('spans', []):
            ner_spans.append([
                span['start'],        # Start position
                span['end'],          # End position
                span['label'].upper() # Convert label to uppercase (GLiNER convention)
            ])
        
        return {
            'tokenized_text': item['text'],
            'ner': ner_spans  # This is the key format GLiNER expects
        }


def evaluate_ner_spans(model, eval_data):
    """
    NER evaluation using nervaluate library
    """
    # Get all unique labels
    all_labels = sorted(set(span['label'] for example in eval_data 
                           for span in example['spans']))
    
    true_spans = []
    pred_spans = []
    
    for example in eval_data:
        text = example['text']
        
        # Convert true spans to nervaluate format
        true_example = [
            {
                'label': span['label'],
                'start': span['start'],
                'end': span['end']
            }
            for span in example['spans']
        ]
        true_spans.append(true_example)
        
        # Get predictions
        try:
            predicted_entities = model.predict_entities(text, all_labels, threshold=0.5)
            pred_example = []
            
            for pred in predicted_entities:
                pred_example.append({
                    'label': pred.label if hasattr(pred, 'label') else pred['label'],
                    'start': pred.start if hasattr(pred, 'start') else pred['start'],
                    'end': pred.end if hasattr(pred, 'end') else pred['end']
                })
                
        except Exception as e:
            print(f"Error predicting: {e}")
            pred_example = []
            
        pred_spans.append(pred_example)
    
    # Evaluate - note that evaluate() returns 4 values
    evaluator = Evaluator(true_spans, pred_spans, tags=all_labels)
    results, results_per_tag, result_indices, result_indices_by_tag = evaluator.evaluate()
    
    # Print overall results
    print("\n" + "="*60)
    print("NER EVALUATION RESULTS")
    print("="*60)
    
    # Print metrics for each evaluation type
    for metric_type, scores in results.items():
        print(f"\n{metric_type.upper()} matching:")
        if 'precision' in scores:
            print(f"  Precision: {scores['precision']:.4f}")
            print(f"  Recall: {scores['recall']:.4f}")
            print(f"  F1: {(2 * scores['precision'] * scores['recall'] / (scores['precision'] + scores['recall']) if scores['precision'] + scores['recall'] > 0 else 0):.4f}")
        print(f"  Correct: {scores['correct']}")
        print(f"  Incorrect: {scores['incorrect']}")
        print(f"  Missed: {scores['missed']}")
        print(f"  Spurious: {scores['spurious']}")
    
    # Print per-tag results
    if results_per_tag:
        print("\n" + "-"*60)
        print("PER-CLASS RESULTS (strict matching):")
        print("-"*60)
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
        print("-" * 65)
        
        for tag, tag_results in results_per_tag.items():
            strict_scores = tag_results.get('strict', {})
            precision = strict_scores.get('precision', 0)
            recall = strict_scores.get('recall', 0)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            support = strict_scores.get('possible', 0)
            
            print(f"{tag:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")
    
    # Return results for saving
    return {
        'overall': results,
        'per_tag': results_per_tag,
        'tags': all_labels
    }

def convert_to_nervaluate_sorted(df):
    """
    Convert a pandas DataFrame to nervaluate format with sorted spans.
    
    Args:
        df: pandas DataFrame with columns: text, start, end, label, input:text
    
    Returns:
        List of dictionaries in nervaluate format
    """
    # Group by the full text (input:text column)
    grouped = df.groupby('input:text')
    
    result = []
    
    for text, group in grouped:
        # Create spans list for this text
        spans = []
        for _, row in group.iterrows():
            span = {
                "text": row['text'],
                "start": int(row['start']),
                "end": int(row['end']),
                "label": row['label']
            }
            spans.append(span)
        
        # Sort spans by start position
        spans.sort(key=lambda x: x['start'])
        
        # Create the document object
        doc = {
            "text": text,
            "spans": spans
        }
        result.append(doc)
    
    return result

def gliner_pred_fn(gliner_model, ocod_data):
    """
    Predicts over the OCOD dataframe using a GLiNER model.
    Includes preprocessing to handle tokenization issues.
    """
    from gliner import GLiNER
    
    print('Loading the GLiNER model')
    model = GLiNER.from_pretrained(gliner_model)
    
    # Define your entity labels for GLiNER
    labels = ["YOUR", "ENTITY", "LABELS", "HERE"]  # Replace with your actual labels
    
    all_entities = []
    
    print('Preprocessing and predicting over the OCOD dataset using GLiNER')
    for idx, row in ocod_data.iterrows():
        # Preprocess the text
        preprocessed_text = preprocess_text_for_tokenization(row['property_address'])
        
        # Get predictions from GLiNER
        entities = model.predict_entities(preprocessed_text, labels, threshold=0.5)
        
        # Format entities similar to your original output
        for entity in entities:
            entity_data = {
                'text': preprocessed_text,
                'datapoint_id': idx,
                'title_number': str(row['title_number']),
                'start': entity['start'],
                'end': entity['end'],
                'label': entity['label'],
                'label_text': entity['text'],
                'score': entity['score']  # GLiNER provides confidence scores
            }
            all_entities.append(entity_data)
    
    # Convert to DataFrame
    all_entities_df = pd.DataFrame(all_entities)
    
    if not all_entities_df.empty:
        all_entities_df['label_id_count'] = all_entities_df.groupby(['datapoint_id', 'label']).cumcount()
    
    print('Named Entity Recognition labelling complete')
    return all_entities_df