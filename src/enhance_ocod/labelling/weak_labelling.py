"""
Weak labelling utilities for applying NER labelling functions to tabular data.

This module provides tools for batch processing pandas DataFrames using address entity labelling functions, handling large-scale weak supervision workflows. Features include:
    - Applying all labelling functions to each row in a DataFrame.
    - Batch processing with progress bars and intermediate saving.
    - Removal of overlapping entity spans for cleaner training data.
    - Generation of binary tags for special address cases (e.g., flats, commercial parks).

Intended for use in pipelines that prepare training data for NER models from weak, rule-based supervision.
"""

import pandas as pd
from typing import List, Dict, Any, Callable, Optional, Union
from tqdm import tqdm
import json
import logging
from pathlib import Path




from .ner_spans import lfs



def apply_labelling_functions_to_row(row: pd.Series, 
                                  functions: List[Callable] = lfs,
                                  text_column = 'text',
                                  include_function_name: bool = False) -> Dict[str, Any]:
    """
    Apply all labelling functions to a single row and return in training format.
    
    Args:
        row: pandas Series with 'text' column and any support columns
        functions: List of labelling functions to apply
        include_function_name: Whether to include which function generated each span (useful for debugging)
    
    Returns:
        Dict in training format with 'text' and 'spans' fields
    """
    
    result = {
        "text": row[text_column],
        "spans": []
    }
    
    for func in functions:
        try:
            # Apply the function
            spans = func(row)
            
            # Convert each span to training format
            for start, end, label in spans:
                span_dict = {
                    "text": row[text_column][start:end],
                    "start": start,
                    "end": end,
                    "label": label
                }
                
                # Optionally include which function generated this span
                if include_function_name:
                    span_dict["function"] = func.__name__
                
                result["spans"].append(span_dict)
                
        except Exception as e:
            # Log the error but continue with other functions
            logging.warning(f"Function {func.__name__} failed on row: {e}")
            continue
    
    return result


def process_dataframe_batch(df: pd.DataFrame, 
                           batch_size: int = 5000,
                           text_column: str = 'text',
                           include_function_name: bool = False,
                           save_intermediate: bool = False,
                           output_dir: Optional[str] = None,
                           verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Process a dataframe in batches, applying labelling functions to each row.
    
    Args:
        df: DataFrame with text and support columns
        batch_size: Number of rows to process at once
        text_column: Name of the column containing text
        include_function_name: Whether to include which function generated each span
        save_intermediate: Whether to save each batch to disk
        output_dir: Directory to save intermediate results (required if save_intermediate=True)
        verbose: Whether to show progress bars
    
    Returns:
        List of results in training format
    """
    
    if save_intermediate and not output_dir:
        raise ValueError("output_dir must be provided when save_intermediate=True")
    
    if save_intermediate:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_results = []
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    if verbose:
        batch_iterator = tqdm(range(0, len(df), batch_size), 
                            desc="Processing batches", 
                            total=total_batches)
    else:
        batch_iterator = range(0, len(df), batch_size)
    
    for batch_idx, start_idx in enumerate(batch_iterator):
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx].copy()

        batch_df['text'] = batch_df[text_column].fillna('')
        
        batch_results = []
        failed_rows = []
        
        # Process each row in the batch
        for datapoint_id, (_, row) in enumerate(batch_df.iterrows()):
            try:
                result = apply_labelling_functions_to_row(
                    row, 
                    include_function_name=include_function_name
                )
                
                # Add metadata for tracking
                result['datapoint_id'] = start_idx + datapoint_id  # Global row index
                batch_results.append(result)
                
            except Exception as e:
                failed_rows.append({
                    'datapoint_id': start_idx + datapoint_id,
                    'error': str(e),
                    'text': row.get(text_column, '')[:100] + '...'  # First 100 chars for debugging
                })
                logging.error(f"Failed to process row {start_idx + datapoint_id}: {e}")
        
        # Log batch statistics
        if verbose and failed_rows:
            logging.warning(f"Batch {batch_idx}: {len(failed_rows)} rows failed out of {len(batch_df)}")
        
        # Save intermediate results if requested
        if save_intermediate:
            batch_file = Path(output_dir) / f"batch_{batch_idx:04d}.json"
            with open(batch_file, 'w') as f:
                json.dump({
                    'batch_idx': batch_idx,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'results': batch_results,
                    'failed_rows': failed_rows
                }, f, indent=2)
        
        all_results.extend(batch_results)
    
    if verbose:
        total_spans = sum(len(result['spans']) for result in all_results)
        print(f"\nProcessing complete:")
        print(f"  - Processed {len(all_results)} rows successfully")
        print(f"  - Found {total_spans} total spans")
        print(f"  - Average {total_spans/len(all_results):.2f} spans per row")
    
    return all_results


# Function to load and combine intermediate results
def load_intermediate_results(output_dir: str) -> List[Dict[str, Any]]:
    """Load and combine results from intermediate batch files"""
    
    output_path = Path(output_dir)
    batch_files = sorted(output_path.glob("batch_*.json"))
    
    all_results = []
    for batch_file in tqdm(batch_files, desc="Loading batches"):
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
            all_results.extend(batch_data['results'])
    
    return all_results

def remove_zero_length_spans(results: List[Dict[str, Any]]) -> None:
    """
    Remove all zero-length spans (where start == end) as they are not valid entities.
    This is much simpler than handling them in overlap detection.
    """
    for result in results:
        # Filter out zero-length spans
        result['spans'] = [span for span in result['spans'] 
                          if span['start'] < span['end']]

def remove_overlapping_spans(results: List[Dict[str, Any]]) -> None:
    """
    In-place version that modifies the input list directly.
    More memory efficient for very large datasets.
    """
    
    for result in results:
        spans = result['spans']
        
        if len(spans) <= 1:
            continue
        
        # Sort by length descending
        spans.sort(key=lambda x: x['end'] - x['start'], reverse=True)
        
        kept_spans = []
        
        for span in spans:
            # Check overlap with kept spans
            if not any(not (span['end'] <= kept['start'] or kept['end'] <= span['start']) 
                      for kept in kept_spans):
                kept_spans.append(span)
        
        # Sort by start position
        kept_spans.sort(key=lambda x: x['start'])
        result['spans'] = kept_spans

def get_overlap_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get statistics about overlaps before removal (for debugging)"""
    
    total_rows = len(results)
    rows_with_overlaps = 0
    total_spans_before = 0
    total_overlaps_removed = 0
    
    for result in results:
        spans = result['spans'][:]  # Copy to avoid modifying original
        total_spans_before += len(spans)
        
        if len(spans) <= 1:
            continue
            
        # Check for overlaps
        spans.sort(key=lambda x: x['start'])
        has_overlap = False
        
        for i in range(len(spans) - 1):
            if spans[i]['end'] > spans[i + 1]['start']:
                has_overlap = True
                break
        
        if has_overlap:
            rows_with_overlaps += 1
            
            # Count overlaps by applying removal algorithm
            spans.sort(key=lambda x: x['end'] - x['start'], reverse=True)
            kept = []
            
            for span in spans:
                if not any(not (span['end'] <= k['start'] or k['end'] <= span['start']) 
                          for k in kept):
                    kept.append(span)
            
            total_overlaps_removed += len(spans) - len(kept)
    
    return {
        'total_rows': total_rows,
        'rows_with_overlaps': rows_with_overlaps,
        'overlap_percentage': (rows_with_overlaps / total_rows) * 100,
        'total_spans_before': total_spans_before,
        'total_overlaps_removed': total_overlaps_removed,
        'spans_after': total_spans_before - total_overlaps_removed
    }

def create_flat_tag(df, text_column='property_address'):
    """Create binary flat_tag based on presence of flat/apartment indicators"""
    df['flat_tag'] = df[text_column].str.contains(
        r'\b(apartment|flat|penthouse|unit)\b',
        case=False, na=False, regex=True
    )
    return df

def create_commercial_park_tag(df, text_column='property_address'):
    """Create binary commercial_park_tag based on presence of commercial park indicators"""
    df['commercial_park_tag'] = df[text_column].str.contains(
            r'\b(business\s+park|industrial\s+park|commercial\s+park|office\s+park|'
            r'technology\s+park|science\s+park|enterprise\s+park|trading\s+estate|'
            r'business\s+estate|industrial\s+estate|retail\s+park|tech\s+park|'
            r'innovation\s+park|corporate\s+park)\b',
            case=False, na=False, regex=True
        )
    return df



def direct_entity_evaluation(ground_truth_data, predicted_data):
    """
    Direct entity-level evaluation without BIO token conversion.
    
    Args:
        ground_truth_data: List of dicts with 'text' and 'spans'
        predicted_data: List of dicts with 'text' and 'spans'
    
    Returns:
        Dict with overall and per-class metrics in seqeval-compatible format
    """
    
    # Convert to entity sets for each text
    def extract_entities(data):
        entity_sets = []
        for item in data:
            entities = set()
            for span in item['spans']:
                # Create entity tuple: (start, end, label) - make sure label is string
                entities.add((span['start'], span['end'], str(span['label'])))
            entity_sets.append(entities)
        return entity_sets
    
    true_entities = extract_entities(ground_truth_data)
    pred_entities = extract_entities(predicted_data)
    
    # Calculate overall metrics
    all_true = set()
    all_pred = set()
    
    for true_set, pred_set in zip(true_entities, pred_entities):
        all_true.update(true_set)
        all_pred.update(pred_set)
    
    tp = len(all_true & all_pred)
    fp = len(all_pred - all_true)
    fn = len(all_true - all_pred)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate per-class metrics
    class_metrics = {}
    all_labels = set()
    
    # Collect all unique labels - be explicit about extracting strings
    for entities in true_entities + pred_entities:
        for start, end, label in entities:
            all_labels.add(str(label))  # Ensure it's a string
    
    for label in all_labels:
        true_label = set()
        pred_label = set()
        
        for true_set, pred_set in zip(true_entities, pred_entities):
            # Filter entities for this specific label
            true_label.update((s, e, l) for s, e, l in true_set if str(l) == label)
            pred_label.update((s, e, l) for s, e, l in pred_set if str(l) == label)
        
        tp_label = len(true_label & pred_label)
        fp_label = len(pred_label - true_label)
        fn_label = len(true_label - pred_label)
        
        prec_label = tp_label / (tp_label + fp_label) if (tp_label + fp_label) > 0 else 0
        rec_label = tp_label / (tp_label + fn_label) if (tp_label + fn_label) > 0 else 0
        f1_label = 2 * prec_label * rec_label / (prec_label + rec_label) if (prec_label + rec_label) > 0 else 0
        
        class_metrics[label] = {
            'precision': prec_label,
            'recall': rec_label,
            'f1-score': f1_label,
            'support': len(true_label)
        }
    
    # Return in seqeval-compatible format
    return {
        'overall_precision': precision,
        'overall_recall': recall,
        'overall_f1': f1,
        'per_class': class_metrics
    }

def convert_weak_labels_to_standard_format(noisy_data):
    """
    Convert weakly-supervised or noisy NER annotations to standardized evaluation format.
    
    This function cleans and standardizes NER data from noisy labelling processes by:
    - Removing duplicate entity spans within each text
    - Filtering out extraneous fields (e.g., datapoint_id, confidence scores)
    - Normalizing the data structure to match evaluation pipeline requirements
    
    The function is designed to handle common issues in weakly-supervised NER data:
    - Multiple identical annotations for the same entity span
    - Inconsistent field naming across different annotation sources
    - Extra metadata fields that interfere with evaluation
    """
    standardized = []
    
    for item in noisy_data:
        # Remove zero-length spans first
        valid_spans = [span for span in item['spans'] 
                      if span['start'] < span['end']]
        
        # Then remove duplicates
        seen_spans = set()
        unique_spans = []
        
        for span in valid_spans:
            span_key = (span['start'], span['end'], span['label'])
            if span_key not in seen_spans:
                seen_spans.add(span_key)
                unique_spans.append({
                    'start': span['start'],
                    'end': span['end'], 
                    'label': span['label']
                })
        
        standardized.append({
            'text': item['text'],
            'spans': unique_spans
        })
    
    return standardized

def evaluate_weak_labels(noisy_predictions, ground_truth_path, output_dir, dataset_name="noisy_system"):
    """
    Evaluate noisy NER predictions using direct entity-level comparison.
    """
    print(f"Evaluating noisy predictions on {dataset_name} set...")
    
    # Load ground truth
    with open(ground_truth_path, 'r') as f:
        ground_truth_data = json.load(f)
    
    # Convert both to standard format and remove duplicates
    ground_truth_clean = convert_weak_labels_to_standard_format(ground_truth_data)
    noisy_predictions_clean = convert_weak_labels_to_standard_format(noisy_predictions)
    
    print(f"Loaded {len(ground_truth_clean)} ground truth examples")
    print(f"Loaded {len(noisy_predictions_clean)} predictions")
    
    # Direct entity-level evaluation
    results = direct_entity_evaluation(ground_truth_clean, noisy_predictions_clean)
    
    # Print results
    print(f"\n{dataset_name.upper()} SET RESULTS:")
    print(f"Overall F1: {results['overall_f1']:.4f}")
    print(f"Overall Precision: {results['overall_precision']:.4f}")  
    print(f"Overall Recall: {results['overall_recall']:.4f}")
    print("\nPer-class results:")
    
    for label, metrics in results['per_class'].items():
        print(f"{label:15s} precision: {metrics['precision']:.4f}  recall: {metrics['recall']:.4f}  f1: {metrics['f1-score']:.4f}  support: {metrics['support']}")
    
    # Save results (same format as original)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Overall performance CSV
    overall_results = pd.DataFrame([{
        'model_path': "noisy_ner_system",
        'dataset': dataset_name,
        'num_examples': len(ground_truth_clean),
        'precision': results['overall_precision'],
        'recall': results['overall_recall'],
        'f1_score': results['overall_f1'],
        'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }])
    
    overall_file = output_path / f"{dataset_name}_overall_performance.csv"
    overall_results.to_csv(overall_file, index=False)
    
    # Class-wise performance CSV
    class_results = []
    for class_name, metrics in results['per_class'].items():
        class_results.append({
            'model_path': "noisy_ner_system",
            'dataset': dataset_name,
            'class_name': class_name,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1-score'],
            'support': metrics['support'],
            'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    class_df = pd.DataFrame(class_results)
    class_file = output_path / f"{dataset_name}_class_performance.csv"
    class_df.to_csv(class_file, index=False)
    
    print("\nResults saved:")
    print(f"  Overall: {overall_file}")
    print(f"  Per-class: {class_file}")
    
    return overall_results, class_df
