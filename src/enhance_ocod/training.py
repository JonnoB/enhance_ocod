
"""
Named Entity Recognition (NER) Data Processing and Evaluation Module

This module provides a complete pipeline for processing Named Entity Recognition (NER) data
and evaluating trained models. It handles data preparation, tokenization, label alignment,
dataset creation, and model performance evaluation using the BIO (Begin-Inside-Outside) 
tagging scheme.

Key Features:
- Automatic alignment of character-level entity spans with subword tokens
- Support for any transformer-based tokenizer from Hugging Face
- BIO tagging scheme implementation with proper B-/I- tag assignment
- Comprehensive model evaluation with entity-level metrics
- Export of evaluation results to CSV format for analysis

Classes:
    NERDataProcessor: Main class for processing NER data and creating datasets

Functions:
    create_label_list: Utility function to generate BIO label lists from entity types
    evaluate_model_performance: Comprehensive model evaluation with CSV output

Dependencies:
    - transformers: For model and tokenizer loading
    - datasets: For HuggingFace dataset creation
    - seqeval: For entity-level evaluation metrics
    - torch: For model inference
    - pandas: For results export
    - numpy: For array operations

Data Format:
    Expected JSON input format for NER data:
    [
        {
            "text": "John Smith works at Google in California",
            "spans": [
                {"start": 0, "end": 10, "label": "PERSON"},
                {"start": 20, "end": 26, "label": "ORG"},
                {"start": 30, "end": 40, "label": "LOC"}
            ]
        }
    ]

Example Usage:
    >>> # Create label list
    >>> entity_types = ['PERSON', 'ORG', 'LOC']
    >>> labels = create_label_list(entity_types)
    >>> 
    >>> # Initialize processor
    >>> processor = NERDataProcessor(labels)
    >>> 
    >>> # Load and process data
    >>> train_data = processor.load_json_data('train.json')
    >>> train_dataset = processor.create_dataset(train_data, max_length=128)
    >>> 
    >>> # Evaluate trained model
    >>> evaluate_model_performance(
    ...     model_path='./trained_model',
    ...     data_path='test.json',
    ...     output_dir='./results',
    ...     dataset_name='test'
    ... )

Author: [Your Name]
Version: 1.0
License: [Your License]
"""

import json
from typing import List, Dict
from datasets import Dataset
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
import torch
from pathlib import Path
from transformers import AutoModelForTokenClassification, AutoTokenizer
import pandas as pd 



class NERDataProcessor:
    """
    A data processor for Named Entity Recognition (NER) tasks using transformers.
    
    This class handles the complete pipeline for preparing NER data for training/inference,
    including tokenization, label alignment, and dataset creation. It uses BIO (Begin-Inside-Outside)
    tagging scheme for entity labeling.
    
    Attributes:
        label_list (List[str]): List of all possible labels including 'O', 'B-', and 'I-' tags
        label2id (Dict[str, int]): Mapping from label names to integer IDs
        id2label (Dict[int, str]): Mapping from integer IDs to label names  
        tokenizer: Hugging Face tokenizer for text tokenization
    
    Args:
        label_list (List[str]): Complete list of NER labels (e.g., ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC'])
        tokenizer_name (str, optional): Name/path of the tokenizer model. 
            Defaults to "answerdotai/ModernBERT-base"
    
    Example:
        >>> labels = ['O', 'B-PERSON', 'I-PERSON', 'B-LOCATION', 'I-LOCATION']
        >>> processor = NERDataProcessor(labels)
        >>> data = processor.load_json_data('train.json')
        >>> dataset = processor.create_dataset(data, max_length=128)
    
    Note:
        Expected JSON data format:
        [
            {
                "text": "John lives in New York",
                "spans": [
                    {"start": 0, "end": 4, "label": "PERSON"},
                    {"start": 14, "end": 22, "label": "LOCATION"}
                ]
            }
        ]
    """
    
    def __init__(self, label_list: List[str], tokenizer_name: str = "answerdotai/ModernBERT-base"):
        """
        Initialize the NERDataProcessor.

        Args:
            label_list (List[str]): List of all NER labels (BIO format).
            tokenizer_name (str, optional): Name or path of the Hugging Face tokenizer.
                Defaults to "answerdotai/ModernBERT-base".
        """
        self.label_list = label_list
        self.label2id = {label: i for i, label in enumerate(label_list)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def load_json_data(self, file_path: str) -> List[Dict]:
        """
        Load NER data from a JSON file.

        Args:
            file_path (str): Path to the JSON file containing NER data.

        Returns:
            List[Dict]: List of data examples, each with 'text' and 'spans' fields.
        """
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def align_labels_with_tokens(self, text: str, spans: List[Dict], tokenized_inputs) -> List[int]:
        """
        Align character-level entity spans with token-level labels for NER tasks.

        Args:
            text (str): The input text sequence.
            spans (List[Dict]): List of entity spans with 'start', 'end', and 'label'.
            tokenized_inputs: Output from the tokenizer for the input text.

        Returns:
            List[int]: List of label IDs aligned to each token. Non-entity tokens are set to 'O'.
        """
        # Initialize all labels as 'O' (outside)
        labels = [-100] * len(tokenized_inputs["input_ids"])
        
        # Get word IDs to handle subword tokens
        word_ids = tokenized_inputs.word_ids()
        
        # Create character to span mapping
        char_to_span = {}
        for span in spans:
            for char_idx in range(span['start'], span['end']):
                char_to_span[char_idx] = span
        
        # Map tokens to labels
        current_entity_span = None  # Track the actual entity span, not just any span
        
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:  # Special tokens ([CLS], [SEP], [PAD])
                labels[token_idx] = -100
                continue
                
            # Get character span for this token
            try:
                char_span = tokenized_inputs.token_to_chars(token_idx)
                if char_span is None:
                    labels[token_idx] = self.label2id['O']
                    continue
                    
                token_start = char_span.start
                token_end = char_span.end
            except Exception:
                labels[token_idx] = self.label2id['O']
                continue
            
            # Check if this token overlaps with any entity span
            entity_span = None
            for char_idx in range(token_start, token_end):
                if char_idx in char_to_span:
                    entity_span = char_to_span[char_idx]
                    break
            
            if entity_span is None:
                # Token is outside any entity
                labels[token_idx] = self.label2id['O']
                current_entity_span = None
            else:
                # Token is inside an entity
                entity_label = entity_span['label']
                
                # FIXED LOGIC: Only use B- tag at the very start of an entity span
                # OR when switching between different entity spans
                if (current_entity_span is None or 
                    current_entity_span != entity_span or
                    token_start == entity_span['start']):
                    # This is the beginning of a new entity
                    labels[token_idx] = self.label2id.get(f'B-{entity_label}', self.label2id['O'])
                    current_entity_span = entity_span
                else:
                    # This is a continuation of the current entity
                    labels[token_idx] = self.label2id.get(f'I-{entity_label}', self.label2id['O'])
        
        return labels
    
    def process_examples(self, examples: List[Dict], max_length: int = 128) -> Dict[str, List]:
        """
        Process a batch of NER examples into model-ready tokenized inputs and aligned labels.

        Args:
            examples (List[Dict]): List of data examples, each with 'text' and 'spans'.
            max_length (int, optional): Maximum sequence length. Defaults to 128.

        Returns:
            Dict[str, List]: Dictionary with tokenized inputs and label IDs for each example.
        """
        texts = [ex['text'] for ex in examples]
        all_spans = [ex['spans'] for ex in examples]
        
        # Tokenize all texts
        tokenized_inputs = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        
        # Align labels for each example
        labels = []
        for i, (text, spans) in enumerate(zip(texts, all_spans)):
            # Get tokenization for this specific example
            example_encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )
            
            label_ids = self.align_labels_with_tokens(text, spans, example_encoding)
            # Pad labels to max_length
            label_ids = label_ids[:max_length]
            label_ids += [-100] * (max_length - len(label_ids))
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        # Remove offset_mapping as it's not needed for training
        tokenized_inputs.pop("offset_mapping", None)
        
        return tokenized_inputs
    
    def create_dataset(self, data: List[Dict], max_length: int = 128) -> Dataset:
        """
        Create a HuggingFace Dataset from a list of NER data examples.

        Args:
            data (List[Dict]): List of examples, each with 'text' and 'spans'.
            max_length (int, optional): Maximum sequence length. Defaults to 128.

        Returns:
            Dataset: HuggingFace Dataset ready for model training or evaluation.
        """
        # Process in batches for efficiency
        batch_size = 32
        all_processed = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            processed = self.process_examples(batch, max_length)
            
            # Convert to list of dicts
            for j in range(len(batch)):
                example = {
                    'input_ids': processed['input_ids'][j],
                    'attention_mask': processed['attention_mask'][j],
                    'labels': processed['labels'][j]
                }
                all_processed.append(example)
        
        return Dataset.from_list(all_processed)

    def compute_entity_metrics(self, eval_pred):
        """
        Compute entity-level precision, recall, and F1-score using seqeval.

        Args:
            eval_pred: Tuple of (predictions, labels) from a model evaluation step.
                - predictions: np.ndarray of shape (batch_size, seq_len, num_labels)
                - labels: np.ndarray of shape (batch_size, seq_len)

        Returns:
            dict: Dictionary with 'precision', 'recall', and 'f1' scores.
        """
        from seqeval.metrics import f1_score, precision_score, recall_score
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for l in label if l != -100]
            for label in labels
        ]
        
        return {
            'precision': precision_score(true_labels, true_predictions),
            'recall': recall_score(true_labels, true_predictions), 
            'f1': f1_score(true_labels, true_predictions),
        }

def create_label_list(entity_types: List[str]) -> List[str]:
    """
    Create a list of BIO-format NER labels from a list of entity types.

    Args:
        entity_types (List[str]): List of entity type strings (e.g., ["PERSON", "ORG"]).

    Returns:
        List[str]: List of labels in BIO format (e.g., ["O", "B-PERSON", "I-PERSON", ...]).
    """
    label_list = ['O']  # Outside any entity
    for entity_type in entity_types:
        label_list.append(f'B-{entity_type}')  # Beginning
        label_list.append(f'I-{entity_type}')  # Inside
    return label_list


def _evaluate_predictions_core(y_true, y_pred, system_identifier, dataset_name, num_examples, output_dir):
    """
    Core evaluation logic shared by both evaluation functions.
    
    Args:
        y_true: List of true label sequences (seqeval format)
        y_pred: List of predicted label sequences (seqeval format)
        system_identifier: String identifier for the system being evaluated
        dataset_name: Name of the dataset
        num_examples: Number of examples evaluated
        output_dir: Directory to save results
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (overall_results, class_df)
    """
    from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
    
    # Calculate metrics
    overall_f1 = f1_score(y_true, y_pred)
    overall_precision = precision_score(y_true, y_pred)
    overall_recall = recall_score(y_true, y_pred)
    
    # Get detailed classification report
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    
    # Print results (same format as original)
    print(f"\n{dataset_name.upper()} SET RESULTS:")
    print(f"Overall F1: {overall_f1:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")  
    print(f"Overall Recall: {overall_recall:.4f}")
    print("\nPer-class results:")
    print(classification_report(y_true, y_pred))
    
    # Save results (same format as original)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Overall performance CSV (same format)
    overall_results = pd.DataFrame([{
        'model_path': system_identifier,  # Using system_identifier instead of model_path
        'dataset': dataset_name,
        'num_examples': num_examples,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': overall_f1,
        'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }])
    
    overall_file = output_path / f"{dataset_name}_overall_performance.csv"
    overall_results.to_csv(overall_file, index=False)
    
    # 2. Class-wise performance CSV (same format)
    class_results = []
    for class_name, metrics in report_dict.items():
        if isinstance(metrics, dict):  # Skip accuracy (which is just a float)
            class_results.append({
                'model_path': system_identifier,
                'dataset': dataset_name,
                'class_name': class_name,
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1-score', 0),
                'support': metrics.get('support', 0),
                'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    class_df = pd.DataFrame(class_results)
    class_file = output_path / f"{dataset_name}_class_performance.csv"
    class_df.to_csv(class_file, index=False)
    
    print("\nResults saved:")
    print(f"  Overall: {overall_file}")
    print(f"  Per-class: {class_file}")
    
    return overall_results, class_df

def convert_weak_labels_to_standard_format(noisy_data):
    """
    Convert weakly-supervised or noisy NER annotations to standardized evaluation format.
    
    This function cleans and standardizes NER data from noisy labeling processes by:
    - Removing duplicate entity spans within each text
    - Filtering out extraneous fields (e.g., row_id, confidence scores)
    - Normalizing the data structure to match evaluation pipeline requirements
    
    The function is designed to handle common issues in weakly-supervised NER data:
    - Multiple identical annotations for the same entity span
    - Inconsistent field naming across different annotation sources
    - Extra metadata fields that interfere with evaluation
    """
    standardized = []
    
    for item in noisy_data:
        # Remove duplicate spans
        seen_spans = set()
        unique_spans = []
        
        for span in item['spans']:
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
    Evaluate noisy NER predictions with the same API as evaluate_model_performance.
    
    Args:
        noisy_predictions: List of noisy prediction dictionaries 
        ground_truth_path (str): Path to ground truth JSON file
        output_dir (str): Directory to save CSV results
        dataset_name (str, optional): Name of dataset for file naming. Defaults to "noisy_system"
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (overall_results, class_df) - same format as evaluate_model_performance
    """
    print(f"Evaluating noisy predictions on {dataset_name} set...")
    
    # Load ground truth
    with open(ground_truth_path, 'r') as f:
        ground_truth_data = json.load(f)
    
    # Convert both to standard format and remove duplicates
    ground_truth_clean = convert_noisy_to_standard_format(ground_truth_data)
    noisy_predictions_clean = convert_noisy_to_standard_format(noisy_predictions)
    
    print(f"Loaded {len(ground_truth_clean)} ground truth examples")
    print(f"Loaded {len(noisy_predictions_clean)} predictions")
    
    # Convert to seqeval format
    def to_seqeval_format(data):
        """Convert span format to seqeval entity format"""
        entity_sequences = []
        for item in data:
            # Convert spans to entity labels: "start-end-label"
            entities = [f"{span['start']}-{span['end']}-{span['label']}" 
                       for span in item['spans']]
            entity_sequences.append(entities)
        return entity_sequences
    
    y_true = to_seqeval_format(ground_truth_clean)
    y_pred = to_seqeval_format(noisy_predictions_clean)
    
    # Use shared evaluation core
    return _evaluate_predictions_core(
        y_true=y_true,
        y_pred=y_pred, 
        system_identifier="noisy_ner_system",
        dataset_name=dataset_name,
        num_examples=len(y_true),
        output_dir=output_dir
    )

def evaluate_model_performance(model_path, data_path, output_dir, dataset_name="test", max_length=128):
    """
    Evaluate a trained NER model and save performance metrics to CSV files.

    Args:
        model_path (str): Path to the trained model directory.
        data_path (str): Path to the evaluation data JSON file.
        output_dir (str): Directory to save the CSV results.
        dataset_name (str, optional): Name of the dataset for file naming. Defaults to "test".
        max_length (int, optional): Maximum sequence length for tokenization. Defaults to 128.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - overall_results: DataFrame with overall metrics.
            - class_df: DataFrame with per-class metrics.
    """
    print(f"Evaluating model on {dataset_name} set...")
    
    # Load model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    if torch.cuda.is_available():
        model = model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    id2label = model.config.id2label
    
    # Load data
    label_list = list(id2label.values())
    processor = NERDataProcessor(label_list, tokenizer.name_or_path)
    processor.tokenizer = tokenizer
    
    val_data = processor.load_json_data(data_path)
    val_dataset = processor.create_dataset(val_data, max_length)
    
    print(f"Loaded {len(val_dataset)} examples")
    
    # Get predictions
    print("Getting predictions...")
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for example in val_dataset:
            input_ids = torch.tensor([example['input_ids']]).to(device)
            attention_mask = torch.tensor([example['attention_mask']]).to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
            
            true_labels = []
            pred_labels = []
            
            for pred_id, true_id in zip(predictions, example['labels']):
                if true_id != -100:
                    true_labels.append(id2label[true_id])
                    pred_labels.append(id2label[pred_id])
            
            y_true.append(true_labels)
            y_pred.append(pred_labels)
    
    # Use shared evaluation core instead of duplicated logic
    return _evaluate_predictions_core(
        y_true=y_true,
        y_pred=y_pred,
        system_identifier=model_path,  # This matches the original 'model_path' field
        dataset_name=dataset_name,
        num_examples=len(val_dataset),
        output_dir=output_dir
    )