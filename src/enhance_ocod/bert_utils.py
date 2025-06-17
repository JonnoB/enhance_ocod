
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
    def __init__(self, label_list: List[str], tokenizer_name: str = "answerdotai/ModernBERT-base"):
        self.label_list = label_list
        self.label2id = {label: i for i, label in enumerate(label_list)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def load_json_data(self, file_path: str) -> List[Dict]:
        """Load NER data from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def align_labels_with_tokens(self, text: str, spans: List[Dict], tokenized_inputs) -> List[int]:
        """
        Align character-level spans with token-level labels.
        Returns a list of label IDs for each token.
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
        previous_word_idx = None
        current_entity = None
        
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
            except:
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
                current_entity = None
            else:
                # Token is inside an entity
                entity_label = entity_span['label']
                
                # Determine if this is the beginning of an entity or continuation
                if (current_entity != entity_span or 
                    word_idx != previous_word_idx or 
                    token_start == entity_span['start']):
                    # This is the beginning of an entity
                    labels[token_idx] = self.label2id.get(f'B-{entity_label}', self.label2id['O'])
                    current_entity = entity_span
                else:
                    # This is a continuation of the current entity
                    labels[token_idx] = self.label2id.get(f'I-{entity_label}', self.label2id['O'])
            
            previous_word_idx = word_idx
        
        return labels
    
    def process_examples(self, examples: List[Dict], max_length: int = 128) -> Dict[str, List]:
        """
        Process a batch of examples into model inputs.
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
        Create a HuggingFace Dataset from the data.
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
        Compute entity-level metrics using seqeval.
        Uses this processor's id2label mapping.
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
    Create BIO label list from entity types.
    """
    label_list = ['O']  # Outside any entity
    for entity_type in entity_types:
        label_list.append(f'B-{entity_type}')  # Beginning
        label_list.append(f'I-{entity_type}')  # Inside
    return label_list


def evaluate_model_performance(model_path, data_path, output_dir, dataset_name="test", max_length=128):
    """
    Evaluate model and save performance metrics to CSV.
    
    Args:
        model_path: Path to the trained model
        data_path: Path to the evaluation data JSON file
        output_dir: Directory to save the CSV results
        dataset_name: Name of the dataset (for file naming)
        max_length: Maximum sequence length for tokenization
    """
    print(f"Evaluating model on {dataset_name} set...")
    
    # Load model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(model_path)
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
    
    # Calculate metrics
    overall_f1 = f1_score(y_true, y_pred)
    overall_precision = precision_score(y_true, y_pred)
    overall_recall = recall_score(y_true, y_pred)
    
    # Get detailed classification report
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    
    # Print results
    print(f"\n{dataset_name.upper()} SET RESULTS:")
    print(f"Overall F1: {overall_f1:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")  
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"\nPer-class results:")
    print(classification_report(y_true, y_pred))
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Overall performance CSV
    overall_results = pd.DataFrame([{
        'model_path': model_path,
        'dataset': dataset_name,
        'num_examples': len(val_dataset),
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': overall_f1,
        'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }])
    
    overall_file = output_path / f"{dataset_name}_overall_performance.csv"
    overall_results.to_csv(overall_file, index=False)
    
    # 2. Class-wise performance CSV
    class_results = []
    for class_name, metrics in report_dict.items():
        if isinstance(metrics, dict):  # Skip accuracy (which is just a float)
            class_results.append({
                'model_path': model_path,
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
    
    print(f"\nResults saved:")
    print(f"  Overall: {overall_file}")
    print(f"  Per-class: {class_file}")
    
    return overall_results, class_df