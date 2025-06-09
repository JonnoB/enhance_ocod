
import json
from typing import List, Dict
from transformers import AutoTokenizer
from datasets import Dataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

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
        
        # Create character to label mapping
        char_labels = ['O'] * len(text)
        for span in spans:
            start = span['start']
            end = span['end']
            label = span['label']
            # Mark the first character as B- (beginning) and rest as I- (inside)
            for i in range(start, end):
                if i == start:
                    char_labels[i] = f"B-{label}"
                else:
                    char_labels[i] = f"I-{label}"
        
        # Map tokens to labels
        previous_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:  # Special tokens
                labels[idx] = -100
            elif word_idx != previous_word_idx:  # First token of a word
                # Get the character position for this token
                token_start = tokenized_inputs.token_to_chars(idx).start
                labels[idx] = self.label2id.get(char_labels[token_start], self.label2id['O'])
            else:  # Other tokens of the word
                # For subword tokens, use -100 to ignore in loss calculation
                # Alternatively, you could propagate the label
                labels[idx] = -100
                
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

def compute_metrics(eval_pred):
    """
    Compute NER metrics (precision, recall, F1).
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for l in label if l != -100]
        for label in labels
    ]

    
    # Flatten the lists
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    flat_predictions = [item for sublist in true_predictions for item in sublist]
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_true_labels, 
        flat_predictions, 
        average='weighted',
        zero_division=0
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
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