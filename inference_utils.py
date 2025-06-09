import pandas as pd
import torch
import zipfile
import json
from pathlib import Path
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import Dataset
from typing import Dict, List, Optional, Union
import numpy as np

class AddressParserInference:
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the inference class with trained model
        
        Args:
            model_path: Path to the trained model directory
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.label2id = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and tokenizer"""
        print(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Get label mappings
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        print(f"Model loaded on {self.device}. Labels: {list(self.label2id.keys())}")
    
    def tokenize_batch(self, examples):
        """Tokenize batch of examples for dataset mapping"""
        return self.tokenizer(
            examples['address'],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_offsets_mapping=True
        )
    
    def predict_batch(self, batch_size: int = 32):
        """
        Predict function for batched processing
        
        Args:
            batch_size: Batch size for inference
            
        Returns:
            Function to be used with dataset.map()
        """
        def _predict_batch(batch):
            addresses = batch['address']
            indices = batch['index']
            batch_size_actual = len(addresses)
            
            # Tokenize
            inputs = self.tokenizer(
                addresses,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
                return_offsets_mapping=True
            )
            
            # Move to device
            model_inputs = {k: v.to(self.device) for k, v in inputs.items() if k != 'offset_mapping'}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**model_inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_token_class_ids = predictions.argmax(dim=-1)
            
            # Process results for each item in batch
            batch_results = []
            for i in range(batch_size_actual):
                address = addresses[i]
                index = indices[i]
                
                # Get tokens and predictions for this item
                input_ids = inputs["input_ids"][i]
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                predicted_labels = [self.id2label[pred.item()] for pred in predicted_token_class_ids[i]]
                offset_mapping = inputs["offset_mapping"][i]
                
                # Extract entities
                entities = self._extract_entities(address, tokens, predicted_labels, offset_mapping)
                
                batch_results.append({
                    "row_index": index,
                    "original_address": address,
                    "entities": entities,
                    "parsed_components": self._group_entities_by_type(entities)
                })
            
            return {
                "results": batch_results
            }
        
        return _predict_batch
    
    def _extract_entities(self, original_text: str, tokens: List[str], labels: List[str], offset_mapping) -> List[Dict]:
        """Extract entities from predictions"""
        entities = []
        current_entity = None
        
        for i, (token, label, offset) in enumerate(zip(tokens, labels, offset_mapping)):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
                
            start_pos, end_pos = offset.tolist()
            
            # Skip if offset is [0, 0] (special tokens)
            if start_pos == 0 and end_pos == 0 and i > 0:
                continue
                
            if label.startswith('B-'):
                # Start of new entity
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]  # Remove 'B-' prefix
                current_entity = {
                    "type": entity_type,
                    "text": original_text[start_pos:end_pos],
                    "start": start_pos,
                    "end": end_pos
                }
            elif label.startswith('I-') and current_entity:
                # Continuation of current entity
                entity_type = label[2:]  # Remove 'I-' prefix
                if entity_type == current_entity["type"]:
                    current_entity["text"] = original_text[current_entity["start"]:end_pos]
                    current_entity["end"] = end_pos
            else:
                # O label or end of entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Don't forget the last entity
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def _group_entities_by_type(self, entities: List[Dict]) -> Dict:
        """Group entities by their type"""
        grouped = {}
        for entity in entities:
            entity_type = entity["type"]
            if entity_type not in grouped:
                grouped[entity_type] = []
            grouped[entity_type].append(entity["text"])
        
        # For single values, return string instead of list
        for key, value in grouped.items():
            if len(value) == 1:
                grouped[key] = value[0]
        
        return grouped


def load_csv_from_zip(zip_path: str, csv_filename: Optional[str] = None) -> pd.DataFrame:
    """Load CSV from zip file"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError("No CSV files found in zip")
        
        if csv_filename:
            if csv_filename not in csv_files:
                raise ValueError(f"CSV file '{csv_filename}' not found in zip. Available: {csv_files}")
            target_file = csv_filename
        else:
            target_file = csv_files[0]
        
        with zip_ref.open(target_file) as csv_file:
            df = pd.read_csv(csv_file)
    
    return df



def parse_addresses_from_csv(
    model_path: str,
    csv_path: str,
    target_column: str = "address",
    index_column: Optional[str] = None,
    csv_filename: Optional[str] = None,
    batch_size: int = 64
) -> Dict:
    """
    Ultra-fast version that processes everything in one go (for datasets that fit in memory)
    
    Args:
        model_path: Path to trained model
        csv_path: Path to CSV file or zip file containing CSV
        target_column: Column name containing addresses
        index_column: Column to use as index (if None, uses pandas index)
        csv_filename: If csv_path is zip, specific CSV file to use
        batch_size: Batch size for inference
        
    Returns:
        Dictionary with parsing results
    """
    # Initialize inference
    parser = AddressParserInference(model_path)
    
    # Load data
    if csv_path.endswith('.zip'):
        df = load_csv_from_zip(csv_path, csv_filename)
        print(f"Loaded CSV from zip: {csv_path}")
    else:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV: {csv_path}")
    
    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found. Available columns: {list(df.columns)}")
    
    # Handle index column
    if index_column:
        if index_column not in df.columns:
            raise ValueError(f"Index column '{index_column}' not found. Available columns: {list(df.columns)}")
        indices = df[index_column].tolist()
    else:
        indices = df.index.tolist()
    
    # Prepare data
    addresses = df[target_column].fillna("").astype(str).tolist()
    
    print(f"Processing {len(addresses)} addresses in batches of {batch_size}")
    
    all_results = []
    
    # Process in batches
    for i in range(0, len(addresses), batch_size):
        batch_addresses = addresses[i:i+batch_size]
        batch_indices = indices[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(addresses) + batch_size - 1)//batch_size}")
        
        # Tokenize batch
        inputs = parser.tokenizer(
            batch_addresses,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        
        # Move to device (excluding offset_mapping)
        model_inputs = {k: v.to(parser.device) for k, v in inputs.items() if k != 'offset_mapping'}
        
        # Predict
        with torch.no_grad():
            outputs = parser.model(**model_inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_token_class_ids = predictions.argmax(dim=-1)
        
        # Process each item in batch
        for j in range(len(batch_addresses)):
            address = batch_addresses[j]
            index = batch_indices[j]
            
            # Get tokens and predictions
            input_ids = inputs["input_ids"][j]
            tokens = parser.tokenizer.convert_ids_to_tokens(input_ids)
            predicted_labels = [parser.id2label[pred.item()] for pred in predicted_token_class_ids[j]]
            offset_mapping = inputs["offset_mapping"][j]
            
            # Extract entities
            entities = parser._extract_entities(address, tokens, predicted_labels, offset_mapping)
            
            all_results.append({
                "row_index": index,
                "original_address": address,
                "entities": entities,
                "parsed_components": parser._group_entities_by_type(entities)
            })
    
    # Calculate statistics
    successful_parses = len([r for r in all_results if len(r["entities"]) > 0])
    
    return {
        "summary": {
            "total_addresses": len(addresses),
            "successful_parses": successful_parses,
            "failed_parses": len(addresses) - successful_parses,
            "success_rate": successful_parses / len(addresses) if len(addresses) > 0 else 0
        },
        "results": all_results
    }





def example_mega_fast_csv_parsing():
    """Example of ultra-fast CSV parsing"""
    model_path = "models/address_parser2/checkpoint-750"
    csv_path = "data/ocod_history/OCOD_FULL_2015_10.zip"
    
    import time
    start_time = time.time()
    
    results = parse_addresses_from_csv(
        model_path=model_path,
        csv_path=csv_path,
        target_column="Property_Address",
        batch_size=2048
    )
    
    end_time = time.time()
    
    print(f"Mega-fast CSV Parsing completed in {end_time - start_time:.2f} seconds")
    print("Results summary:")
    print(json.dumps(results["summary"], indent=2))
    
    return results

