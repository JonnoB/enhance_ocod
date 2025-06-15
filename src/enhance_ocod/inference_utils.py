import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import Dataset
from typing import Dict, List, Optional
from tqdm import tqdm

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from datasets import Dataset
import torch


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
        #print(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Get label mappings
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        #print(f"Model loaded on {self.device}. Labels: {list(self.label2id.keys())}")
    
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

def parse_addresses_batch(
    df: pd.DataFrame,
    model_path: str,
    target_column: str = "address", 
    index_column: Optional[str] = None,
    batch_size: int = 256,
    use_fp16: bool = True,
    show_progress: bool = True
) -> Dict:
    """
    Production-ready batch address parsing with mixed precision optimization.
    
    Args:
        df: DataFrame containing addresses
        model_path: Path to the trained model
        target_column: Column name containing addresses
        index_column: Column to use as index (optional)
        batch_size: Batch size for processing
        use_fp16: Enable mixed precision for speed
        show_progress: Show progress bar
    
    Returns:
        Dict with summary and results
    """
    # Initialize multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    parser = AddressParserInference(model_path)
    
    # Setup indexing
    if index_column is None and "datapoint_id" in df.columns:
        index_column = "datapoint_id"
    
    # Prepare dataset
    dataset_dict = {
        "address": df[target_column].fillna("").astype(str).tolist(),
        "row_index": df[index_column].tolist() if index_column else df.index.tolist()
    }
    
    if "datapoint_id" in df.columns:
        dataset_dict["datapoint_id"] = df["datapoint_id"].tolist()
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Tokenization function
    def tokenize_function(examples):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        tokenized = tokenizer(
            examples["address"],
            padding=False,
            truncation=True,
            max_length=128,
            return_offsets_mapping=True
        )
        
        tokenized["original_address"] = examples["address"]
        tokenized["row_index"] = examples["row_index"]
        
        if "datapoint_id" in examples:
            tokenized["datapoint_id"] = examples["datapoint_id"]
        
        return tokenized
    
    # Apply tokenization
    try:
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            num_proc=4,
            remove_columns=[]
        )
    except Exception:
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            remove_columns=[]
        )
    
    # Entity extraction
    def extract_entities(address, tokens, predicted_labels, offset_mapping):
        entities = []
        current_entity = None
        
        for token, label, offset in zip(tokens, predicted_labels, offset_mapping):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
                
            if hasattr(offset, 'tolist'):
                start_pos, end_pos = offset.tolist()
            else:
                start_pos, end_pos = offset
            
            if start_pos == end_pos:
                continue
                
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]
                current_entity = {
                    'type': entity_type,
                    'start': start_pos,
                    'end': end_pos,
                    'text': address[start_pos:end_pos]
                }
                
            elif label.startswith('I-') and current_entity:
                entity_type = label[2:]
                if entity_type == current_entity['type']:
                    current_entity['end'] = end_pos
                    current_entity['text'] = address[current_entity['start']:end_pos]
                else:
                    entities.append(current_entity)
                    current_entity = {
                        'type': entity_type,
                        'start': start_pos,
                        'end': end_pos,
                        'text': address[start_pos:end_pos]
                    }
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
            
        return entities
    
    # Collate function
    def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        offset_mapping = [item["offset_mapping"] for item in batch]
        row_indices = [item["row_index"] for item in batch]
        addresses = [item["original_address"] for item in batch]
        
        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_attention_mask = []
        
        for ids, mask in zip(input_ids, attention_mask):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [parser.tokenizer.pad_token_id] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)
        
        return {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(padded_attention_mask),
            "offset_mapping": offset_mapping,
            "addresses": addresses,
            "row_indices": row_indices,
            "datapoint_ids": [item.get("datapoint_id") for item in batch] if "datapoint_id" in batch[0] else None
        }
    
    # Create DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        shuffle=False
    )
    
    # Process batches
    all_results = []
    successful_parses = 0
    
    iterator = tqdm(dataloader, desc="Processing addresses") if show_progress else dataloader
    
    for batch in iterator:
        # Move to GPU
        input_ids = batch["input_ids"].to(parser.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(parser.device, non_blocking=True)
        
        # Inference
        with torch.no_grad():
            if use_fp16:
                with torch.amp.autocast('cuda'):
                    outputs = parser.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = parser.model(input_ids=input_ids, attention_mask=attention_mask)
            
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_token_class_ids = predictions.argmax(dim=-1)
        
        # Move to CPU
        predicted_token_class_ids_cpu = predicted_token_class_ids.cpu()
        input_ids_cpu = input_ids.cpu()
        torch.cuda.synchronize()
        
        # Extract entities
        for j in range(len(batch["addresses"])):
            address = batch["addresses"][j]
            tokens = parser.tokenizer.convert_ids_to_tokens(input_ids_cpu[j])
            predicted_labels = [parser.id2label[pred.item()] for pred in predicted_token_class_ids_cpu[j]]
            offset_mapping = batch["offset_mapping"][j]
            
            try:
                entities = extract_entities(address, tokens, predicted_labels, offset_mapping)
                if len(entities) > 0:
                    successful_parses += 1
            except Exception:
                entities = []
            
            result = {
                "row_index": batch["row_indices"][j],
                "original_address": address,
                "entities": entities,
                "parsed_components": parser._group_entities_by_type(entities) if hasattr(parser, '_group_entities_by_type') else {}
            }
            
            if batch["datapoint_ids"] and batch["datapoint_ids"][j]:
                result["datapoint_id"] = batch["datapoint_ids"][j]
            
            all_results.append(result)
        
        # Cleanup
        del outputs, predictions, predicted_token_class_ids
        torch.cuda.empty_cache()
    
    return {
        "summary": {
            "total_addresses": len(dataset),
            "successful_parses": successful_parses,
            "failed_parses": len(dataset) - successful_parses,
            "success_rate": successful_parses / len(dataset) if len(dataset) > 0 else 0
        },
        "results": all_results
    }


def convert_to_entity_dataframe(modernbert_results: Dict, batch_size: int = 50000) -> pd.DataFrame:
    """
    Converts ModernBERT parsing results to a structured entity DataFrame
    Optimized for large datasets (hundreds of thousands of entities)
    
    Args:
        modernbert_results: Dictionary output from parse_addresses_from_csv
        
    Returns:
        pandas DataFrame with structured entity data
    """
    
    # Pre-calculate total entities for memory allocation
    total_entities = sum(len(result["entities"]) for result in modernbert_results["results"])
    
    if total_entities == 0:
        return pd.DataFrame(columns=['datapoint_id', 'label', 'start', 'end', 'text', 'label_text', 'label_id_count'])
    
    print(f'Processing {total_entities:,} entities...')
    
    # Pre-allocate arrays for better performance
    datapoint_ids = []
    labels = []
    starts = []
    ends = []
    texts = []
    label_texts = []
    
    # Batch process to avoid memory issues
    processed = 0
    
    for result in modernbert_results["results"]:
        datapoint_id = result["row_index"]
        original_address = result["original_address"]
        entities = result["entities"]
        
        # Batch append for efficiency
        entity_count = len(entities)
        if entity_count > 0:
            datapoint_ids.extend([datapoint_id] * entity_count)
            labels.extend([entity['type'] for entity in entities])
            starts.extend([entity['start'] for entity in entities])
            ends.extend([entity['end'] for entity in entities])
            texts.extend([original_address] * entity_count)
            label_texts.extend([entity['text'] for entity in entities])
            
            processed += entity_count
            if processed % batch_size == 0:
                print(f'Processed {processed:,}/{total_entities:,} entities')
    
    # Create DataFrame directly from arrays (much faster than list of dicts)
    all_entities = pd.DataFrame({
        'datapoint_id': datapoint_ids,
        'label': labels,
        'start': starts,
        'end': ends,
        'text': texts,
        'label_text': label_texts
    })
    
    print('Computing label counts...')
    # Optimize the groupby operation
    all_entities['label_id_count'] = all_entities.groupby(['datapoint_id', 'label'], sort=False).cumcount()
    
    print('Named Entity Recognition labelling complete')
    print(f'Total entities extracted: {len(all_entities):,}')
    
    return all_entities