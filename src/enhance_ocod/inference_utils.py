import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import Dataset
from typing import Dict, List, Optional
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

class AddressParserInference:
    def __init__(self, model_path: str, device: Optional[str] = None, max_length: int = 128):
        """
        Initialize the inference class with trained model
        
        Args:
            model_path: Path to the trained model directory
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length (should match training)
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.label2id = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
    
    def _extract_entities_with_word_alignment(self, original_text: str, tokens: List[str], 
                                            predicted_labels: List[str], offset_mapping) -> List[Dict]:
        """
        Extract entities with proper subword token handling - matches training logic
        """
        entities = []
        current_entity = None
        
        # Track word boundaries for proper entity extraction
        previous_word_start = None
        
        for i, (token, label, offset) in enumerate(zip(tokens, predicted_labels, offset_mapping)):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
                
            if hasattr(offset, 'tolist'):
                start_pos, end_pos = offset.tolist()
            else:
                start_pos, end_pos = offset
            
            # Skip invalid offsets
            if start_pos == end_pos or start_pos >= len(original_text) or end_pos > len(original_text):
                continue
                
            # Determine if this token starts a new word (not a subword continuation)
            is_word_start = (previous_word_start != start_pos) if previous_word_start is not None else True
            
            if label.startswith('B-'):
                # Beginning of new entity - always start new entity
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]
                current_entity = {
                    'type': entity_type,
                    'start': start_pos,
                    'end': end_pos,
                    'text': original_text[start_pos:end_pos]
                }
                
            elif label.startswith('I-') and current_entity:
                # Inside entity - extend current entity
                entity_type = label[2:]
                if entity_type == current_entity['type']:
                    # Extend entity to include this token
                    current_entity['end'] = end_pos
                    current_entity['text'] = original_text[current_entity['start']:end_pos]
                else:
                    # Different entity type - treat as B- (beginning of new entity)
                    entities.append(current_entity)
                    current_entity = {
                        'type': entity_type,
                        'start': start_pos,
                        'end': end_pos,
                        'text': original_text[start_pos:end_pos]
                    }
            else:
                # 'O' label - outside entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            
            # Track word boundaries
            if token.startswith('##'):  # Subword token
                pass  # Don't update word start
            else:
                previous_word_start = start_pos
        
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
    max_length: int = 128,  # Should match training
    use_fp16: bool = True,
    show_progress: bool = True,
    num_workers: int = 4
) -> Dict:
    """
    Production-ready batch address parsing with consistent tokenization and performance optimization.
    
    Args:
        df: DataFrame containing addresses
        model_path: Path to the trained model
        target_column: Column name containing addresses
        index_column: Column to use as index (optional)
        batch_size: Batch size for processing
        max_length: Maximum sequence length (should match training)
        use_fp16: Enable mixed precision for speed
        show_progress: Show progress bar
        num_workers: Number of workers for data loading
    
    Returns:
        Dict with summary and results
    """
    # Initialize multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    parser = AddressParserInference(model_path, max_length=max_length)
    
    # Setup indexing
    if index_column is None and "datapoint_id" in df.columns:
        index_column = "datapoint_id"
    
    # Prepare dataset - FIXED: Use consistent max_length
    dataset_dict = {
        "address": df[target_column].fillna("").astype(str).tolist(),
        "row_index": df[index_column].tolist() if index_column else df.index.tolist()
    }
    
    if "datapoint_id" in df.columns:
        dataset_dict["datapoint_id"] = df["datapoint_id"].tolist()
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # FIXED: Tokenization function that matches training exactly
    def tokenize_function(examples):
        tokenized = parser.tokenizer(
            examples["address"],
            padding=False,  # We'll pad in collate_fn
            truncation=True,
            max_length=max_length,  # Use consistent max_length
            return_offsets_mapping=True
        )
        
        tokenized["original_address"] = examples["address"]
        tokenized["row_index"] = examples["row_index"]
        
        if "datapoint_id" in examples:
            tokenized["datapoint_id"] = examples["datapoint_id"]
        
        return tokenized
    
    # Apply tokenization with multiprocessing
    try:
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            num_proc=num_workers,
            remove_columns=[]
        )
    except Exception:
        # Fallback without multiprocessing
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            remove_columns=[]
        )
    
    # FIXED: Collate function with proper padding to max_length
    def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        offset_mapping = [item["offset_mapping"] for item in batch]
        row_indices = [item["row_index"] for item in batch]
        addresses = [item["original_address"] for item in batch]
        
        # Pad to max_length (consistent with training)
        padded_input_ids = []
        padded_attention_mask = []
        padded_offset_mapping = []
        
        for ids, mask, offsets in zip(input_ids, attention_mask, offset_mapping):
            pad_len = max_length - len(ids)
            if pad_len > 0:
                padded_input_ids.append(ids + [parser.tokenizer.pad_token_id] * pad_len)
                padded_attention_mask.append(mask + [0] * pad_len)
                padded_offset_mapping.append(offsets + [(0, 0)] * pad_len)
            else:
                padded_input_ids.append(ids[:max_length])
                padded_attention_mask.append(mask[:max_length])
                padded_offset_mapping.append(offsets[:max_length])
        
        return {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(padded_attention_mask),
            "offset_mapping": padded_offset_mapping,  # Keep as list for processing
            "addresses": addresses,
            "row_indices": row_indices,
            "datapoint_ids": [item.get("datapoint_id") for item in batch] if "datapoint_id" in batch[0] else None
        }
    
    # Create DataLoader with performance optimizations
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 for GPU inference to avoid issues
        pin_memory=torch.cuda.is_available(),
        shuffle=False
    )
    
    # Process batches with performance optimizations
    all_results = []
    successful_parses = 0
    
    iterator = tqdm(dataloader, desc="Processing addresses") if show_progress else dataloader
    
    for batch in iterator:
        # Move to GPU with non-blocking transfer
        input_ids = batch["input_ids"].to(parser.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(parser.device, non_blocking=True)
        
        # MAINTAINED: FP16 inference for speed
        with torch.no_grad():
            if use_fp16 and parser.device != 'cpu':
                with torch.amp.autocast('cuda'):
                    outputs = parser.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = parser.model(input_ids=input_ids, attention_mask=attention_mask)
            
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_token_class_ids = predictions.argmax(dim=-1)
        
        # Move to CPU for processing
        predicted_token_class_ids_cpu = predicted_token_class_ids.cpu()
        input_ids_cpu = input_ids.cpu()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # FIXED: Extract entities with proper tokenization handling
        for j in range(len(batch["addresses"])):
            address = batch["addresses"][j]
            tokens = parser.tokenizer.convert_ids_to_tokens(input_ids_cpu[j])
            predicted_labels = [parser.id2label[pred.item()] for pred in predicted_token_class_ids_cpu[j]]
            offset_mapping = batch["offset_mapping"][j]
            
            try:
                entities = parser._extract_entities_with_word_alignment(
                    address, tokens, predicted_labels, offset_mapping
                )
                if len(entities) > 0:
                    successful_parses += 1
            except Exception as e:
                entities = []
                print(f"Error processing address: {address[:50]}... Error: {e}")
            
            result = {
                "row_index": batch["row_indices"][j],
                "original_address": address,
                "entities": entities,
                "parsed_components": parser._group_entities_by_type(entities)
            }
            
            if batch["datapoint_ids"] and batch["datapoint_ids"][j]:
                result["datapoint_id"] = batch["datapoint_ids"][j]
            
            all_results.append(result)
        
        # MAINTAINED: Memory cleanup
        del outputs, predictions, predicted_token_class_ids
        if torch.cuda.is_available():
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
        datapoint_id = result.get("datapoint_id") or result["row_index"]
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