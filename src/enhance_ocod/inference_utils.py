import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from typing import Dict, List, Optional
from tqdm import tqdm
from seqeval.metrics.sequence_labeling import get_entities
import numpy as np

class AddressParserInference:
    """
    Batch-optimized address parser using ModernBERT
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None, use_fp16: bool = True, 
                 max_length: int = 512, stride: int = 50):
        """
        Initialize using direct model inference.
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_fp16 = use_fp16 and self.device == 'cuda'
        self.max_length = max_length
        self.stride = stride
        
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.id2label = self.model.config.id2label
        
        # Setup device and precision
        self.model.to(self.device)
        if self.use_fp16:
            self.model.half()
        self.model.eval()
        
        print(f"AddressParserInference initialized: {self.device}, FP16={self.use_fp16}")
    
    def _predict_batch_tokens(self, texts: List[str]) -> List[tuple]:
        """
        Get token predictions and offset mappings for a batch of texts.
        This is the key optimization - process entire batch in one forward pass.
        """
        # Tokenize entire batch
        batch_encoding = self.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            padding=True,  # Pad to max length in batch
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        input_ids = batch_encoding['input_ids'].to(self.device)
        attention_mask = batch_encoding['attention_mask'].to(self.device)
        offset_mappings = batch_encoding['offset_mapping'].cpu().numpy()
        
        # Single forward pass for entire batch
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        
        # Process results for each item in batch
        batch_results = []
        for i in range(len(texts)):
            pred_labels = []
            valid_offsets = []
            
            for j in range(len(predictions[i])):
                # Skip special tokens and padding
                if attention_mask[i][j] == 0:  # Skip padding
                    break
                if (offset_mappings[i][j][0] == 0 and offset_mappings[i][j][1] == 0):  # Skip special tokens
                    continue
                
                pred_labels.append(self.id2label[predictions[i][j]])
                valid_offsets.append(offset_mappings[i][j])
            
            batch_results.append((pred_labels, valid_offsets))
        
        return batch_results

    def predict_single_address(self, address: str, row_index: Optional[int] = None) -> Dict:
        """
        Predict entities for a single address - now uses batch method with batch_size=1
        """
        batch_results = self.predict_batch([address], batch_size=1, show_progress=False)
        result = batch_results[0]
        if row_index is not None:
            result["row_index"] = row_index
        return result

    def predict_batch(self, addresses: List[str], batch_size: int = 32, 
                     show_progress: bool = True) -> List[Dict]:
        """
        Predict entities for a batch of addresses using true batch inference.
        """
        results = []
        total = len(addresses)
        
        if show_progress:
            pbar = tqdm(total=total, desc="Processing addresses")
        
        for i in range(0, total, batch_size):
            batch_addresses = addresses[i:i + batch_size]
            batch_results = self._process_address_batch(batch_addresses, start_index=i)
            results.extend(batch_results)
            
            if show_progress:
                pbar.update(len(batch_addresses))
        
        if show_progress:
            pbar.close()
        
        return results
    
    def _process_address_batch(self, addresses: List[str], start_index: int = 0) -> List[Dict]:
        """
        Process a batch of addresses with true batch inference.
        """
        batch_results = []
        
        try:
            # Handle empty addresses
            non_empty_addresses = []
            non_empty_indices = []
            
            for i, address in enumerate(addresses):
                if not address or address.strip() == "":
                    batch_results.append({
                        "row_index": start_index + i,
                        "original_address": address,
                        "entities": [],
                        "parsed_components": {},
                        "error": "Empty address"
                    })
                else:
                    non_empty_addresses.append(address)
                    non_empty_indices.append(i)
            
            if not non_empty_addresses:
                return batch_results
            
            # Get predictions for all non-empty addresses in one batch
            batch_predictions = self._predict_batch_tokens(non_empty_addresses)
            
            # Process each address result
            for i, (pred_labels, offsets) in enumerate(batch_predictions):
                original_index = non_empty_indices[i]
                address = non_empty_addresses[i]
                
                # Use seqeval to extract entities
                entities_seqeval = get_entities(pred_labels)
                
                # Convert to our format with character positions
                entities = []
                for entity_type, start_idx, end_idx in entities_seqeval:
                    if start_idx < len(offsets) and end_idx < len(offsets):
                        char_start = offsets[start_idx][0]
                        char_end = offsets[end_idx][1]
                        entity_text = address[char_start:char_end].strip()
                        
                        entities.append({
                            "type": entity_type,
                            "text": entity_text,
                            "start": int(char_start),
                            "end": int(char_end),
                            "confidence": 1.0
                        })
                
                result = {
                    "row_index": start_index + original_index,
                    "original_address": address,
                    "entities": entities,
                    "parsed_components": self._group_entities_by_type(entities)
                }
                
                # Insert result at correct position
                while len(batch_results) <= original_index:
                    batch_results.append(None)
                batch_results[original_index] = result
            
            # Fill any remaining None slots (shouldn't happen but safety check)
            for i in range(len(batch_results)):
                if batch_results[i] is None:
                    batch_results[i] = {
                        "row_index": start_index + i,
                        "original_address": addresses[i] if i < len(addresses) else "",
                        "entities": [],
                        "parsed_components": {},
                        "error": "Processing error"
                    }
                    
        except Exception as e:
            import traceback
            print(f"Error processing batch starting at index {start_index}: {str(e)}")
            traceback.print_exc()
            
            # Return error results for entire batch
            batch_results = []
            for i, address in enumerate(addresses):
                batch_results.append({
                    "row_index": start_index + i,
                    "original_address": address,
                    "entities": [],
                    "parsed_components": {},
                    "error": str(e)
                })
        
        return batch_results

    def _group_entities_by_type(self, entities: List[Dict]) -> Dict[str, List[str]]:
        """
        Group entities by their type for easy access.
        """
        grouped = {}
        for entity in entities:
            entity_type = entity['type']
            if entity_type not in grouped:
                grouped[entity_type] = []
            grouped[entity_type].append(entity['text'])
        return grouped

    def get_config(self) -> Dict:
        """Return configuration for summary reporting."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "use_fp16": self.use_fp16,
            "max_length": self.max_length,
            "stride": self.stride
        }

# The parse_addresses_batch function remains the same
def parse_addresses_batch(
    df: pd.DataFrame,
    model_path: str,
    target_column: str = "address", 
    index_column: Optional[str] = None,
    batch_size: int = 32,
    use_fp16: bool = True,
    max_length: int = 512,
    stride: int = 50,
    show_progress: bool = True
) -> Dict:
    """
    Batch address parsing using the AddressParserInference class.
    """
    
    # Initialize parser
    parser = AddressParserInference(
        model_path=model_path,
        use_fp16=use_fp16,
        max_length=max_length,
        stride=stride
    )
    
    # Setup indexing
    if index_column is None and "datapoint_id" in df.columns:
        index_column = "datapoint_id"
    
    # Handle missing values and convert to string
    addresses = df[target_column].fillna("").astype(str).tolist()
    indices = df[index_column].tolist() if index_column else df.index.tolist()
    
    # Create datapoint_id mapping if needed
    datapoint_ids = df["datapoint_id"].tolist() if "datapoint_id" in df.columns else None
    
    # Process all addresses with true batch inference
    all_results = parser.predict_batch(addresses, batch_size=batch_size, show_progress=show_progress)
    
    # Update results with proper indexing and datapoint_ids
    successful_parses = 0
    errors = []
    
    for i, result in enumerate(all_results):
        # Update with proper index
        if index_column:
            result["original_index"] = indices[i]
        
        # Add datapoint_id if available
        if datapoint_ids is not None:
            result["datapoint_id"] = datapoint_ids[i]
        
        if "error" in result:
            errors.append((i, result["error"]))
        elif len(result["entities"]) > 0:
            successful_parses += 1
    
    # Print error summary if any
    if errors:
        print(f"\nEncountered {len(errors)} errors during processing:")
        for idx, error in errors[:5]:  # Show first 5 errors
            print(f"  Row {idx}: {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
    
    return {
        "summary": {
            "total_addresses": len(addresses),
            "successful_parses": successful_parses,
            "failed_parses": len(addresses) - successful_parses,
            "success_rate": successful_parses / len(addresses) if len(addresses) > 0 else 0,
            "batch_size_used": batch_size,
            "errors": len(errors)
        },
        "results": all_results
    }
def convert_to_entity_dataframe(results: Dict, batch_size: int = 50000) -> pd.DataFrame:
    """
    Convert parsing results to structured entity DataFrame.
    
    This function is unchanged - it works with the new entity format
    from the inference class.
    
    Creates a long-format DataFrame where each row is one entity,
    suitable for further analysis or database storage.
    """
    total_entities = sum(len(result["entities"]) for result in results["results"])
    
    if total_entities == 0:
        print("Warning: No entities found in results!")
        return pd.DataFrame(columns=['datapoint_id', 'label', 'start', 'end', 'text', 'label_text', 'label_id_count'])
    
    print(f'Processing {total_entities:,} entities...')
    
    # Pre-allocate arrays for efficiency
    datapoint_ids = []
    labels = []
    starts = []
    ends = []
    texts = []
    label_texts = []
    
    processed = 0
    
    for result in results["results"]:
        datapoint_id = result.get("datapoint_id", result.get("row_index", "unknown"))
        original_address = result["original_address"]
        entities = result["entities"]
        
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
    
    all_entities = pd.DataFrame({
        'datapoint_id': datapoint_ids,
        'label': labels,
        'start': starts,
        'end': ends,
        'text': texts,
        'label_text': label_texts
    })
    
    print('Computing label counts...')
    # Add counter for multiple entities of same type within same address
    all_entities['label_id_count'] = all_entities.groupby(['datapoint_id', 'label'], sort=False).cumcount()
    
    print('Named Entity Recognition labelling complete')
    print(f'Total entities extracted: {len(all_entities):,}')
    
    return all_entities