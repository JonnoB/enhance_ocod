import pandas as pd
import torch
from transformers import pipeline
from typing import Dict, List, Optional
from tqdm import tqdm

def parse_addresses_pipeline(
    df: pd.DataFrame,
    model_path: str,
    target_column: str = "address",
    batch_size: int = 512,
    device: Optional[int] = None,
    show_progress: bool = True
) -> Dict:
    """
    Parse addresses using HuggingFace pipeline - simple and effective.
    """
    
    # Auto-detect device
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    
    # Create pipeline
    print("Loading model...")
    nlp = pipeline(
        "token-classification",
        model=model_path,
        device=device,
        batch_size=batch_size,
        aggregation_strategy="simple",  # Handles B-I-O tags automatically
        #return_all_scores=False
    )
    
    # Prepare data
    addresses = df[target_column].fillna("").astype(str).tolist()
    datapoint_ids = df.get("datapoint_id", df.index).tolist()
    
    print(f"Processing {len(addresses)} addresses with batch_size={batch_size}")
    
    # Process all addresses - pipeline handles batching internally
    all_predictions = nlp(addresses)
    
    # Convert to your preferred format
    results = []
    successful_parses = 0
    
    iterator = zip(addresses, all_predictions, datapoint_ids)
    if show_progress:
        iterator = tqdm(iterator, total=len(addresses), desc="Converting results")
    
    for i, (address, predictions, datapoint_id) in enumerate(iterator):
        # Convert HF format to your format
        entities = []
        for pred in predictions:
            entities.append({
                "type": pred["entity_group"],
                "text": pred["word"],
                "start": pred["start"],
                "end": pred["end"],
                "confidence": pred["score"]
            })
        
        # Group by type
        parsed_components = {}
        for entity in entities:
            entity_type = entity['type']
            if entity_type not in parsed_components:
                parsed_components[entity_type] = []
            parsed_components[entity_type].append(entity['text'])
        
        result = {
            "row_index": i,
            "datapoint_id": datapoint_id,
            "original_address": address,
            "entities": entities,
            "parsed_components": parsed_components
        }
        
        if entities:
            successful_parses += 1
        
        results.append(result)
    
    return {
        "summary": {
            "total_addresses": len(addresses),
            "successful_parses": successful_parses,
            "failed_parses": len(addresses) - successful_parses,
            "success_rate": successful_parses / len(addresses) if addresses else 0,
            "batch_size_used": batch_size
        },
        "results": results
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