"""

"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support
from collections import defaultdict
import json
import torch
from pathlib import Path
from transformers import AutoModelForTokenClassification, AutoTokenizer
from enhance_ocod.training import NERDataProcessor

from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from seqeval.metrics.sequence_labeling import get_entities

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
model_path = str(SCRIPT_DIR / ".." / "models" / "address_parser_dev"/ "final_model")
val_data_path = str(SCRIPT_DIR / ".." / "data" / "training_data" / 'ner_ready' /"ground_truth_test_set_labels.json")
max_length = 128

def evaluate_ner_simple():
    """
    Simple NER evaluation using seqeval.
    Returns data needed for entity-level error analysis.
    """
    print("Loading model and data...")
    
    # Load model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    id2label = model.config.id2label
    
    # Load data using your existing processor
    label_list = list(id2label.values())
    processor = NERDataProcessor(label_list, tokenizer.name_or_path)
    processor.tokenizer = tokenizer
    
    val_data = processor.load_json_data(val_data_path)
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
    
    # Show seqeval results
    print("\n" + "="*50)
    print("SEQEVAL RESULTS")
    print("="*50)
    
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred))
    
    return y_true, y_pred, val_data, tokenizer

def analyze_entity_errors(y_true, y_pred, val_data, tokenizer):
    """
    Clean entity-level error analysis using seqeval.
    No more tokenization artifacts!
    """
    print("\n" + "="*60)
    print("ENTITY-LEVEL ERROR ANALYSIS")
    print("="*60)
    
    errors = []
    entity_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for i, (true_seq, pred_seq, data_item) in enumerate(zip(y_true, y_pred, val_data)):
        text = data_item['text']
        
        # Extract entities using seqeval - this handles BIO properly
        true_entities = get_entities(true_seq)  # Returns [(label, start, end), ...]
        pred_entities = get_entities(pred_seq)
        
        # Convert to sets for easy comparison
        true_set = set(true_entities)
        pred_set = set(pred_entities)
        
        # Count true positives, false positives, false negatives
        true_positives = true_set.intersection(pred_set)
        false_negatives = true_set - pred_set  # Missed entities
        false_positives = pred_set - true_set  # Incorrectly predicted entities
        
        # Update stats
        for entity_type, start, end in true_positives:
            entity_stats[entity_type]['tp'] += 1
            
        for entity_type, start, end in false_negatives:
            entity_stats[entity_type]['fn'] += 1
            
        for entity_type, start, end in false_positives:
            entity_stats[entity_type]['fp'] += 1
        
        # Record individual errors for review
        for entity_type, start, end in false_negatives:
            entity_text = get_entity_text_from_tokens(true_seq[start:end+1], text, tokenizer, start, end)
            errors.append({
                'example_id': i,
                'text': text[:100] + "..." if len(text) > 100 else text,
                'error_type': 'Missing Entity (False Negative)',
                'entity_type': entity_type,
                'entity_text': entity_text,
                'token_positions': f"{start}-{end}",
                'ground_truth': f"{entity_type}: {entity_text}",
                'model_prediction': "Nothing (missed)",
                'human_review_needed': 'Is this entity correctly labeled?',
                'model_correct': ''
            })
        
        for entity_type, start, end in false_positives:
            entity_text = get_entity_text_from_tokens(pred_seq[start:end+1], text, tokenizer, start, end)
            errors.append({
                'example_id': i,
                'text': text[:100] + "..." if len(text) > 100 else text,
                'error_type': 'Extra Entity (False Positive)',
                'entity_type': entity_type,
                'entity_text': entity_text,
                'token_positions': f"{start}-{end}",
                'ground_truth': "Nothing (not labeled)",
                'model_prediction': f"{entity_type}: {entity_text}",
                'human_review_needed': 'Should this be labeled as an entity?',
                'model_correct': ''
            })
    
    # Create summary statistics
    print(f"Total entity-level errors: {len(errors)}")
    print(f"Examples with errors: {len(set(error['example_id'] for error in errors))}")
    
    error_types = defaultdict(int)
    entity_type_errors = defaultdict(int)
    
    for error in errors:
        error_types[error['error_type']] += 1
        entity_type_errors[error['entity_type']] += 1
    
    print(f"\nError breakdown:")
    for error_type, count in error_types.items():
        print(f"  {error_type}: {count}")
    
    print(f"\nErrors by entity type:")
    for entity_type, count in sorted(entity_type_errors.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type}: {count}")
    
    # Show entity-level precision/recall/F1
    print(f"\nPer-entity performance:")
    for entity_type in sorted(entity_stats.keys()):
        tp = entity_stats[entity_type]['tp']
        fp = entity_stats[entity_type]['fp'] 
        fn = entity_stats[entity_type]['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  {entity_type:15} - P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f} (TP:{tp}, FP:{fp}, FN:{fn})")
    
    return pd.DataFrame(errors)

def get_entity_text_from_tokens(token_labels, original_text, tokenizer, start_pos, end_pos):
    """
    Try to extract the actual entity text from the original text.
    This is approximate since we don't have perfect token-to-char mapping.
    """
    # For now, just show the token sequence - could be improved with better alignment
    entity_tokens = []
    for label in token_labels:
        if label != 'O':
            entity_type = label.split('-')[-1]
            break
    
    return f"tokens_{start_pos}-{end_pos}"  # Placeholder - could be improved



def get_entity_text_from_positions(text, token_positions, tokenizer, label_sequence):
    """
    Extract actual entity text from the original text using token positions.
    """
    try:
        # Re-tokenize the text to get offset mappings
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            return_offsets_mapping=True
        )
        
        # Parse token positions (e.g., "19-19" -> start=19, end=19)
        start_token, end_token = map(int, token_positions.split('-'))
        
        # Adjust for special tokens (skip CLS token)
        offset_mapping = encoding['offset_mapping']
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        
        # Find the content token offset (skip special tokens)
        content_start = 1 if tokens[0] in ['<s>', '[CLS]', '<cls>'] else 0
        
        # Get character positions for the entity tokens
        actual_start_token = content_start + start_token
        actual_end_token = content_start + end_token
        
        if actual_start_token < len(offset_mapping) and actual_end_token < len(offset_mapping):
            char_start = offset_mapping[actual_start_token][0]
            char_end = offset_mapping[actual_end_token][1]
            
            # Extract the actual text
            entity_text = text[char_start:char_end].strip()
            return entity_text if entity_text else f"tokens_{start_token}-{end_token}"
        else:
            return f"tokens_{start_token}-{end_token}"
            
    except Exception as e:
        return f"tokens_{token_positions}"

def analyze_entity_errors_improved(y_true, y_pred, val_data, tokenizer):
    """
    Entity-level error analysis with proper text extraction.
    """
    print("\n" + "="*60)
    print("ENTITY-LEVEL ERROR ANALYSIS")
    print("="*60)
    
    errors = []
    
    for i, (true_seq, pred_seq, data_item) in enumerate(zip(y_true, y_pred, val_data)):
        text = data_item['text']
        
        # Extract entities using seqeval
        true_entities = get_entities(true_seq)
        pred_entities = get_entities(pred_seq)
        
        # Convert to sets for comparison
        true_set = set(true_entities)
        pred_set = set(pred_entities)
        
        # Find mismatches
        false_negatives = true_set - pred_set
        false_positives = pred_set - true_set
        
        # Record missing entities
        for entity_type, start, end in false_negatives:
            token_positions = f"{start}-{end}"
            entity_text = get_entity_text_from_positions(text, token_positions, tokenizer, true_seq)
            
            errors.append({
                'example_id': i,
                'full_text': text,
                'text_preview': text[:80] + "..." if len(text) > 80 else text,
                'error_type': 'Missing Entity',
                'entity_type': entity_type,
                'entity_text': entity_text,
                'token_positions': token_positions,
                'issue': f"You labeled '{entity_text}' as {entity_type}, but model didn't predict it",
                'question': f"Should '{entity_text}' be labeled as {entity_type}?",
                'model_correct': '',
                'notes': ''
            })
        
        # Record extra entities
        for entity_type, start, end in false_positives:
            token_positions = f"{start}-{end}"
            entity_text = get_entity_text_from_positions(text, token_positions, tokenizer, pred_seq)
            
            errors.append({
                'example_id': i,
                'full_text': text,
                'text_preview': text[:80] + "..." if len(text) > 80 else text,
                'error_type': 'Extra Entity',
                'entity_type': entity_type,
                'entity_text': entity_text,
                'token_positions': token_positions,
                'issue': f"Model predicted '{entity_text}' as {entity_type}, but you didn't label it",
                'question': f"Should '{entity_text}' be labeled as {entity_type}?",
                'model_correct': '',
                'notes': ''
            })
    
    return pd.DataFrame(errors)

def show_human_readable_analysis(errors_df):
    """
    Show errors in a human-readable format.
    """
    print(f"\n" + "="*80)
    print("HUMAN-READABLE ERROR ANALYSIS")
    print("="*80)
    
    print(f"Total errors: {len(errors_df)}")
    
    # Group by error type
    missing = errors_df[errors_df['error_type'] == 'Missing Entity']
    extra = errors_df[errors_df['error_type'] == 'Extra Entity']
    
    print(f"\nMissing entities (you labeled, model missed): {len(missing)}")
    print(f"Extra entities (model found, you didn't label): {len(extra)}")
    
    # Show the "39 to 44 chapel street" example clearly
    print(f"\n" + "="*60)
    print("EXAMPLE: Your '39 to 44 chapel street' case")
    print("="*60)
    
    example_2_errors = errors_df[errors_df['example_id'] == 2]
    if len(example_2_errors) > 0:
        print(f"Full text: {example_2_errors.iloc[0]['full_text']}")
        print(f"\nConflicts:")
        
        for _, error in example_2_errors.iterrows():
            print(f"  • {error['issue']}")
        
        print(f"\nThis shows:")
        print(f"  - You labeled: '39', 'to', '44', 'chapel', 'street' all as CITY")
        print(f"  - Model predicted: '39'=street_number, 'to'=nothing, '44'=street_number, 'chapel street'=street_name")
        print(f"  - The model is probably correct here!")
    
    # Show top entity types with issues
    print(f"\n" + "="*60)
    print("TOP ISSUES BY ENTITY TYPE")
    print("="*60)
    
    entity_issues = errors_df.groupby(['entity_type', 'error_type']).size().reset_index(name='count')
    entity_issues = entity_issues.sort_values('count', ascending=False)
    
    for _, row in entity_issues.head(10).iterrows():
        print(f"{row['entity_type']} - {row['error_type']}: {row['count']} cases")

def main():
    """
    Complete evaluation with human-readable error analysis.
    """
    # Get evaluation results
    y_true, y_pred, val_data, tokenizer = evaluate_ner_simple()
    """   
        # Analyze errors with proper text extraction
        errors_df = analyze_entity_errors_improved(y_true, y_pred, val_data, tokenizer)
        
        # Show human-readable analysis
        show_human_readable_analysis(errors_df)
        
        # Save results
        output_file = "human_readable_errors.csv"
        errors_df.to_csv(output_file, index=False)
        print(f"\nHuman-readable errors saved to: {output_file}")
        
        # Show first few examples
        print(f"\n" + "="*60)
        print("FIRST 5 ERRORS FOR REVIEW")
        print("="*60)
        
        for i, (_, error) in enumerate(errors_df.head(5).iterrows()):
            print(f"\n{i+1}. {error['issue']}")
            print(f"   Question: {error['question']}")
            print(f"   Context: {error['text_preview']}")
            print(f"   [Mark in CSV: model_correct = yes/no]")
    """
if __name__ == "__main__":
    main()