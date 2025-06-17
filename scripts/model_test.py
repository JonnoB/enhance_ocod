"""
Simple NER evaluation using seqeval - the way it should be done.
"""

import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForTokenClassification, AutoTokenizer
from enhance_ocod.bert_utils import NERDataProcessor
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
model_path = str(SCRIPT_DIR / ".." / "models" / "address_parser_dev"/ "final_model")
val_data_path = str(SCRIPT_DIR / ".." / "data" / "training_data" / 'ner_ready' /"ground_truth_test_set_labels.json")
max_length = 128

def evaluate_ner_simple():
    """
    Dead simple NER evaluation using seqeval.
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
    
    y_true = []  # True label sequences
    y_pred = []  # Predicted label sequences
    
    with torch.no_grad():
        for example in val_dataset:
            # Get predictions
            input_ids = torch.tensor([example['input_ids']]).to(device)
            attention_mask = torch.tensor([example['attention_mask']]).to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
            
            # Convert to label strings (skip special tokens)
            true_labels = []
            pred_labels = []
            
            for pred_id, true_id in zip(predictions, example['labels']):
                if true_id != -100:  # Skip special tokens
                    true_labels.append(id2label[true_id])
                    pred_labels.append(id2label[pred_id])
            
            y_true.append(true_labels)
            y_pred.append(pred_labels)
    
    # Evaluate with seqeval - that's it!
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
    
    # Debug: Show a few examples
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    
    for i in range(min(3, len(y_true))):
        print(f"\nExample {i+1}:")
        print(f"Text: {val_data[i]['text']}")
        print(f"True:  {y_true[i]}")
        print(f"Pred:  {y_pred[i]}")
        
        # Check if they match
        matches = sum(1 for t, p in zip(y_true[i], y_pred[i]) if t == p)
        total = len(y_true[i])
        print(f"Token accuracy: {matches}/{total} = {matches/total:.2f}")

if __name__ == "__main__":
    evaluate_ner_simple()