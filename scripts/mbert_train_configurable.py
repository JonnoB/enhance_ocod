"""
mbert_train_configurable.py

Configurable training script for ModernBERT-based token classification model for Named Entity Recognition (NER).
Accepts command line arguments to specify different training configurations.
"""

from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from pathlib import Path
from enhance_ocod.training import NERDataProcessor, create_label_list, evaluate_model_performance
import torch
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from seqeval.metrics.sequence_labeling import get_entities

torch.set_float32_matmul_precision('medium')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train NER model with configurable data sources')
    
    parser.add_argument('--data_folder', type=str, required=True,
                       choices=['ner_ready', 'ner_ready_preprocessed'],
                       help='Data folder to use (ner_ready or ner_ready_preprocessed)')
    
    parser.add_argument('--train_file', type=str, required=True,
                       choices=['ground_truth_dev_set_labels.json', 'full_dataset_no_overlaps.json'],
                       help='Training data file to use')
    
    parser.add_argument('--model_suffix', type=str, required=True,
                       help='Suffix for the model output directory')
    
    parser.add_argument('--num_epochs', type=int, default=6,
                       help='Number of training epochs')
    
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    SCRIPT_DIR = Path(__file__).parent.absolute()
    model_name = "answerdotai/ModernBERT-base"
    
    # Construct paths based on arguments
    data_base_path = SCRIPT_DIR / ".." / "data" / "training_data" / args.data_folder
    train_data_path = str(data_base_path / args.train_file)
    val_data_path = str(data_base_path / "ground_truth_test_set_labels.json")  # Always use test set for validation
    
    # Create output directory name based on configuration
    output_dir_name = f"address_parser_{args.model_suffix}"
    output_dir = str(SCRIPT_DIR / ".." / "models" / output_dir_name)
    final_model_path = f"/teamspace/studios/this_studio/enhance_ocod/models/{output_dir_name}/final_model"
    
    print(f"Configuration:")
    print(f"  Data folder: {args.data_folder}")
    print(f"  Training file: {args.train_file}")
    print(f"  Validation file: ground_truth_test_set_labels.json")
    print(f"  Output directory: {output_dir}")
    print(f"  Final model path: {final_model_path}")
    print("-" * 50)
    
    entity_types = [
        "building_name",
        "street_name",
        "street_number",
        "filter_type",
        "unit_id",
        "unit_type",
        "city",
        "postcode"
    ]
    
    # Create label list
    label_list = create_label_list(entity_types)
    print(f"Label list: {label_list}")
    
    # Initialize data processor
    processor = NERDataProcessor(label_list, model_name)
    
    # Load data
    print("Loading training data...")
    train_data = processor.load_json_data(train_data_path)
    val_data = processor.load_json_data(val_data_path)
    
    # Create datasets
    print("Processing training data...")
    train_dataset = processor.create_dataset(train_data, args.max_length)
    val_dataset = processor.create_dataset(val_data, args.max_length)
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=processor.id2label,
        label2id=processor.label2id,
        ignore_mismatched_sizes=True
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=processor.tokenizer,
        pad_to_multiple_of=8,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=500,
        eval_strategy="epoch",
        eval_steps=500,
        save_strategy="epoch", 
        save_steps=1,  
        save_total_limit=1, 
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
        report_to="tensorboard",
        learning_rate=args.learning_rate,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer,
        compute_metrics=processor.compute_entity_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save the model and tokenizer
    trainer.model.save_pretrained(final_model_path)
    processor.tokenizer.save_pretrained(final_model_path)
    
    print(f"Final model saved to: {final_model_path}")
    
    # Evaluate model performance
    overall_df, class_df = evaluate_model_performance(
        model_path=final_model_path,
        data_path=val_data_path,
        output_dir=final_model_path,
        dataset_name="test",
        max_length=args.max_length
    )
    
    print(f"Performance metrics saved to: {final_model_path}")
    print("Training and evaluation complete!")

if __name__ == "__main__":
    main()