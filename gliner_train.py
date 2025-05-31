from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollator
from data_utils import load_data, train_val_split, GLiNERDataset, evaluate_ner_spans
from config import TrainingConfig
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Any
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from datetime import datetime

import os
def train():
    # Load config
    config = TrainingConfig()
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    train_data = load_data(config.train_file)
    eval_data = load_data(config.eval_file) if Path(config.eval_file).exists() else None
    
    # If no eval data, split train data
    if not eval_data:
        train_data, eval_data = train_val_split(train_data)
    
    # Initialize model
    model = GLiNER.from_pretrained(config.model_name)

    id2label = {i: label for i, label in enumerate(config.entity_types)}
    label2id = {label: i for i, label in enumerate(config.entity_types)}

    # Update model config
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.config.num_labels = len(config.entity_types)
    
    # Initialize data collator
    data_collator = DataCollator(
        model.config, 
        data_processor=model.data_processor, 
        prepare_labels=True
    )
    
    # Create datasets
    train_dataset = GLiNERDataset(train_data, model.data_processor.transformer_tokenizer)
    eval_dataset = GLiNERDataset(eval_data, model.data_processor.transformer_tokenizer) if eval_data else None
    
    # Calculate number of epochs based on steps
    num_steps = 1000
    batch_size = config.batch_size
    data_size = len(train_dataset)
    num_batches = max(1, data_size // batch_size)
    num_epochs = max(1, num_steps // num_batches)
    
    # Use run_name from config
    run_name = config.run_name

    run_output_dir = output_dir / run_name
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(run_output_dir),
        run_name=run_name,
        learning_rate=config.lr,
        weight_decay=0.01,
        others_lr=1e-5,
        others_weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        eval_strategy="steps" if eval_data else "no",
        eval_steps=100,
        save_steps=100,
        logging_steps=50,
        report_to="tensorboard",
        save_total_limit=3,
        dataloader_num_workers=0,
        use_cpu=config.device == "cpu",
    )
    
    def compute_metrics(pred):
        """Compute metrics for evaluation."""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        # Flatten predictions and labels
        preds_flat = preds.flatten()
        labels_flat = labels.flatten()
        
        # Remove padding tokens (label = -100)
        mask = labels_flat != -100
        labels_flat = labels_flat[mask]
        preds_flat = preds_flat[mask]
        
        # Calculate metrics
        precision = precision_score(labels_flat, preds_flat, average='micro')
        recall = recall_score(labels_flat, preds_flat, average='micro')
        f1 = f1_score(labels_flat, preds_flat, average='micro')
        
        # Get per-class metrics
        class_report = classification_report(
            labels_flat, 
            preds_flat, 
            output_dict=True,
            target_names=model.config.id2label.values()
        )
        
        # Format metrics
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
        
        # Add per-class metrics
        for label, scores in class_report.items():
            if isinstance(scores, dict):
                metrics[f"{label}_precision"] = scores['precision']
                metrics[f"{label}_recall"] = scores['recall']
                metrics[f"{label}_f1"] = scores['f1-score']
                metrics[f"{label}_support"] = scores['support']
        
        return metrics

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_data else None,
    )
    
    # Train the model
    trainer.train()

    # Run evaluation if eval data exists
    if eval_data:
        # Standard trainer evaluation (loss metrics)
        eval_results = trainer.evaluate()
        print("\nTrainer evaluation results:")
        for key, value in eval_results.items():
            print(f"{key}: {value:.4f}")
        
        # NER-specific evaluation
        print("\nRunning NER evaluation...")
        try:
            ner_results = evaluate_ner_spans(model, eval_data)
            
            # Optionally save evaluation results
            import json
            with open(run_output_dir / "ner_evaluation_results.json", 'w') as f:
                json.dump(ner_results, f, indent=2)
                
        except Exception as e:
            print(f"Error in NER evaluation: {e}")
    
    # Save the final model
    model.save_pretrained(str(run_output_dir / "final_model"))
    print(f"\nTraining complete! Model saved to {run_output_dir / 'final_model'}")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    train()