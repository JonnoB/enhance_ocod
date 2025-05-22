from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from data_utils import load_data, save_data, train_val_split, GLiNERDataset
from config import TrainingConfig
from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import List, Dict

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
    
    # Create datasets
    train_dataset = GLiNERDataset(train_data, model.tokenizer)
    eval_dataset = GLiNERDataset(eval_data, model.tokenizer) if eval_data else None
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        learning_rate=config.lr,
        eval_strategy="epoch" if eval_data else "no",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=bool(eval_data),
        metric_for_best_model="f1" if eval_data else None,
        greater_is_better=True,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    model.save_pretrained(str(output_dir / "final_model"))
    print(f"Training complete! Model saved to {output_dir / 'final_model'}")

if __name__ == "__main__":
    train()