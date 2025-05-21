from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from data_utils import load_data, save_data, train_val_split
from config import TrainingConfig
from pathlib import Path
import torch

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
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        learning_rate=config.lr,
        evaluation_strategy="epoch" if eval_data else "no",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True if eval_data else False,
        metric_for_best_model="f1" if eval_data else None,
        greater_is_better=True,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data if eval_data else None,
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    model.save_pretrained(str(output_dir / "final_model"))
    print(f"Training complete! Model saved to {output_dir / 'final_model'}")

if __name__ == "__main__":
    train()