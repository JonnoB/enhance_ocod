from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollator
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
    num_steps = 1000  # You can adjust this
    batch_size = config.batch_size
    data_size = len(train_dataset)
    num_batches = max(1, data_size // batch_size)
    num_epochs = max(1, num_steps // num_batches)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
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
        logging_steps=50,  # Log metrics every 50 steps
        report_to="tensorboard",
        save_total_limit=3,
        dataloader_num_workers=0,
        use_cpu=config.device == "cpu",
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )
    
    # Train the model
    trainer.train()

    if eval_data:
        eval_results = trainer.evaluate()
        print("\nEvaluation results:")
        for key, value in eval_results.items():
            print(f"{key}: {value:.4f}")
    
    # Save the final model
    model.save_pretrained(str(output_dir / "final_model"))
    print(f"Training complete! Model saved to {output_dir / 'final_model'}")

if __name__ == "__main__":
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    train()