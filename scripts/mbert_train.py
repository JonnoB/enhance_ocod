"""
mbert_train.py

This script trains a ModernBERT-based token classification model for Named Entity Recognition (NER) on address components.
It performs the following steps:

1. Loads and preprocesses training and validation data from JSON files.
2. Defines the entity types to be recognized (e.g., building name, street name, postcode, etc.).
3. Configures the HuggingFace Trainer with appropriate hyperparameters and data collators.
4. Trains the model for a specified number of epochs.
5. Evaluates the model on the validation set using custom metrics.
6. Saves the trained model and tokenizer to the specified output directory.

Key dependencies:
    - transformers (HuggingFace)
    - torch
    - enhance_ocod.bert_utils (custom utilities for data processing and metrics)

Environment:
    Expects data files and output directories to be structured relative to the script location.

Typical usage:
    python mbert_train.py
"""

from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from pathlib import Path
from enhance_ocod.bert_utils import NERDataProcessor, compute_metrics, create_label_list
import torch

torch.set_float32_matmul_precision('high')

SCRIPT_DIR = Path(__file__).parent.absolute()

model_name = "answerdotai/ModernBERT-base"

train_data_path = str(SCRIPT_DIR / ".." / "data" / "training_data" / "training_data_dev.json") 
val_data_path = str(SCRIPT_DIR / ".." / "data" / "training_data" / "training_data_test.json")

num_train_epochs = 6
batch_size = 16
learning_rate = 5e-5
max_length  = 128
output_dir = str(SCRIPT_DIR / ".." / "models" / "address_parser")
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
val_data = processor.load_json_data(val_data_path) if val_data_path else None

# Create datasets
print("Processing training data...")
train_dataset = processor.create_dataset(train_data, max_length)
val_dataset = processor.create_dataset(val_data, max_length) if val_data else None

print(f"Training examples: {len(train_dataset)}")
if val_dataset:
    print(f"Validation examples: {len(val_dataset)}")



# Load model
print(f"Loading model: {model_name}")
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=processor.id2label,
    label2id=processor.label2id,
    ignore_mismatched_sizes=True  # In case of different vocab size
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
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=f"{output_dir}/logs",
    logging_steps=50,
    eval_strategy="epoch" if val_dataset else "no",
    eval_steps=50 if val_dataset else None,
    save_strategy="epoch",
    save_steps=200,
    load_best_model_at_end=True if val_dataset else False,
    metric_for_best_model="f1" if val_dataset else None,
    greater_is_better=True,
    push_to_hub=False,
    report_to="tensorboard",
    learning_rate=learning_rate,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics if val_dataset else None,
)

print("Starting training...")
trainer.train()