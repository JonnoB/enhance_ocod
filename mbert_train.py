from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from bert_utils import NERDataProcessor, compute_metrics, create_label_list
import torch

torch.set_float32_matmul_precision('high')

model_name = "answerdotai/ModernBERT-base"

train_data_path = "data/training_data/training_data_dev.json" 
val_data_path =  "data/training_data/training_data_test.json"

num_train_epochs = 6
batch_size = 16
learning_rate = 5e-5
max_length  = 128
output_dir = "models/address_parser"
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