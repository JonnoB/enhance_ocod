from dataclasses import dataclass, field
from typing import List, Dict

"""
    # Entity types to predict
    entity_types: List[str] = field(default_factory = lambda: [
        "building_name",
        "street_name",
        "street_number",
        "filter_type",
        "unit_id",
        "unit_type",
        "city",
        "postcode"
    ])
"""
@dataclass
class TrainingConfig:
    model_name: str = "urchade/gliner_medium"
    train_file: str =  "data/training_data/training_data_dev.json" #"data/training_data/training_data_dev.json" #"data/training_data/ground_truth_dev_set_labels.csv" 
    eval_file: str =  "data/training_data/training_data_test.json" # "None"#
    output_dir: str = "models/address_parser"
    batch_size: int = 32
    lr: float = 2e-5
    epochs: int = 10
    device: str = "cuda"  # or "cpu"
    

    entity_types: List[str] = field(default_factory = lambda: [
        "BUILDING_NAME",
        "STREET_NAME",
        "STREET_NUMBER",
        "NUMBER_FILTER",
        "UNIT_ID",
        "UNIT_TYPE",
        "CITY",
        "POSTCODE"
    ])
    
    # Run configuration
    run_name: str = "simple_test"  # Custom name for this training run

