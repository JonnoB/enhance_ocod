from gliner import GLiNER
from typing import List, Dict, Any
from pathlib import Path
import torch

class AddressParser:
    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GLiNER.from_pretrained(model_path).to(self.device)
        self.model.eval()
    
    def predict(self, text: str, entity_types: List[str] = None, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Predict entities in a single address string."""
        if not entity_types:
            entity_types = [
                "building_name",
                "street_name",
                "unit_id",
                "unit_type",
                "city",
                "postcode"
            ]
            
        entities = self.model.predict_entities(text, entity_types, threshold=threshold)
        
        return [
            {
                "text": ent["text"],
                "start": ent["start"],
                "end": ent["end"],
                "label": ent["label"],
                "score": float(ent["score"])
            }
            for ent in entities
        ]
    
    def predict_batch(self, texts: List[str], entity_types: List[str] = None, 
                     batch_size: int = 32, threshold: float = 0.5) -> List[List[Dict[str, Any]]]:
        """Predict entities in a batch of address strings."""
        if not entity_types:
            entity_types = [
                "building_name",
                "street_name",
                "unit_id",
                "unit_type",
                "city",
                "postcode"
            ]
            
        all_entities = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_entities = self.model.predict_entities_batch(
                texts=batch,
                entity_types=entity_types,
                threshold=threshold
            )
            all_entities.extend(batch_entities)
            
        return all_entities

# Example usage
if __name__ == "__main__":
    # Initialize parser
    parser = AddressParser(model_path="models/address_parser")
    
    # Example prediction
    address = "Flat 3, 25 Test Road, London, SW1A 1AA"
    entities = parser.predict(address)
    
    print(f"Address: {address}")
    for entity in entities:
        print(f"- {entity['label']}: {entity['text']} (confidence: {entity['score']:.2f})")