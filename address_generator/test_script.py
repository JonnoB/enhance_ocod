#!/usr/bin/env python3
"""
Example script showing how to use the AddressGenerator
"""

from address_generator import AddressGenerator
import json

def main():
    # Initialize the generator with config file
    generator = AddressGenerator('address_generator_config.yaml')
    
    # Generate a single address
    print("=== Single Address Example ===")
    address = generator.generate_address()
    print(f"Generated Address: {address['text']}")
    print("\nSpans:")
    for span in address['spans']:
        print(f"  {span['label']}: '{span['text']}' (pos {span['start']}-{span['end']})")
    
    print("\n" + "="*50 + "\n")
    
    # Generate multiple addresses for training data
    print("=== Generating Training Dataset ===")
    training_data = []
    
    for i in range(10):
        address = generator.generate_address()
        training_data.append(address)
        print(f"{i+1:2d}. {address['text']}")
    
    # Save to file for training
    with open('synthetic_addresses.json', 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"\nSaved {len(training_data)} addresses to 'synthetic_addresses.json'")
    
    # Show some statistics
    print("\n=== Dataset Statistics ===")
    entity_counts = {}
    for address in training_data:
        for span in address['spans']:
            label = span['label']
            entity_counts[label] = entity_counts.get(label, 0) + 1
    
    print("Entity type distribution:")
    for entity, count in sorted(entity_counts.items()):
        print(f"  {entity}: {count}")

if __name__ == "__main__":
    main()