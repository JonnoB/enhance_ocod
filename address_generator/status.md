Address Generator Project Status & Continuation Plan
📋 Project Overview
This project creates synthetic UK address data for training NER models using a tag-based generation system. The generator produces realistic, messy addresses that match the quality and variety found in UK government datasets.
✅ Completed Components
Core Architecture (100% Complete)

AddressGenerator: Main orchestrator class with tag parsing system
ComponentBuilder: Generates realistic UK address components with tags
StructureGenerator: Determines address complexity (single/range/list properties)
Configuration System: YAML-based config with external name file imports
Tag Processing: Accurate span extraction with underscore support

Generation Features (90% Complete)

✅ Property type sampling (residential, business, land, carpark, airspace)
✅ UK postcode generation with realistic case variations
✅ Street/building name generation with possessives ("Rose's Road")
✅ Number generation: single, ranges ("5 to 15"), lists ("2, 4, and 6")
✅ Granular entity tagging (individual numbers get separate spans)
✅ Filter conditions ("odds only", "evens") with plural variations
✅ Unit types and IDs for residential properties

Technical Implementation (95% Complete)

✅ Accurate span calculation with cumulative offset tracking
✅ Error handling for missing config files and imports
✅ Extensible base name + suffix combination system
✅ Configurable probability distributions

🚧 In Progress / Needs Implementation
1. CorruptionEngine (Priority: HIGH)
Status: Planned but not implemented
Location: Currently commented out in AddressGenerator.__init__() and generate_address()
What it should do:

Remove components based on property type (Land parcels often missing cities)
Add formatting inconsistencies (spacing, punctuation)
Create partial addresses
Apply corruption to tagged components before parsing

Implementation outline:
python


