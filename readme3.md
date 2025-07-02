# Enhanced OCOD: Offshore Companies Ownership Data Processing Pipeline

## Overview
This repository provides a comprehensive pipeline and Python library for cleaning, enhancing, and analyzing the UK Land Registry's Offshore Companies Ownership Data (OCOD). The enhanced OCOD dataset resolves many issues with the raw OCOD data, making it suitable for research, analysis, and reporting on UK property owned by offshore companies.

The project includes:
- A reusable, modular Python library (`src/enhance_ocod`) for all data processing stages
- Example and utility scripts (`scripts/`) for training NER models, running the pipeline, and more
- Documentation and reproducible workflows to create, update, and analyze the enhanced OCOD dataset

## Key Features
- **End-to-End Pipeline:** From raw OCOD data to a classified, enriched, and structured dataset
- **Advanced Address Parsing:** Disaggregates multi-property titles and parses free-text addresses
- **Integration with External Data:** Uses ONS Postcode Directory, Land Registry Price Paid Data, and VOA business ratings for enrichment
- **Property Classification:** Assigns properties to categories (Residential, Business, Airspace, Land, Carpark, Unknown)
- **NER Model Training & Weak Labelling:** Tools for custom NER models and weak supervision
- **Reproducible & Extensible:** Library-based design for maintainability and reuse

## Project Structure
```
enhance_ocod/
├── src/enhance_ocod/   # Core Python library
│   ├── address_parsing.py
│   ├── inference.py
│   ├── labelling/
│   │   ├── ner_regex.py
│   │   ├── ner_spans.py
│   │   └── weak_labelling.py
│   ├── locate_and_classify.py
│   ├── preprocess.py
│   ├── price_paid_process.py
│   └── training.py
├── scripts/            # Example and utility scripts
├── data/               # Input and output data
├── full_ocod_parse_process.py # Main script to run the pipeline
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Project metadata
├── README.md, readme2.md, readme3.md # Documentation
```

## Installation
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd enhance_ocod
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Data Requirements
To recreate or update the enhanced OCOD dataset, several open datasets must be downloaded and placed in the `data/` (or `empty_homes_data/`) directory. Rename as follows:

| Dataset                                                                                             | Rename to            | Type   | API Available |
|-----------------------------------------------------------------------------------------------------|----------------------|--------|--------------|
| [OCOD dataset](https://use-land-property-data.service.gov.uk/datasets/ocod)                         | OCOD.csv             | csv    | Yes          |
| [ONSPD](https://open-geography-portalx-ons.hub.arcgis.com/datasets/ons::ons-postcode-directory-february-2025-for-the-uk/about) | ONSPD.zip            | zip    | Yes          |
| [Price Paid dataset](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads) | price_paid_files     | folder | No           |
| [VOA ratings list](https://voaratinglists.blob.core.windows.net/html/rlidata.htm)                   | VOA_ratings.csv      | csv    | Yes          |

**Note:**
- The OCOD dataset requires account creation and ID verification for download.
- See the original README for details on downloading and renaming files.

## Usage
You can use the project in two main ways:

### 1. As a Library
Import modules from `src/enhance_ocod` in your own scripts:
```python
from enhance_ocod.address_parsing import parse_address
# ...
```

### 2. Using Provided Scripts
- **Run the full pipeline:**
  ```bash
  python full_ocod_parse_process.py ./data/ OCOD.csv OCOD_enhanced.csv
  ```
  - Arguments: `<data_folder> <input_file> <output_file>` (all files must be in the data folder)

- **Train an NER Model:**
  ```bash
  python scripts/run_experiments.py
  ```

- **Other scripts:**
  See the `scripts/` directory for additional utilities (e.g., `parse_ocod_history.py`, `model_test.py`, etc.)

## Pipeline Stages
1. **NER Labelling** using a pre-trained spaCy model or custom models
2. **Parsing** and expanding addresses
3. **Disaggregation** so each row is a single property
4. **Geographic Location** using ONS/OA system
5. **Classification** into property types
6. **Cleanup** and deduplication
7. **Saving** the enhanced dataset

## Notebooks
Several Jupyter notebooks are included for development and analysis:
- `unit_tag_and_span_cleaning.ipynb`
- `expanding_tagged_addresses.ipynb`
- `locating_and_classifying_the_ocod_dataset.ipynb`

## Contributing
Contributions and suggestions are welcome! Please open issues or pull requests.

## Citation
If you use this repository, please cite:
- Bourne, J. (2023). "What's in the laundromat? Mapping and characterising offshore owned residential property in London"	 [https://doi.org/10.1177/2399808323115548](https://doi.org/10.1177/2399808323115548)


## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
- The enhanced OCOD dataset and pipeline were demonstrated in the paper: [Inspecting the laundromat](https://doi.org/10.1177/23998083231155483)
- Built on open data from Land Registry, ONS, and VOA

---
For more details, see the original `README.md` and `readme2.md` files, or contact the maintainer.
