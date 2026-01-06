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
- **NER Model Training & Weak Labelling:** Fine-tuned modernBERT model automaticall downloaded from [HF](https://huggingface.co/Jonnob/OCOD_NER)
- **Reproducible & Extensible:** Library-based design for maintainability and reuse

## Project Structure
```
enhance_ocod/
├── src/enhance_ocod/   # Core Python library
│   ├── address_parsing.py
│   ├── analysis.py
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
├── notebooks/          # Analysis performed for the paper
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Project metadata
├── README.md           # Documentation
```

## Installation

### Option 1: Install from PyPI (Recommended)
```bash
pip install enhance-ocod
```

### Option 2: Install from GitHub (Latest Development Version)
```bash
pip install git+https://github.com/JonnoB/enhance_ocod.git
```

### Option 3: Development Installation
If you want to contribute or modify the code:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JonnoB/enhance_ocod.git
   cd enhance_ocod
   ```

2. **Install in development mode:**
   ```bash
   pip install -e .
   ```
   
   Or if you're using uv:
   ```bash
   uv pip install -e .
   ```

**Notes:**
- The package name for installation is `enhance-ocod` (with hyphen)
- The import name is `enhance_ocod` (with underscore)
- Python automatically handles this naming conversion

## Data Requirements
To recreate or update the enhanced OCOD dataset, several open datasets are required. 
The `get_data` module has the functionality to download the required files, and the `download_hist.py` script can be used to perform downloading automatically. If done manually the files must be downloaded and placed in subd-directories of the `data/` directory. The sub-directories should be named as follows:

| Dataset                                                                                             | Folder          | Type   | API Available |
|-----------------------------------------------------------------------------------------------------|----------------------|--------|--------------|
| [OCOD dataset](https://use-land-property-data.service.gov.uk/datasets/ocod)                         | ocod_history        | csv    | Yes          |
| [ONSPD](https://open-geography-portalx-ons.hub.arcgis.com/datasets/ons::ons-postcode-directory-february-2025-for-the-uk/about) | onspd           | zip    | Yes          |
| [Price Paid dataset](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads) | price_paid_data     | folder | No           |
| [VOA ratings list](https://voaratinglists.blob.core.windows.net/html/rlidata.htm)                   | voa     | csv    | Yes          |

**Note:**
- The OCOD dataset is a convoluted experience to get hold of you need to create an account and also use a bank card to confim identity, the bank card will be charged £0.0. Whether this much security is necessary is debatable, and in fact can be debated by contacting your [MP to complain](https://members.parliament.uk/FindYourMP).


## Usage
You can use the project in two main ways:

### 1. As a Library
Import modules from `src/enhance_ocod` in your own scripts:
```python
import pandas as pd
from enhance_ocod.inference import parse_addresses_basic

# Create example DataFrame with the two addresses
example_df = pd.DataFrame({
    'address': [
        "36 - 49, chapel street, London, se45 6pq",
        "Flat 14a, 14 Barnsbury Road, London N1 1JU"
    ],
    'datapoint_id': ['addr_001', 'addr_002']  # Optional unique identifiers
})

print("Example DataFrame:")
print(example_df)

# Default behaviour is to download the finetuned model from Hugginface model library.
results = parse_addresses_basic(example_df)
print(f"Parsed {results['summary']['successful_parses']} addresses")
# ...
```

### 2. Using Provided Scripts
- **Run the full pipeline:**
  ```bash
  python scripts/parse_ocod_history.py
  ```

  **With custom paths:**
  ```bash
  # Specify custom input and output directories
  python scripts/parse_ocod_history.py --input-dir /path/to/input --output-dir /path/to/output

  # Use a local model instead of HuggingFace
  python scripts/parse_ocod_history.py --model-path /path/to/local/model

  # Combine all options
  python scripts/parse_ocod_history.py --input-dir /custom/input --output-dir /custom/output --model-path /custom/model
  ```

- **Train an NER Model:**
  ```bash
  python scripts/run_experiments.py
  ```

### Order to run the scripts in
To Parse the OCOD history you can simply run

```bash
python scripts/download_hist.py && python scripts/parse_ocod_history.py
```
This will download all necessary data and then create a folder called `ocod_history_processed` with one standardised OCOD parquet file per original ocod file. Using an L4 GPU with 24GB VRAM and 16 GB RAM it will take 2-3 hours to process the first decade of OCOD. This will typically cost less than $2.

**Command-line options:**
The `parse_ocod_history.py` script supports the following optional arguments:
- `--input-dir`: Input directory containing OCOD_FULL_*.zip files (default: `data/ocod_history`)
- `--output-dir`: Output directory for processed parquet files (default: `data/ocod_history_processed`)
- `--model-path`: Model path - either a HuggingFace model ID or local path (default: `Jonnob/OCOD_NER`)

If no arguments are provided, the script uses the default paths and downloads the model from HuggingFace.


To reproduce the Paper run the files in the following order

- `scripts/download_hist.py`: Downloads the entire OCOD dataset history and saves by year as zip files. Requires a 'LANDREGISTRY_API' in the .env file.
- `scripts/create_weak_labelling_data.py`: Using the regex rules weakly label the OCOD February 2022 data set
- `scripts/ready_csv_for_training.py`: Create the datasets for training and evaluation of the models out of the development set, weakly labelled set and test set.
- `scripts/run_experiments.py`: Using the dev and weakly labelled sets, train the ModernBERT models. The script also calls the `mbert_train_configurable.py` script.
- `scripts/parse_ocod_history.py`: Processes the entire history of the OCOD dataset. Use `--model-path` to specify a local model or omit to use the HuggingFace model.
- `scripts/price_paide_msoa_averages.py`: Calculates the mean price per MSOA, for a rolling three years. This is used by `price_paid_msoa_averages.ipynb.

After you can run the .ipynb files to output the analysis.

## Pipeline Stages

The entire process containsed in `parse_ocod_history.py` is as follows

1. **NER Labelling** using a pre-trained modernBERT model
2. **Parsing** Create a dataframe using the entities
3. **Geographic Location** using ONS/OA system
5. **Classification** into property types
6. **Cleanup** Expand addresses that are actually multiple addresses (e.g. "Flats 3-10")
7. **Contraction** ensure non-residential properties are only a single row

## Notebooks
Several Jupyter notebooks are included for development and analysis (located in the `notebooks/` directory). These are primarily for the analysis used in the paper:
- `notebooks/exploratory_analysis.ipynb`
- `notebooks/price_paid_msoa.ipynb`
- `notebooks/test_regex.ipynb`

## Pre-trained NER model

The fine-tuned modernBERT model is available to [download](https://huggingface.co/Jonnob/OCOD_NER) from huggingface. The model can be run directly on address strings using huggingface 'pipeline' functionality, see the [model card](https://huggingface.co/Jonnob/OCOD_NER) for details.

## Contributing
Contributions and suggestions are welcome! Please open issues or pull requests.

## Citation
If you use this repository, please cite:
- J Bourne et al (2023). "What's in the laundromat? Mapping and characterising offshore owned residential property in London"	 [https://doi.org/10.1177/2399808323115548](https://doi.org/10.1177/2399808323115548)


## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
- The enhanced OCOD dataset and pipeline were demonstrated in the paper: [Inspecting the laundromat](https://doi.org/10.1177/23998083231155483)
- Built on open data from Land Registry, ONS, and VOA

---