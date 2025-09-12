"""
The `enhance_ocod.labelling` submodule provides tools and utilities for weak supervision and rule-based Named Entity Recognition (NER) labelling, tailored for property address and related entity extraction tasks.

This submodule includes:
- **Regex Patterns** (`ner_regex.py`): A collection of reusable regular expressions for identifying address components such as roads, buildings, postcodes, cities, and business types.
- **NER Labelling Functions** (`ner_spans.py`): A suite of labelling functions that extract entity spans (start, end, label) from text. Functions are grouped by entity type (buildings, cities, postcodes, streets, units, etc.) and are designed to be library-agnostic.
- **Weak Labelling Utilities** (`weak_labelling.py`): Batch processing tools for applying labelling functions to pandas DataFrames, handling overlapping spans, and generating binary tags for special cases (e.g., flats, commercial parks).

Typical usage involves importing labelling functions from `ner_spans.py` and applying them to text data using the utilities in `weak_labelling.py`. The output is a list of entity spans suitable for training or evaluating NER models.

Example:
    ```python
    from enhance_ocod.labelling.weak_labelling import process_dataframe_batch
    import pandas as pd

    df = pd.DataFrame({'text': ["123 Main Street, London", "Flat 2, 45 Park Road"]})
    results = process_dataframe_batch(df)
    print(results[0]['spans'])  # [{'text': 'Main Street', 'start': 4, 'end': 15, 'label': 'ROAD'}, ...]
    ```

"""
