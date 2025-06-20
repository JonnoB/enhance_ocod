import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    from enhance_ocod.locate_and_classify import (classification_type1, classification_type2, contract_ocod_after_classification)
    return (
        classification_type1,
        classification_type2,
        contract_ocod_after_classification,
        os,
        pd,
    )


@app.cell
def _(os):
    processed_files_folder = 'data/ocod_history_processed/'

    processed_file_names = os.listdir(processed_files_folder)

    file_path = processed_file_names[83]
    return file_path, processed_files_folder


@app.cell
def _(
    classification_type1,
    classification_type2,
    contract_ocod_after_classification,
    file_path,
    os,
    pd,
    processed_files_folder,
):
    processed_file = pd.read_parquet(os.path.join(processed_files_folder, file_path))


    print('Classification type 1')
    processed_file = classification_type1(processed_file)
    print('Classification type 2')
    processed_file = classification_type2(processed_file)

    print('Contract ocod dataset')
    processed_file = contract_ocod_after_classification(processed_file, class_type='class2', classes=['residential'])

    columns = ['title_number', 'within_title_id', 'within_larger_title', 'unique_id', 
                  'unit_id', 'unit_type', 'building_name', 'street_number', 'street_name', 
                  'postcode', 'city', 'district', 'region', 'property_address', 'oa11cd', 
                  'lsoa11cd', 'msoa11cd', 'lad11cd', 'class', 'class2']

    processed_file = processed_file.loc[:, columns].rename(columns={
            'within_title_id': 'nested_id',
            'within_larger_title': 'nested_title'
        })
    return (processed_file,)


@app.cell
def _(processed_file):
    processed_file
    return


if __name__ == "__main__":
    app.run()
