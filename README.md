# This project is currently underdevelopment

# The Enhanced OCOD dataset

This repo provides the code necessary to create the enhanced OCOD dataset. The Enhanced OCOD dataset is based on the The paper cleans and enhances the publicly available [OCOD dataset](https://use-land-property-data.service.gov.uk/datasets/ocod) produced by Land Registry. This dataset contains the addresses and additional metadata, UK property owned by offshore companies.
produced by the land Registry. The OCOD dataset is a record of all UK property owned by and offshore company. The enhanced OCOD dataset, tidies the data ensuring that there is a single property per line, it also parses the address to make it easier to use and locates the properties within the [LSOA/OA](https://www.ons.gov.uk/methodology/geography/ukgeographies/censusgeography) system developed by the Office of National Statistics. Finally the OCOD dataset classifies properties into 5 categories, Domestic, Business, Airspace, Land, Carpark. Any properties which do not have enough information to be put in one of the categories are given the class "unknown".

The code in this repo has been used for the academic paper Offshore property in London where is it and what is?

# The Dataset

The compressed dataset from the time the paper was writted is included with this repo as 'OCOD_classes.tar.xz'. However, it will become out of date and so it is reccomended to rebuild the dataset with current information. The following section describes how to generate the enhanced OCOD dataset.

# How to create the Enhanced OCOD dataset
This repo contains the code needed to create the enchanced OCOD dataset. The simplest way is to build and run the Docker image found [here](dockerfile). Otherwise the script [full_ocod_parse_process.py] can be run from a python environment.

## Project set-up

- clone repo
- navigate to repo folder
- navigate to empty_homes_data
- download required opensource data and rename files/folder (see below)
- download spaCy model from xxx xxxx and put in empty_homes_data directory of the repo

## Docker version

See the docker readme for detailed instructions

- install docker
- build docker
- run docker

## Python version

in the command line type the following from the repo root folder

- pip install spacy numpy pandas #installs the required libraries
- python -m spacy download en_core_web_lg #the spaCy model uses the large vector language model optimized for CPU
- python full_ocod_parse_process.py ./empty_homes_data/

The script itself is [full_ocod_parse_process.py](full_ocod_parse_process.py)

## During the parsing and classification process

The process takes about 15 minutes and is broken into several stages. The stages provide information on progress and at the a file called
'enhanced_ocod_dataset.csv' is saved into the the 'empty_homes_data' folder.

The key stages are

- NER labelling using a pre-trained spaCy model
- Parsing
- Expanding so that there is only one property per row
- Locating in the Census Geography system
- Classifying into the 5 types and 'unknown'
- Contracting, by removing irrelevant duplicated data
- Saving the enhanced ocod dataset


# Required Datasets

In order to re-create or update the Enhanced OCOD dataset several opensource datasets are required. This datasets should be downloaded into the empty_homes_data folder in this repo and renamed as below. All the data is free and covered by an open government licence (OGL), however the OCOD dataset requires the user create and account.

| Dataset                                                                                             | Change file/folder name to | Type   |
|-----------------------------------------------------------------------------------------------------|----------------------------|--------|
| [ OCOD dataset ]( https://use-land-property-data.service.gov.uk/datasets/ocod )                     | OCOD.csv                   | CSV    |
| [ONSPD](https://geoportal.statistics.gov.uk/search?q=onspd)                                         | ONSPD.csv                  | csv    |
| [Price Paid dataset](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads)* | price_paid_files           | Folder |
| [VOA ratings list](https://voaratinglists.blob.core.windows.net/html/rlidata.htm)                   | VOA_ratings.csv            | CSV    |

*Note: The price paid dataset should be downloaded as yearly files and places inside a folder called 'price_paid_files'. It is advisable to download several years. The paper used 2017-2021. Having more years increases the chances of being able to fill in missing information in OCOD, however after a few years the benefits reduce and the memory costs become high.

# Paper  Code

The markdown and notebooks used in this paper are as follows.

1. [Unit tag and span cleaning](unit_tag_and_span_cleaning.ipynb)
2. [expanding tagged addresses](expanding_tagged_addresses.ipynb)
3. [Locating and classifying the ocod dataset](locating_and_classifying_the_ocod_dataset.ipynb)
4. [London empty homes](London_empty_homes.Rmd)

In order to run these scripts you must download several opensource datasets produced by the UK government.
Please see the paper's data section in the method for details.

# Citing this dataset

If you use this dataset please cite

XXXX

# OGL notices

- Contains HM Land Registry data © Crown copyright and database right 2021. This data is licensed under the Open Government Licence v3.0.
- Contains OS data © Crown copyright and database right 2022
- Contains Royal Mail data © Royal Mail copyright and database right 2022
- Source: Office for National Statistics licensed under the Open Government Licence v.3.0

