# The repo for empty homes london project

This is the code repo for the paper "Offshore homes in London, where are they, how many are there and what does it mean?"

The paper cleans and enhances the publicly available [OCOD dataset](https://use-land-property-data.service.gov.uk/datasets/ocod) produced by Land Registry. This dataset contains the addresses and additional metadata, UK property owned by offshore companies.

The repo contains all the code used in the paper as well as the datasets produced in it's creation.

#Code

The markdown and notebook files needed to re-create the work done in the paper as follows

1. Unit tag and span cleaning.ipynb
2. expanding tagged addresses.ipynb
3. Analysing the ocod dataset.ipynb
4. London_empty_homes.Rmd

In order to run these scripts you must download several opensource datasets produced by the UK government.
Please see the paper's data section in the method for details.

# Data
The datasets held in this repo are

- json file of NER labels produced by the [progammatic](https://programmatic.humanloop.com/) weak labelling process
- CSV of parsed OCOD addresses
- The enhanced OCOD data set with property type classifications

## This project is currently underdevelopment
