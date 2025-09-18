"""
Address geolocation and property classification for OCOD database.

This module provides functionality for geolocating addresses in the OCOD database within 
the UK government's Output Area (OA), Lower Super Output Area (LSOA), 
and Local Authority District (LAD) geographical framework.

The module also provides classification capabilities for identifying property 
types (residential, business, land, etc.) and detecting multi-property 
developments through rule-based classification and business address matching 
against Valuation Office Agency (VOA) data.

Key Features
------------
- Postcode-based geographic area lookup and assignment
- Address standardization and cleaning for improved matching
- Business property identification using VOA rating list data
- Rule-based property type classification system
- Address component matching (building names, street names, numbers)
- Geographic gazetteer enhancement for missing location data
- Classification performance evaluation metrics

Main Functions
--------------
load_postcode_district_lookup : Load ONS postcode to geographic area mappings
load_voa_ratinglist : Load and process VOA business property data
add_geographic_metadata : Enrich addresses with geographic area codes
enhance_ocod_with_gazetteers : Fill missing geographic data using gazetteers
add_business_matches : Perform address matching against business database
property_class : Apply rule-based property type classification
evaluate_classification_predictions : Assess classification performance

Dependencies
------------
- pandas : Data manipulation and analysis
- numpy : Numerical computing
- sklearn.metrics : Classification evaluation metrics
- .labelling.ner_regex : Address pattern recognition utilities

Notes
-----
This module is designed for processing UK property data and requires 
specific data formats from ONS and VOA sources. Classification rules 
are configurable and can be customized for different use cases.

The geographic framework follows UK government statistical geography 
standards with hierarchical area codes (OA → LSOA → MSOA → LAD).
"""

import io
import re
import zipfile
import pandas as pd
import numpy as np
from .labelling.ner_regex import xx_to_yy_regex

from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    classification_report,
    accuracy_score
)

def load_postcode_district_lookup(file_path, target_post_area=None):
    """
    Load the ONS postcode district lookup table from a ZIP file.

    Loads the ONSPD (Office for National Statistics Postcode Directory) and reduces
    it to relevant columns to save memory. Filters for English and Welsh postcodes.

    Args:
        file_path (str): Path to the ZIP file containing the ONSPD data.
        target_post_area (str, optional): Specific CSV file within the ZIP to load.
            If None, automatically finds the appropriate file.

    Returns:
        pd.DataFrame: Processed postcode district lookup table with selected columns.
    """
    with zipfile.ZipFile(file_path) as zf:
        # If no target file specified, find it automatically
        if target_post_area is None:
            target_post_area = [
                i for i in zf.namelist() if re.search(r"^Data/ONSPD.+csv$", i)
            ][0]

        with io.TextIOWrapper(zf.open(target_post_area), encoding="latin-1") as f:
            # Define dtypes for mixed-type columns to suppress warning and improve performance
            dtype_spec = {
                "pcds": "string",
                "oslaua": "string",
                "oa11": "string",
                "lsoa11": "string",
                "msoa11": "string",
                "ctry": "string",
            }

            # Read only required columns with specified dtypes
            postcode_district_lookup = pd.read_csv(
                f,
                usecols=["pcds", "oslaua", "oa11", "lsoa11", "msoa11", "ctry"],
                dtype=dtype_spec,
                low_memory=False,
            )

            # Filter for English and Welsh postcodes using isin() for better performance
            target_countries = ["E92000001", "W92000004"]
            postcode_district_lookup = postcode_district_lookup[
                postcode_district_lookup["ctry"].isin(target_countries)
            ]

            # Process postcode column more efficiently
            postcode_district_lookup["pcds"] = (
                postcode_district_lookup["pcds"]
                .str.lower()
                .str.replace(" ", "", regex=False)  # Non-regex replacement is faster
            )

            # Rename columns
            postcode_district_lookup.rename(
                columns={
                    "pcds": "postcode2",
                    "oslaua": "lad11cd",
                    "oa11": "oa11cd",
                    "lsoa11": "lsoa11cd",
                    "msoa11": "msoa11cd",
                },
                inplace=True,
            )

            # Drop country column
            postcode_district_lookup.drop("ctry", axis=1, inplace=True)

    return postcode_district_lookup


def clean_street_numbers(df, original_column="number_or_name"):
    """
    Optimized version with fewer pandas operations and better vectorization
    """

    # Create a copy to work with
    street_num = df[original_column].copy()

    # Chain multiple replacements in fewer operations
    street_num = (
        street_num.str.replace(r"\(.+\)", "", regex=True, case=False)
        .str.replace(r"\+", " ", regex=True, case=False)
        .str.replace(r"@", " at ", regex=True, case=False)
        .str.replace(r"&", " and ", regex=True, case=False)
        .str.replace(r"\s*-\s*", "-", regex=True, case=False)
    )

    # Handle unit/suite/room in one vectorized operation
    unit_mask = street_num.str.contains(
        r"unit|suite|room", regex=True, case=False, na=False
    )
    street_num = street_num.where(~unit_mask)

    # Continue with remaining operations in a chain
    street_num = (
        street_num.str.extract(r"([^\s]+)$")[0]
        .str.replace(r"[a-zA-Z]+", "", regex=True, case=False)
        .str.replace(r"^-+|-+$", "", regex=True, case=False)
        .str.replace(r"[\\\/]", " ", regex=True, case=False)
        .str.replace(r"-{2,}", "-", regex=True, case=False)
    )

    # Clean up empty strings and single hyphens in one operation
    cleanup_mask = (street_num.str.len() > 0) & (street_num != "-")
    street_num = street_num.where(cleanup_mask)

    df["street_number"] = street_num
    return df


##street matching

def load_voa_ratinglist(file_path, postcode_district_lookup):
    """
    Optimized version - only load necessary columns, use categorical data types,
    and minimize memory usage
    """
    
    def find_largest_file(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            largest_file = max(zip_ref.infolist(), key=lambda x: x.file_size)
            return largest_file.filename

    # Only load the columns we actually need for processing
    needed_column_indices = [4, 5, 7, 9, 10, 14]  # Only the essential columns
    column_names = [
        "primary_and_secondary_description_code",  # 4 - for filtering
        "primary_description_text",                # 5 - for filtering  
        "full_property_identifier",                # 7 - for westfield processing
        "number_or_name",                         # 9 - for street_number and building_name
        "street",                                 # 10 - for street_name2
        "postcode"                                # 14 - for merge
    ]
    

    dtype_dict = {
        "primary_and_secondary_description_code": "category",
        "primary_description_text": "category", 
        "full_property_identifier": "string",
        "number_or_name": "string", 
        "street": "string",
        "postcode": "string"
    }

    # Read data with optimizations
    if file_path.lower().endswith(".zip"):
        largest_file = find_largest_file(file_path)
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            with zip_ref.open(largest_file) as csv_file:
                voa_businesses = pd.read_csv(
                    csv_file,
                    sep="*",
                    encoding_errors="ignore",
                    header=None,
                    names=column_names,
                    usecols=needed_column_indices,
                    dtype=dtype_dict,
                    index_col=False,
                )
    else:
        voa_businesses = pd.read_csv(
            file_path,
            sep="*",
            encoding_errors="ignore",
            header=None,
            names=column_names,
            usecols=needed_column_indices,
            dtype=dtype_dict,
            index_col=False,
        )


    # Early filtering with optimized operations
    excluded_codes = {"C0", "CP", "CP1", "CX", "MX"}
    
    # Use vectorized operations and chain filters
    mask = (
        ~voa_businesses["primary_description_text"].str.contains("ADVERTISING", na=False) &
        ~voa_businesses["primary_and_secondary_description_code"].isin(excluded_codes)
    )
    voa_businesses = voa_businesses[mask]
    
    # Efficient string processing - do all operations at once
    voa_businesses["postcode"] = voa_businesses["postcode"].str.lower()
    voa_businesses["postcode2"] = voa_businesses["postcode"].str.replace(r"\s", "", regex=True)
    
    # Convert postcode2 to categorical after processing
    voa_businesses["postcode2"] = voa_businesses["postcode2"].astype("category")

    # Process street names efficiently
    street_lower = voa_businesses["street"].str.lower()
    voa_businesses["street_name2"] = (
        street_lower
        .str.replace(r"'", "", regex=True)
        .str.replace(r"s(s)?(?=\s)", "", regex=True)
        .str.replace(r"\s", "", regex=True)
    ).astype("category")  # Convert to categorical

    # Early merge to reduce dataset size
    voa_businesses = voa_businesses.merge(
        postcode_district_lookup, 
        left_on="postcode2", 
        right_on="postcode2", 
        how="inner"
    )
    

    # Process on smaller dataset
    voa_businesses = clean_street_numbers(
        voa_businesses, original_column="number_or_name"
    )

    # Efficient conditional operations
    westfield_mask = voa_businesses["full_property_identifier"].str.contains(
        "WESTFIELD SHOPPING CENTRE", na=False
    )
    if westfield_mask.any():
        voa_businesses.loc[westfield_mask, "street_number"] = pd.NA

    if "street_number" in voa_businesses.columns:
        phone_mask = voa_businesses["street_number"].str.contains(
            r"-\d+-", regex=True, na=False
        )
        if phone_mask.any():
            voa_businesses.loc[phone_mask, "street_number"] = pd.NA

    # Building name extraction
    voa_businesses["building_name"] = (
        voa_businesses["number_or_name"]
        .str.lower()
        .str.extract(
            r"((?<!\S)(?:(?!\b(?:\)|\(|r\/o|floor|floors|pt|and|annexe|room|gf|south|north|east|west|at|on|in|of|adjoining|adj|basement|bsmt|fl|flr|flrs|wing)\b)[^\n\d])*? house\b)",
            expand=False,
        )
    )

    # Convert final columns to optimal types
    if "street_number" in voa_businesses.columns:
        voa_businesses["street_number"] = voa_businesses["street_number"].astype("category")
    
    if "building_name" in voa_businesses.columns:
        voa_businesses["building_name"] = voa_businesses["building_name"].astype("category")

    # Return only the columns we need
    final_columns = ['street_name2', 'street_number', 'building_name', 
                    'oa11cd', 'lsoa11cd', 'msoa11cd', 'lad11cd']
    
    # Only select columns that exist
    existing_columns = [col for col in final_columns if col in voa_businesses.columns]
    
    return voa_businesses[existing_columns].copy()


def add_geographic_metadata(df, postcode_district_lookup):
    """
    Add geographic area data and create standardized address fields, for entries with a postcode.
    
    This function enriches address data by merging geographic area codes 
    (LSOA, MSOA, LAD, etc.) based on postcode lookup, and creates 
    standardized versions of street numbers and street names for 
    improved data matching and consistency.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing address data with columns including 
        'postcode', 'street_number', and 'street_name'.
    postcode_district_lookup : pandas.DataFrame
        Lookup table containing postcode to geographic area mappings
        with 'postcode2' column for merging.
    
    Returns
    -------
    pandas.DataFrame
        Enhanced DataFrame with additional columns:
        - postcode2: Normalized postcode (lowercase, no spaces)
        - Geographic area columns from the lookup table
        - street_number2: Standardized street number (digits only)
        - street_name2: Standardized street name (no apostrophes, 
          's/ss' removed, no spaces)
    
    Notes
    -----
    The standardized street number extracts only digits for use in 
    LSOA matching where apartment letters and other characters are 
    not relevant.
    """

    ##add in the geographic area data like lsoa etc
    df["postcode2"] = (
        df["postcode"].str.lower().str.replace(r"\s", "", regex=True)
    )

    df = df.merge(
        postcode_district_lookup, "left", left_on="postcode2", right_on="postcode2"
    )

    # this is to ensure that the street number includes only digits as it is used in the LSOA matching where the only thing
    # that matters is the street number not whether it is a or b or whatever.
    df["street_number2"] = (
        df["street_number"]
        .str.replace(r"^.*(?=\b[0-9]+$)", "", regex=True)
        .str.replace(r"[^\d]", "", regex=True)
    )

    # This stripped out versionof street name is used several times throughout the notebook
    df["street_name2"] = (
        df.loc[:, "street_name"]
        .str.replace(r"'", "", regex=True)
        .str.replace(r"s(s)?(?=\s)", "", regex=True)
        .str.replace(r"\s", "", regex=True)
    )

    return df

def forward_fill_by_title(df):
    """Forward fill missing geography columns within a title_number
    This function is used by enhance_ocod_with_gazetteers and is not intended for general use.
    """
    
    geog_cols = ['oa11cd', 'msoa11cd', 'lad11cd', 'match_prob']
    
    # Sort by title_number and geog_match (building matches first)
    df_sorted = df.sort_values(['title_number', 'geog_match'], na_position='last')
    
    # Forward fill within each title_number group
    df_sorted[geog_cols] = (
        df_sorted.groupby('title_number')[geog_cols]
        .ffill()
    )
    
    # Also forward fill geog_match for newly filled rows
    df_sorted['geog_match'] = (
        df_sorted.groupby('title_number')['geog_match']
        .ffill()
    )
    
    return df_sorted.sort_index()  # Restore original order


def enhance_ocod_with_gazetteers(pre_process_ocod, building_gazetteer, district_gazetteer, street_gazetteer):
    """
    Enhance OCOD data by adding missing geographic codes using gazetteers.
    
    This function fills in missing LAD codes using district information, and missing
    LSOA codes using building and street gazetteers where appropriate.
    
    Parameters
    ----------
    pre_process_ocod : pandas.DataFrame
        The OCOD dataset that may contain missing geographic codes
    building_gazetteer : pandas.DataFrame
        Building gazetteer with columns: building_name, oa11cd, lsoa11cd, msoa11cd, lad11cd, fraction
    district_gazetteer : pandas.DataFrame
        District gazetteer with columns: district, lad11cd
    street_gazetteer : pandas.DataFrame
        Street gazetteer with columns: street_name2, lsoa11cd, oa11cd, msoa11cd, lad11cd, fraction
        
    Returns
    -------
    pandas.DataFrame
        Enhanced OCOD dataset with missing geographic codes filled where possible.
        Includes 'match_prob' column with fraction values and 'geog_match' categorical column
        indicating whether match was from 'building' or 'street' gazetteer.
        
    Notes
    -----
    - First adds missing LAD codes using district gazetteer
    - Then adds missing LSOA codes using building gazetteer (where building_name and lad11cd match)
    - Finally adds missing LSOA codes using street gazetteer (where street_name and lad11cd match)
    - Only fills missing values, does not overwrite existing ones
    - Geographic codes are added hierarchically (LAD first, then LSOA/OA/MSOA)
    - Performs case-insensitive matching by converting to lowercase
    - match_prob and geog_match columns track the source and quality of gazetteer matches
        
    Examples
    --------
    >>> enhanced_ocod = enhance_ocod_with_gazetteers(pre_process_ocod, 
    ...                                             building_gaz, 
    ...                                             district_gaz, 
    ...                                             street_gaz)
    """
    
    # Create a copy to avoid modifying the original
    enhanced_ocod = pre_process_ocod.copy()
    
    # Initialize new columns
    enhanced_ocod['match_prob'] = float('nan')
    enhanced_ocod['geog_match'] = None
    
    # Create lowercase versions of gazetteers for matching
    building_gazetteer_lower = building_gazetteer.copy()
    building_gazetteer_lower['building_name'] = building_gazetteer_lower['building_name'].str.lower()
    
    street_gazetteer_lower = street_gazetteer.copy()
    street_gazetteer_lower['street_name2'] = street_gazetteer_lower['street_name2'].str.lower()
    
    # Step 1: Add missing LAD codes using district gazetteer
    missing_lad_mask = enhanced_ocod['lad11cd'].isna()
    
    if missing_lad_mask.sum() > 0:
        missing_lad_data = enhanced_ocod[missing_lad_mask].copy()
        merged_districts = missing_lad_data.merge(
            district_gazetteer[['district', 'lad11cd']], 
            on='district', 
            how='left', 
            suffixes=('', '_new')
        )
        enhanced_ocod.loc[missing_lad_mask, 'lad11cd'] = merged_districts['lad11cd_new'].values
    
    # Step 2: Add missing LSOA codes using building gazetteer
    missing_lsoa_mask = enhanced_ocod['lsoa11cd'].isna()
    has_building_and_lad = (enhanced_ocod['building_name'].notna() & 
                           enhanced_ocod['lad11cd'].notna())
    building_fillable_mask = missing_lsoa_mask & has_building_and_lad
    
    if building_fillable_mask.sum() > 0:
        building_data = enhanced_ocod[building_fillable_mask].copy()
        merged_buildings = building_data.merge(
            building_gazetteer_lower[['building_name', 'lad11cd', 'oa11cd', 'lsoa11cd', 'msoa11cd', 'fraction']], 
            on=['building_name', 'lad11cd'], 
            how='left', 
            suffixes=('', '_new')
        )
        
        # Update geographic codes and tracking columns where matches found
        match_found_mask = merged_buildings['lsoa11cd_new'].notna()
        if match_found_mask.sum() > 0:
            building_indices = enhanced_ocod.index[building_fillable_mask][match_found_mask]
            
            for col in ['oa11cd', 'lsoa11cd', 'msoa11cd']:
                enhanced_ocod.loc[building_indices, col] = merged_buildings.loc[match_found_mask, f'{col}_new'].values
            
            enhanced_ocod.loc[building_indices, 'match_prob'] = merged_buildings.loc[match_found_mask, 'fraction'].values
            enhanced_ocod.loc[building_indices, 'geog_match'] = 'building'
    
    # Step 3: Add missing LSOA codes using street gazetteer
    missing_lsoa_mask = enhanced_ocod['lsoa11cd'].isna()
    has_street_and_lad = (enhanced_ocod['street_name2'].notna() & 
                         enhanced_ocod['lad11cd'].notna())
    street_fillable_mask = missing_lsoa_mask & has_street_and_lad
    
    # Street matches do not fill in the oa as they are likely to cross these.
    if street_fillable_mask.sum() > 0:
        street_data = enhanced_ocod[street_fillable_mask].copy()
        merged_streets = street_data.merge(
            street_gazetteer_lower[['street_name2', 'lad11cd', 'lsoa11cd', 'msoa11cd', 'fraction']], 
            on=['street_name2', 'lad11cd'], 
            how='left', 
            suffixes=('', '_new')
        )
        
        # Update geographic codes and tracking columns where matches found
        match_found_mask = merged_streets['lsoa11cd_new'].notna()
        if match_found_mask.sum() > 0:
            street_indices = enhanced_ocod.index[street_fillable_mask][match_found_mask]
            
            for col in ['lsoa11cd', 'msoa11cd']:
                enhanced_ocod.loc[street_indices, col] = merged_streets.loc[match_found_mask, f'{col}_new'].values
            
            enhanced_ocod.loc[street_indices, 'match_prob'] = merged_streets.loc[match_found_mask, 'fraction'].values
            enhanced_ocod.loc[street_indices, 'geog_match'] = 'street'

    enhanced_ocod = forward_fill_by_title(enhanced_ocod)

    # Convert geog_match to categorical for memory efficiency
    enhanced_ocod['geog_match'] = enhanced_ocod['geog_match'].astype('category')
    
    return enhanced_ocod


def add_business_matches(df, voa_businesses):
    """Add boolean columns indicating business address matches.
    
    Performs address matching between a main dataframe and VOA business data
    by comparing building names, street names, and street numbers within the
    same Local Authority District (LAD). Creates three new boolean columns
    to indicate successful matches at different address components.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Main dataframe containing address components (building_name, 
        street_name2, street_number, lad11cd) to match against.
    voa_businesses : pandas.DataFrame
        VOA business dataframe containing reference address data with
        the same address component columns for matching.
    
    Returns
    -------
    pandas.DataFrame
        Copy of input dataframe with three additional boolean columns:
        - building_match: True if building_name + lad11cd found in voa_businesses
        - street_match: True if street_name2 + lad11cd found in voa_businesses  
        - number_match: True if street_number + street_name2 + lad11cd found
          in voa_businesses
    
    Notes
    -----
    - Only performs matching where required columns are not null
    - Uses left joins to preserve all original records
    - Returns a copy of the original dataframe, leaving input unchanged
    - Matching is exact and case-sensitive
    """
    
    def add_match_column(enhanced_df, match_columns, result_column):
        """Helper function to add a single match column."""
        # Check for non-null values in all required columns
        has_required_data = enhanced_df[match_columns].notna().all(axis=1)
        
        if has_required_data.sum() > 0:
            # Get reference data and merge
            reference_data = voa_businesses[match_columns].dropna().drop_duplicates()
            subset_data = enhanced_df[has_required_data]
            
            merged = subset_data.merge(
                reference_data,
                on=match_columns,
                how='left',
                indicator='_merge_indicator'
            )
            
            # Update the result column where matches were found
            match_found_mask = merged['_merge_indicator'] == 'both'
            if match_found_mask.sum() > 0:
                matched_indices = enhanced_df.index[has_required_data][match_found_mask]
                enhanced_df.loc[matched_indices, result_column] = True
    
    # Create a copy and initialize columns
    enhanced_df = df.copy()
    
    # Define the matching configurations
    match_configs = [
        (['oa11cd'], 'oa_match'),
        (['lsoa11cd'], 'lsoa_match'),
        (['building_name', 'lad11cd'], 'building_match'),
        (['street_name2', 'lad11cd'], 'street_match'),
        (['street_number', 'street_name2', 'lad11cd'], 'number_match')
    ]
    
    # Initialize all result columns
    for _, result_column in match_configs:
        enhanced_df[result_column] = False
    
    # Apply matching for each configuration
    for match_columns, result_column in match_configs:
        add_match_column(enhanced_df, match_columns, result_column)
    
    return enhanced_df

def property_class(df, rules, include_rule_name=True):
    """
    Create a hierarchical classification of property types using configurable rules.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing property data to be classified.
    rules : list of dict
        List of classification rules. Each rule dict should contain:
        - 'rule_name': string description
        - 'condition': lambda function that takes df and returns boolean Series
        - 'class': string of class name to assign
        - 'comments': optional comments for documentation
    include_rule_name : bool, default True
        If True, adds a 'matched_rule' column showing which rule was applied
        (stored as categorical for memory efficiency)
    
    Returns
    -------
    pandas.DataFrame
        Input dataframe with additional 'class' column containing the
        property classification. If include_rule_name=True, also includes
        'matched_rule' column with the rule name as categorical data.
        
    Raises
    ------
    ValueError
        If rule names are not unique when include_rule_name=True
    """

    df = df.copy()
    
    # Check for unique rule names if we're including them
    if include_rule_name:
        rule_names = [rule['rule_name'] for rule in rules]
        if len(rule_names) != len(set(rule_names)):
            duplicates = [name for name in rule_names if rule_names.count(name) > 1]
            raise ValueError(f"Rule names must be unique. Duplicates found: {set(duplicates)}")
    
    conditions = []
    choices = []
    rule_names = []
    
    for rule in rules:
        try:
            condition = rule['condition'](df)
            conditions.append(condition)
            choices.append(rule['class'])
            if include_rule_name:
                rule_names.append(rule['rule_name'])
        except Exception as e:
            print(f"Warning: Rule '{rule['rule_name']}' failed: {e}")
            continue
    
    # Handle case where no valid conditions exist
    if len(conditions) == 0:
        df["class"] = "unknown"
        if include_rule_name:
            df["matched_rule"] = pd.Categorical(["none"] * len(df), categories=["none"])
    else:
        df["class"] = np.select(conditions, choices, default="unknown")
        if include_rule_name:
            # Create categorical with all possible rule names plus "none" as categories
            all_categories = rule_names + ["none"]
            matched_rules = np.select(conditions, rule_names, default="none")
            df["matched_rule"] = pd.Categorical(matched_rules, categories=all_categories)
    
    return df


def get_default_property_rules():
    """Defines the default property classification rules."""
    

    return [
        {
            'rule_name': 'Land plots at start of address',
            'condition': lambda df: df["property_address"].str.contains(r"^(?:land|plot)", case=False),
            'class': 'land',
            'comments': 'Identifies properties starting with "land" or "plot"'
        },
        {
            'rule_name': 'Parking and garage spaces',
            'condition': lambda df: df["property_address"].str.contains(
                r"^(?:[a-z\s]*)(?:garage|parking(?:\s)?space|parking space|car park(?:ing)?)",
                case=False
            ),
            'class': 'carpark',
            'comments': 'Identifies parking spaces, garages, and car parks'
        },
        {
            'rule_name': 'Airspace properties',
            'condition': lambda df: df["property_address"].str.contains(
                r"^(?:the airspace|airspace|air space)", case=False
            ),
            'class': 'airspace',
            'comments': 'Properties related to airspace rights'
        },
        {
            'rule_name': 'Penthouse/Flat/Apartment',
            'condition': lambda df: df["property_address"].str.contains(
                r"penthouse|flat|apartment", case=False
            ),
            'class': 'residential',
            'comments': 'Clear residential indicators like penthouse, flat, apartment'
        },
        {
            'rule_name': 'Business properties by address keywords',
            'condition': lambda df: df["property_address"].str.contains(
                r"cinema|hotel|office|centre|\bpub|holiday(?:\s)?inn|travel lodge|travelodge|medical|business|cafe|^shop| shop|service|logistics|building supplies|restaurant|home|suite|^store(?:s|rooms?)\b|^storage\b|company|ltd|limited|plc|retail|leisure|industrial|hall of|trading|commercial|technology|works|club,|advertising|school|church|(?:^room)|vault",
                case=False
            ),
            'class': 'business',
            'comments': 'Business properties identified by commercial keywords in address'
        },
        {
            'rule_name': 'Land with words before it',
            'condition': lambda df: df["property_address"].str.contains(
                r"^[a-z\s']+\b(?:land(?:s)?|plot(?:s)?)\b", case=False
            ),
            'class': 'land',
            'comments': 'Land or plots that have descriptive words before them'
        },
        {
            'rule_name': 'Pubs by building name',
            'condition': lambda df: df["building_name"].str.contains(
                r"\binn$|public house|^the\s\w+\sand\s\w+|(?:tavern$)|^the\s+(?:red|white|black|blue|green|golden|silver|old|royal)\s+\w+$",
                case=False,
                na=False
            ),
            'class': 'business',
            'comments': 'Pubs identified by building name patterns'
        },
        {
            'rule_name': 'Named building crossing multiple addresses',
            'condition': lambda df: (
                df['building_name'].notna() & 
                df['street_number'].str.contains(xx_to_yy_regex, na=False) & 
                df['unit_id'].isna()
            ),
            'class': 'business',
            'comments': 'Named building which crosses multiple street addresses but has no sub-units'
        },
        {
            'rule_name': 'Addresses starting with part/parts',
            'condition': lambda df: df["property_address"].str.contains(r"^(?:part|parts)\b", case=False),
            'class': 'business',
            'comments': 'Addresses starting with "part" or "parts" are typically businesses'
        },
        {
            'rule_name': 'Units without building name',
            'condition': lambda df: (
                df["property_address"].str.contains(r"\b(?:unit|units)\b", case=False) & 
                df["building_name"].isna()
            ),
            'class': 'business',
            'comments': 'Properties with "unit" or "units" but no building name are businesses'
        },
        {
            'rule_name': 'Building match indicator',
            'condition': lambda df: df["building_match"] == True,
            'class': 'business',
            'comments': 'Properties where a business building was matched'
        },
        {
            'rule_name': 'Number match indicator',
            'condition': lambda df: df["number_match"] == True,
            'class': 'business',
            'comments': 'Properties where the street and number of a business was matched'
        },
        {
            'rule_name': 'No street match',
            'condition': lambda df: df["street_match"] == False,
            'class': 'residential',
            'comments': 'If there is no business on the street then it must be residential'
        },
            {
            'rule_name': 'oa match',
            'condition': lambda df: df["oa_match"] == False,
            'class': 'residential',
            'comments': 'If there is no business in the oa then it must be residential'
        },
                    {
            'rule_name': 'lsoa match',
            'condition': lambda df: df["lsoa_match"] == False,
            'class': 'residential',
            'comments': 'If there is no business in the lsoa then it must be residential'
        }


    ]


def fill_unknown_classes_by_group(df, group_col='title_number', class_col='class', 
                                 unknown_value='unknown', exclude_classes=None, 
                                 condition_col='is_multi', unique_id_col='unique_id'):
    """
    Fill unknown class values using known class values from the same group.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    group_col : str, default 'title_number'
        Column to group by when filling unknown values
    class_col : str, default 'class'
        Column containing the classification values
    unknown_value : str, default 'unknown'
        Value representing unknown/missing classifications
    exclude_classes : list, default None
        List of class values to exclude from being used as fill sources
    condition_col : str, default 'within_larger_title'
        Additional condition column that must be True for filling
    unique_id_col : str, default 'unique_id'
        Column containing unique identifiers for each row
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with unknown values filled where possible
    """
    if exclude_classes is None:
        exclude_classes = ["unknown", "airspace", "carpark"]
    
    # Step 1: Create lookup table of known classifications by group
    lookup_conditions = (
        ~df[class_col].isin(exclude_classes) & 
        df[condition_col].fillna(False)
    )
    
    lookup_table = (
        df[lookup_conditions]
        .groupby([group_col, class_col])
        .size()
        .reset_index()[[group_col, class_col]]
        .drop_duplicates()
    )
    
    # Step 2: Find records that can be filled
    fill_conditions = (
        df[group_col].isin(lookup_table[group_col]) & 
        (df[class_col] == unknown_value) & 
        df[condition_col].fillna(False)
    )
    
    records_to_fill = df[fill_conditions].copy()
    
    if records_to_fill.empty:
        return df  # No records to fill
    
    # Step 3: Fill unknown classifications
    records_to_fill = records_to_fill.drop(class_col, axis=1)
    records_to_fill = records_to_fill.merge(lookup_table, how='left', on=group_col)
    
    # Step 4: Combine with unchanged records
    unchanged_records = df[~df[unique_id_col].isin(records_to_fill[unique_id_col])]
    result = pd.concat([records_to_fill, unchanged_records], ignore_index=True)
    
    return result

def evaluate_classification_predictions(gt_df, pred_df):
    """
    Evaluate classification predictions and return classification report.

    Args:
        gt_df: DataFrame with columns ['title_number', 'class'] containing ground truth
        pred_df: DataFrame with columns ['title_number', 'class'] containing predictions

    Returns:
        str: Classification report
    """
    
    # Join the dataframes on title_number
    merged_df = gt_df.merge(pred_df, on='title_number', suffixes=('_true', '_pred'))
    
    # Check if we lost any examples in the merge
    if len(merged_df) != len(gt_df):
        print(f"Warning: Ground truth has {len(gt_df)} examples, But there are {len(merged_df)} matched predictions")
    
    # Extract true and predicted labels
    y_true = merged_df['class_true'].tolist()
    y_pred = merged_df['class_pred'].tolist()
    
    # Calculate metrics (using macro average to match typical classification evaluation)
    overall_f1 = f1_score(y_true, y_pred, average='macro')
    overall_precision = precision_score(y_true, y_pred, average='macro')
    overall_recall = recall_score(y_true, y_pred, average='macro')

    # Print results
    print(f"Overall F1: {overall_f1:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print("\nPer-class results:")
    print(classification_report(y_true, y_pred))

    class_dict = classification_report(y_true, y_pred, output_dict=True )
    df = pd.DataFrame(class_dict).transpose()

    return df

def drop_non_residential_duplicates(df, class_col='class'):
    """
    Drop duplicates based on title_number and class columns for all rows 
    that are not residential, while keeping all residential rows.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    class_col : str, default 'class'
        Name of the class column
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with duplicates removed from non-residential rows
    """
    
    # Separate residential and non-residential rows
    residential_rows = df[df[class_col] == 'residential']
    non_residential_rows = df[df[class_col] != 'residential']
    
    # Drop duplicates on non-residential rows
    non_residential_deduped = non_residential_rows.drop_duplicates(subset=['title_number', class_col])
    
    # Combine back together
    result_df = pd.concat([residential_rows, non_residential_deduped], ignore_index=True)
    
    return result_df


def get_performance_by_rule(df):
    """Calculate performance metrics for property classification rule separately"""
    
    results = []
    
    for rule in df['matched_rule'].unique():
        rule_data = df[df['matched_rule'] == rule]
        
        # Skip if rule has no predictions
        if len(rule_data) == 0:
            continue
            
        accuracy = accuracy_score(rule_data['class_true'], rule_data['class_pred'])
        precision = precision_score(rule_data['class_true'], rule_data['class_pred'], average='weighted', zero_division=0)
        recall = recall_score(rule_data['class_true'], rule_data['class_pred'], average='weighted', zero_division=0)
        f1 = f1_score(rule_data['class_true'], rule_data['class_pred'], average='weighted', zero_division=0)
        
        results.append({
            'rule': rule,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sample_count': len(rule_data)
        })
    
    return pd.DataFrame(results).sort_values('f1_score', ascending=False)

