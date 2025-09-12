# the functions used by the empty homes project python notebooks
import io
import re
import time
import zipfile
import pandas as pd
import numpy as np
from .labelling.ner_regex import xx_to_yy_regex


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
    Performance-optimized version focusing on string operations
    """

    def find_largest_file(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            largest_file = max(zip_ref.infolist(), key=lambda x: x.file_size)
            return largest_file.filename

    VOA_headers_needed = [
        "Primary And Secondary Description Code",  # 4
        "Primary Description Text",  # 5
        "Unique Address Reference Number UARN",  # 6
        "Full Property Identifier",  # 7
        "Firms Name",  # 8
        "Number Or Name",  # 9
        "Street",  # 10
        "Town",  # 11
        "Postal District",  # 12
        "County",  # 13
        "Postcode",  # 14
    ]

    VOA_headers = [x.lower().replace(" ", "_") for x in VOA_headers_needed]
    usecols = list(range(4, 15))

    # Read data (same as before)
    if file_path.lower().endswith(".zip"):
        largest_file = find_largest_file(file_path)
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            with zip_ref.open(largest_file) as csv_file:
                voa_businesses = pd.read_csv(
                    csv_file,
                    sep="*",
                    encoding_errors="ignore",
                    header=None,
                    names=VOA_headers,
                    usecols=usecols,
                    index_col=False,
                    dtype=str,
                )
    else:
        voa_businesses = pd.read_csv(
            file_path,
            sep="*",
            encoding_errors="ignore",
            header=None,
            names=VOA_headers,
            usecols=usecols,
            index_col=False,
            dtype=str,
        )

    # Early filtering BEFORE expensive string operations
    # Filter out unwanted data first to reduce processing volume
    print(f"Initial rows: {len(voa_businesses)}")

    # Use faster operations for filtering
    advertising_mask = voa_businesses["primary_description_text"].str.contains(
        "ADVERTISING", na=False
    )
    voa_businesses = voa_businesses[~advertising_mask]

    excluded_codes = {"C0", "CP", "CP1", "CX", "MX"}  # Use set for faster lookup
    parking_mask = voa_businesses["primary_and_secondary_description_code"].isin(
        excluded_codes
    )
    voa_businesses = voa_businesses[~parking_mask]

    voa_businesses["postcode"] = voa_businesses["postcode"].str.lower()
    voa_businesses["street"] = voa_businesses["street"].str.lower()

    # Chain regex operations to reduce passes through data
    voa_businesses["postcode2"] = voa_businesses["postcode"].str.replace(
        r"\s", "", regex=True
    )

    voa_businesses["street_name2"] = (
        voa_businesses["street"]
        .str.replace(r"'", "", regex=True)
        .str.replace(r"s(s)?(?=\s)", "", regex=True)
        .str.replace(r"\s", "", regex=True)
    )

    # First merge to reduce dataset size
    voa_businesses = voa_businesses.merge(
        postcode_district_lookup, left_on="postcode2", right_on="postcode2", how="inner"
    )

    # Now do expensive operations on smaller dataset
    voa_businesses = clean_street_numbers(
        voa_businesses, original_column="number_or_name"
    )

    westfield_mask = voa_businesses["full_property_identifier"].str.contains(
        "WESTFIELD SHOPPING CENTRE", na=False
    )
    if westfield_mask.any():  # Only process if there are matches
        voa_businesses.loc[westfield_mask, "street_number"] = np.nan

    if "street_number" in voa_businesses.columns:
        phone_mask = voa_businesses["street_number"].str.contains(
            r"-\d+-", regex=True, na=False
        )
        if phone_mask.any():  # Only process if there are matches
            voa_businesses.loc[phone_mask, "street_number"] = np.nan

    # OPTIMIZATION 6: Optimize building_name extraction
    voa_businesses["building_name"] = (
        voa_businesses["number_or_name"]
        .str.lower()
        .str.extract(
            r"((?<!\S)(?:(?!\b(?:\)|\(|r\/o|floor|floors|pt|and|annexe|room|gf|south|north|east|west|at|on|in|of|adjoining|adj|basement|bsmt|fl|flr|flrs|wing)\b)[^\n\d])*? house\b)",
            expand=False,
        )
    )

    return voa_businesses

def counts_of_businesses_per_oa_lsoa(ocod_data, voa_businesses):
    """
    This function allows areas with no  businesses to automatically exclude business from the classification
    """
    # Create a dataframe that contains the counts of businesses per OA
    postcode_counts_voa = (
        voa_businesses.groupby("oa11cd").size().reset_index(name="business_counts")
    )
    ocod_data = pd.merge(ocod_data, postcode_counts_voa, on="oa11cd", how="left")
    ocod_data["business_counts"] = ocod_data["business_counts"].fillna(0)

    # do the same for lsoa
    lsoa_counts_voa = (
        voa_businesses.groupby("lsoa11cd")
        .size()
        .reset_index(name="lsoa_business_counts")
    )
    ocod_data = pd.merge(ocod_data, lsoa_counts_voa, on="lsoa11cd", how="left")
    ocod_data["lsoa_business_counts"] = ocod_data["lsoa_business_counts"].fillna(0)

    return ocod_data




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
        df["postcode"].str.lower().str.replace("\s", "", regex=True)
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
    
    if street_fillable_mask.sum() > 0:
        street_data = enhanced_ocod[street_fillable_mask].copy()
        merged_streets = street_data.merge(
            street_gazetteer_lower[['street_name2', 'lad11cd', 'oa11cd', 'lsoa11cd', 'msoa11cd', 'fraction']], 
            on=['street_name2', 'lad11cd'], 
            how='left', 
            suffixes=('', '_new')
        )
        
        # Update geographic codes and tracking columns where matches found
        match_found_mask = merged_streets['lsoa11cd_new'].notna()
        if match_found_mask.sum() > 0:
            street_indices = enhanced_ocod.index[street_fillable_mask][match_found_mask]
            
            for col in ['oa11cd', 'lsoa11cd', 'msoa11cd']:
                enhanced_ocod.loc[street_indices, col] = merged_streets.loc[match_found_mask, f'{col}_new'].values
            
            enhanced_ocod.loc[street_indices, 'match_prob'] = merged_streets.loc[match_found_mask, 'fraction'].values
            enhanced_ocod.loc[street_indices, 'geog_match'] = 'street'
    
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
    
    # Create a copy to avoid modifying the original
    enhanced_df = df.copy()
    
    # Initialize the new columns
    enhanced_df['building_match'] = False
    enhanced_df['street_match'] = False
    enhanced_df['number_match'] = False
    
    # Building match
    has_building_and_lad = (enhanced_df['building_name'].notna() & 
                           enhanced_df['lad11cd'].notna())
    
    if has_building_and_lad.sum() > 0:
        building_data = enhanced_df[has_building_and_lad].copy()
        building_match_df = voa_businesses[['building_name', 'lad11cd']].dropna().drop_duplicates()
        
        merged_buildings = building_data.merge(
            building_match_df, 
            on=['building_name', 'lad11cd'], 
            how='left', 
            indicator='_building_merge'
        )
        
        match_found_mask = merged_buildings['_building_merge'] == 'both'
        if match_found_mask.sum() > 0:
            building_indices = enhanced_df.index[has_building_and_lad][match_found_mask]
            enhanced_df.loc[building_indices, 'building_match'] = True
    
    # Street match
    has_street_and_lad = (enhanced_df['street_name2'].notna() & 
                         enhanced_df['lad11cd'].notna())
    
    if has_street_and_lad.sum() > 0:
        street_data = enhanced_df[has_street_and_lad].copy()
        street_match_df = voa_businesses[['street_name2', 'lad11cd']].dropna().drop_duplicates()
        
        merged_streets = street_data.merge(
            street_match_df, 
            on=['street_name2', 'lad11cd'], 
            how='left', 
            indicator='_street_merge'
        )
        
        match_found_mask = merged_streets['_street_merge'] == 'both'
        if match_found_mask.sum() > 0:
            street_indices = enhanced_df.index[has_street_and_lad][match_found_mask]
            enhanced_df.loc[street_indices, 'street_match'] = True
    
    # Street number match
    has_number_street_and_lad = (enhanced_df['street_number'].notna() & 
                                enhanced_df['street_name2'].notna() & 
                                enhanced_df['lad11cd'].notna())
    
    if has_number_street_and_lad.sum() > 0:
        number_data = enhanced_df[has_number_street_and_lad].copy()
        number_match_df = voa_businesses[['street_number', 'street_name2', 'lad11cd']].dropna().drop_duplicates()
        
        merged_numbers = number_data.merge(
            number_match_df, 
            on=['street_number', 'street_name2', 'lad11cd'], 
            how='left', 
            indicator='_number_merge'
        )
        
        match_found_mask = merged_numbers['_number_merge'] == 'both'
        if match_found_mask.sum() > 0:
            number_indices = enhanced_df.index[has_number_street_and_lad][match_found_mask]
            enhanced_df.loc[number_indices, 'number_match'] = True
    
    return enhanced_df


def property_class(df):
    """
    Create a hierarchical classification of property types using logical rules.
    
    Applies a series of pattern-matching conditions to classify properties into
    categories such as land, carpark, airspace, residential, business, or unknown
    based on property address patterns, building names, and matching indicators.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing property data to be classified.
    
    Returns
    -------
    pandas.DataFrame
        Input dataframe with additional 'class' column containing the
        property classification.
    """

    df["class"] = np.select(
        [
            df["property_address"].str.contains(r"^(?:land|plot)", case=False),
            df["property_address"].str.contains(
                r"^(?:[a-z\s]*)(?:garage|parking(?:\s)?space|parking space|car park(?:ing)?)",
                case=False,
            ),
            df["property_address"].str.contains(
                r"^(?:the airspace|airspace)", case=False
            ),
            df["property_address"].str.contains(
                r"penthouse|flat|apartment", case=False
            ),
            ~df["street_match"],# If there is no business on the street then it must be a residential
            df["property_address"].str.contains(
                r"cinema|hotel|office|centre|\bpub|holiday(?:\s)?inn|travel lodge|travelodge|medical|business|cafe|^shop| shop|service|logistics|building supplies|restaurant|home|^store(?:s)?\b|^storage\b|company|ltd|limited|plc|retail|leisure|industrial|hall of|trading|commercial|technology|works|club,|advertising|school|church|(?:^room)",
                case=False,
            ),
            df["property_address"].str.contains(
                r"^[a-z\s']+\b(?:land(?:s)?|plot(?:s)?)\b", case=False
            ),  # land with words before it
            df["building_name"].str.contains(
                r"\binn$|public house|^the\s\w+\sand\s\w+|(?:tavern$)",
                case=False,
                na=False,
            ),  # pubs in various guises
            df['building_name'].notna() & df['street_number'].str.contains(xx_to_yy_regex, na=False)  & df['unit_id'].isna(), # A named building which crosses multiple street addresses but doesn't have sub-units is a business
            df["building_match"],  # a business building was matched
            df["number_match"], # The street and number of a business was matched
         ],
        [
            "land",
            "carpark",
            "airspace",
            "residential",
            "residential",
            "business",
            "land",
            'business',
            "business",
            "business",
            "business",
        ],
        default="unknown",
    )

    return df

def property_class_no_match(df):
    """
    Perform additional property classification for previously unmatched properties.
    
    A more permissive classification approach that treats unmatched addresses
    with street numbers as residential properties when they don't match known
    business addresses but do match street patterns. This represents an upper
    bound estimate for residential property counts.
    
    Note: This function should be run after property_class() has been applied.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe that has already been processed by property_class().
    
    Returns
    -------
    pandas.DataFrame
        Input dataframe with additional 'class_no_match' column containing
        the updated property classification.
    """
    df["class_no_match"] = np.where(
        (df["class"] == "unknown") & 
        (~df["number_match"]) & 
        df["street_match"] & 
        df['street_number'].notna(),
        "residential",
        df["class"]
    )
    
    return df