import pandas as pd
import re
import numpy as np
import time
import zipfile
from typing import Optional, List, Callable, Dict, Optional, Any
import io
from .labelling.ner_regex import xx_to_yy_regex

# This  module is supposed to contain all the relevant functions for parsing the labeled json file


def load_postcode_district_lookup(file_path, target_post_area=None, column_config=None):
    """
    Load the ONS postcode district lookup table from a ZIP file.

    Loads the ONSPD (Office for National Statistics Postcode Directory) and reduces
    it to relevant columns to save memory. Filters for English and Welsh postcodes.
    Converts geographic code columns to categorical for optimal memory usage. 
    Automatically detects pre/post-2025 format.

    Args:
        file_path (str): Path to the ZIP file containing the ONSPD data.
        target_post_area (str, optional): Specific CSV file within the ZIP to load.
            If None, automatically finds the appropriate file.
        column_config (dict, optional): Configuration dictionary mapping source column names
            to their properties. Each key is a source column name, and each value is a dict with:
                - 'rename' (str): Target column name after loading
                - 'dtype' (str): Pandas dtype for the column
                - 'drop' (bool, optional): Whether to drop this column after filtering. Defaults to False.
            If None, uses default ONSPD configuration.

    Returns:
        pd.DataFrame: Processed postcode district lookup table with mixed dtypes optimized for memory.

    Example:
        # Use default configuration
        df = load_postcode_district_lookup('ONSPD.zip')
        
        # Custom configuration
        custom_config = {
            "pcds": {"rename": "postcode", "dtype": "string"},
            "oslaua": {"rename": "local_authority", "dtype": "category"},
            "ctry": {"rename": "country", "dtype": "category", "drop": True}
        }
        df = load_postcode_district_lookup('ONSPD.zip', column_config=custom_config)
    """
    # Default configuration - define both old and new formats
    if column_config is None:
        # Old format (pre-2025)
        old_format_config = {
            "pcds": {"rename": "postcode2", "dtype": "string"},
            "oslaua": {"rename": "lad11cd", "dtype": "category"},
            "oa11": {"rename": "oa11cd", "dtype": "category"},
            "lsoa11": {"rename": "lsoa11cd", "dtype": "category"},
            "msoa11": {"rename": "msoa11cd", "dtype": "category"},
            "ctry": {"rename": "ctry", "dtype": "category", "drop": True}
        }

        # New format (2025+)
        new_format_config = {
            "pcds": {"rename": "postcode2", "dtype": "string"},
            "lad25cd": {"rename": "lad11cd", "dtype": "category"},
            "oa11cd": {"rename": "oa11cd", "dtype": "category"},
            "lsoa11cd": {"rename": "lsoa11cd", "dtype": "category"},
            "msoa11cd": {"rename": "msoa11cd", "dtype": "category"},
            "ctry25cd": {"rename": "ctry", "dtype": "category", "drop": True}
        }

        # Auto-detect format by reading column headers
        with zipfile.ZipFile(file_path) as zf:
            # Find the CSV file
            if target_post_area is None:
                target_post_area = [
                    i for i in zf.namelist() if re.search(r"^Data/ONSPD.+csv$", i)
                ][0]

            # Read just the header to detect format
            with io.TextIOWrapper(zf.open(target_post_area), encoding="latin-1") as f:
                header = pd.read_csv(f, nrows=0)
                available_columns = set(header.columns)

            # Check which format matches
            old_format_keys = set(old_format_config.keys())
            new_format_keys = set(new_format_config.keys())

            if old_format_keys.issubset(available_columns):
                column_config = old_format_config
                print("Pre-2025 ONSPD format detected")
            elif new_format_keys.issubset(available_columns):
                column_config = new_format_config
                print("Post-2025 ONSPD format detected")
            else:
                # Neither format matches - fail with clear error
                old_matches = len(old_format_keys.intersection(available_columns))
                new_matches = len(new_format_keys.intersection(available_columns))
                old_missing = old_format_keys - available_columns
                new_missing = new_format_keys - available_columns

                error_msg = (
                    f"ONSPD format not recognized. Neither old nor new format column sets found.\n\n"
                    f"Pre-2025 formar- matched {old_matches}/{len(old_format_keys)} columns:\n"
                    f"  Expected: {sorted(old_format_keys)}\n"
                    f"  Missing: {sorted(old_missing)}\n\n"
                    f"Post-2025 - matched {new_matches}/{len(new_format_keys)} columns:\n"
                    f"  Expected: {sorted(new_format_keys)}\n"
                    f"  Missing: {sorted(new_missing)}\n\n"
                    f"Available columns in file: {sorted(available_columns)}\n\n"
                    f"Please check the ONSPD file format or update the column configuration."
                )
                raise ValueError(error_msg)

    # Extract components from configuration
    columns_to_read = list(column_config.keys())
    dtype_dict = {col: config["dtype"] for col, config in column_config.items()}
    rename_mapping = {col: config["rename"] for col, config in column_config.items()}
    columns_to_drop = [config["rename"] for col, config in column_config.items()
                       if config.get("drop", False)]

    with zipfile.ZipFile(file_path) as zf:
        # If no target file specified, find it automatically
        if target_post_area is None:
            target_post_area = [
                i for i in zf.namelist() if re.search(r"^Data/ONSPD.+csv$", i)
            ][0]

        with io.TextIOWrapper(zf.open(target_post_area), encoding="latin-1") as f:
            postcode_district_lookup = pd.read_csv(
                f, 
                dtype=dtype_dict,
                usecols=columns_to_read, 
                low_memory=True  
            )

            # Filter for English and Welsh postcodes
            # Note: This assumes 'ctry' column exists and is renamed to something
            # Find the country column (original or renamed)
            country_col = None
            for source_col, config in column_config.items():
                if source_col == "ctry" or config["rename"] == "ctry":
                    country_col = source_col
                    break
            
            if country_col and country_col in postcode_district_lookup.columns:
                postcode_district_lookup = postcode_district_lookup[
                    (postcode_district_lookup[country_col] == "E92000001")
                    | (postcode_district_lookup[country_col] == "W92000004")
                ]

            # Rename columns
            postcode_district_lookup.rename(columns=rename_mapping, inplace=True)

            # Process postcodes - find the postcode column and transform it
            # Look for the renamed postcode column (default is 'postcode2')
            postcode_col = None
            for source_col, config in column_config.items():
                if source_col == "pcds":
                    postcode_col = config["rename"]
                    break
            
            if postcode_col and postcode_col in postcode_district_lookup.columns:
                postcode_district_lookup[postcode_col] = (
                    postcode_district_lookup[postcode_col]
                    .str.lower()
                    .str.replace(r"\s", r"", regex=True)
                )

            # Drop marked columns
            if columns_to_drop:
                postcode_district_lookup.drop(
                    [col for col in columns_to_drop if col in postcode_district_lookup.columns], 
                    axis=1, 
                    inplace=True
                )

            # Remove unused categories from all categorical columns
            for col in postcode_district_lookup.select_dtypes(include=['category']).columns:
                postcode_district_lookup[col] = postcode_district_lookup[col].cat.remove_unused_categories()

    return postcode_district_lookup


##
##
## Expanding the addresses
##

def needs_expansion(df, class_var = 'class'):

    """Identify rows that need expansion due to ranged street or unit identifiers.
    
    Identifies which rows in the dataframe need to be expanded to multiple rows
    because they have street or unit id's of the form "34-41", "10-20", etc.
    This is not the same as identifying whether a title contains multiple
    properties, and this operation is performed by "tag_multi_property".
    
    Args:
        df (pandas.DataFrame): Input dataframe to analyze.
        class_var (str, optional): Column name containing property class
            information. Defaults to 'class'.
    
    Returns:
        pandas.DataFrame: Copy of input dataframe with additional 'needs_expansion'
            boolean column indicating which rows need expansion.
    
    Note:
        Only residential properties are considered for expansion. The function
        checks for range patterns in 'unit_id' column, or 'street_number' 
        column when 'unit_id' is missing.
    """

    df = df.copy()
    
    df['number_filter'] = df['number_filter'].fillna('none')
    df['number_filter'] = df['number_filter'].astype(str)

    residential_mask = df[class_var] == 'residential'
    expansion_condition = (df['unit_id'].str.contains(xx_to_yy_regex, na=False) | 
                    (df['unit_id'].isna() & df['street_number'].str.contains(xx_to_yy_regex, na=False)))

    df['needs_expansion'] = np.where(residential_mask & expansion_condition, True, False)

    return df

def expand_multi_id(multi_id_string):
    # the function takes a string that is in the form '\d+(\s)?(-|to)(\s)?\d+'
    # and outputs a continguous list of numbers between the two numbers in the string
    multi_id_list = [int(x) for x in re.findall(r"\d+", multi_id_string)]
    # min and max has to be used becuase somtimes the numbers are in descending order 4-3... I don't know why someone would do that
    out = list(range(min(multi_id_list), max(multi_id_list) + 1))
    return out


def filter_contiguous_numbers(number_list, number_filter):
    # this function filters a list of contiguous house numbers/unit_id's to be even, odd, or unchanged
    # it takes as an argument a list of integers and a filter condition.
    # these values are contained in the label dictionary and reformated dataframe
    # The function ouputs the correct list of integers according to the filter condition

    if number_filter == "odd":
        out = [x for x in number_list if x % 2 == 1]
    elif number_filter == "even":
        out = [x for x in number_list if x % 2 == 0]
    else:
        out = number_list
    return out


def expand_dataframe_numbers_core(df, column_name, print_every=1000, min_count=1):
    """Expand number range formats in a dataframe column with performance monitoring.
    
    Processes each row in the dataframe to expand number ranges in the specified 
    column from 'xx-to-yy' format into individual entries. Includes timing 
    diagnostics and progress reporting.
    
    Args:
        df (pandas.DataFrame): Input dataframe to process.
        column_name (str): Name of the column containing number ranges to expand.
        print_every (int, optional): Print progress every nth iteration. Defaults to 1000.
        min_count (int, optional): Minimum count threshold for expansion logic. 
            Defaults to 1.
    
    Returns:
        pandas.DataFrame: Expanded dataframe with individual number entries and 
            timing performance metrics logged to console.
    
    Note:
        This function is typically called internally by expand_dataframe_numbers 
        rather than used directly.
    """
    # Handle empty dataframe case
    if df.shape[0] == 0:
        return df

    temp_list = []
    expand_time = 0
    filter_time = 0
    make_dataframe_time = 0

    for i in range(0, df.shape[0]):
        start_expand_time = time.time()
        numbers_list = expand_multi_id(df.loc[i][column_name])
        end_expand_time = time.time()

        numbers_list = filter_contiguous_numbers(
            numbers_list, df.loc[i]["number_filter"]
        )

        end_filter_time = time.time()

        dataframe_len = len(numbers_list)

        # This prevents large properties counting as several properties
        if dataframe_len > min_count:
            tmp = pd.concat(
                [df.iloc[i].to_frame().T] * dataframe_len, ignore_index=True
            )

            tmp[column_name] = numbers_list
        else:
            tmp = df.iloc[i].to_frame().T

        temp_list.append(tmp)
        end_make_dataframe_time = time.time()

        expand_time = expand_time + (end_expand_time - start_expand_time)
        filter_time = filter_time + (end_filter_time - end_expand_time)
        make_dataframe_time = make_dataframe_time + (
            end_make_dataframe_time - end_filter_time
        )

        if (i > 0) & (i % print_every == 0):
            print(
                "i=",
                i,
                " expand time,"
                + str(round(expand_time, 3))
                + " filter time"
                + str(round(filter_time, 3))
                + " make_dataframe_time "
                + str(round(make_dataframe_time, 3)),
            )

    # once all the lines have been expanded concatenate them into a single dataframe

    out = pd.concat(temp_list)
    # The data type coming into the function is a string as it is in the form xx-yy
    # It needs to return a string as well otherwise there will be a pandas columns of mixed types
    # ehich causes problems later on
    out.loc[:, column_name] = out.loc[:, column_name].astype(str)

    return out

def expand_dataframe_numbers(df, class_var = 'class', print_every=1000, min_count=1):
    """Expand number ranges in dataframe based on unit_id availability and expansion flags.
    
    Conditionally expands number ranges in either unit_id or street_number columns 
    depending on data availability and expansion requirements. Processes rows marked 
    for expansion while preserving non-expansion rows unchanged.
    
    Args:
        df (pandas.DataFrame): Input dataframe with expansion flags and number data.
        column_name (str): Primary column name for number range expansion.
        print_every (int, optional): Progress reporting interval. Defaults to 1000.
        min_count (int, optional): Minimum count threshold for expansion. Defaults to 1.
    
    Returns:
        pandas.DataFrame: Combined dataframe with expanded number ranges and 
            unchanged non-expansion rows, reset with continuous index.
    """

    df = needs_expansion(df, class_var = class_var)

    unit_expanded = expand_dataframe_numbers_core(
        df.loc[df['unit_id'].notna() & df['needs_expansion']].reset_index(drop=True), 
        column_name='unit_id', 
        print_every=print_every, 
        min_count=min_count
    )

    street_expanded = expand_dataframe_numbers_core(
        df.loc[df['unit_id'].isna() & df['needs_expansion']].reset_index(drop=True), 
        column_name='street_number', 
        print_every=print_every, 
        min_count=min_count
    )

    expanded_df = pd.concat([df.loc[~df['needs_expansion']], unit_expanded, street_expanded], ignore_index = True)

    return expanded_df


def create_unique_id(df):
    """
    Create unique identifiers for DataFrame rows grouped by title number.
    
    This function adds two new columns to the DataFrame:
    - 'multi_id': Sequential number within each title_number group
    - 'unique_id': Combination of title_number and multi_id
    
    Args:
        df (pandas.DataFrame): Input DataFrame containing a 'title_number' column.
    
    Returns:
        pandas.DataFrame: Modified DataFrame with added 'multi_id' and 'unique_id' columns.
    
    Example:
        >>> df = pd.DataFrame({'title_number': [1, 1, 2, 2, 2]})
        >>> result = create_unique_id(df)
        >>> print(result[['title_number', 'multi_id', 'unique_id']])
           title_number  multi_id unique_id
        0             1         1       1-1
        1             1         2       1-2
        2             2         1       2-1
        3             2         2       2-2
        4             2         3       2-3
    """
    df['multi_id'] = df.groupby('title_number').cumcount() + 1
    df['unique_id'] = [
        str(x) + "-" + str(y)
        for x, y in zip(
            df["title_number"], df["multi_id"]
        )
    ]
    df['is_multi'] = df.groupby('title_number')['multi_id'].transform('max') > 1
    
    return df
##
##
##
##
##
##


def load_csv_from_zip(
    zip_path: str,
    csv_filename: Optional[str] = None,
    usecols: Optional[List[str]] = None,
    usecols_callable: Optional[Callable] = None,
    encoding_errors: str = "ignore",
) -> pd.DataFrame:
    """
    Load CSV from zip file with column filtering support

    Args:
        zip_path: Path to the ZIP file
        csv_filename: Optional specific CSV filename to load
        usecols: List of column names to load
        usecols_callable: Callable function to filter columns (like lambda)
        encoding_errors: How to handle encoding errors

    Returns:
        pd.DataFrame: Loaded dataframe with specified columns
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        csv_files = [f for f in zip_ref.namelist() if f.endswith(".csv")]

        if not csv_files:
            raise ValueError("No CSV files found in zip")

        if csv_filename:
            if csv_filename not in csv_files:
                raise ValueError(
                    f"CSV file '{csv_filename}' not found in zip. Available: {csv_files}"
                )
            target_file = csv_filename
        else:
            target_file = csv_files[0]

        with zip_ref.open(target_file) as csv_file:
            # Build pandas read_csv arguments
            read_csv_kwargs = {"encoding_errors": encoding_errors}

            if usecols is not None:
                read_csv_kwargs["usecols"] = usecols
            elif usecols_callable is not None:
                read_csv_kwargs["usecols"] = usecols_callable

            df = pd.read_csv(csv_file, **read_csv_kwargs)

    return df


# This function is significantly changed but I don't think it will impact anything
def load_and_prep_OCOD_data(file_path, csv_filename=None, keep_columns=None):
    """
    Load and preprocess OCOD dataset for address parsing.

    Args:
        file_path: Path to the OCOD CSV file or ZIP file containing CSV
        csv_filename: Optional specific CSV filename if loading from ZIP
        keep_columns: List of columns to keep

    Returns:
        pd.DataFrame: Processed OCOD data with normalized addresses
    """

    if keep_columns is None:
        keep_columns = [
            "title_number",
            "tenure",
            "district",
            "county",
            "region",
            "price_paid",
            "property_address",
            "country_incorporated_1",
            "country_incorporated_(1)",
        ]

    def column_filter(x):
        return x.lower().replace(" ", "_") in keep_columns

    try:
        if file_path.lower().endswith(".zip"):
            try:
                ocod_data = load_csv_from_zip(
                    zip_path=file_path,
                    csv_filename=csv_filename,
                    usecols_callable=column_filter,
                    encoding_errors="ignore",
                )
            except ValueError:
                ocod_data = load_csv_from_zip(
                    zip_path=file_path,
                    csv_filename=csv_filename,
                    encoding_errors="ignore",
                )
                ocod_data = ocod_data.rename(
                    columns=lambda x: x.lower().replace(" ", "_")
                )
                available_columns = [
                    col for col in keep_columns if col in ocod_data.columns
                ]
                ocod_data = ocod_data[available_columns]
        else:
            try:
                ocod_data = pd.read_csv(
                    file_path, usecols=column_filter, encoding_errors="ignore"
                )
            except ValueError:
                ocod_data = pd.read_csv(file_path, encoding_errors="ignore")
                ocod_data = ocod_data.rename(
                    columns=lambda x: x.lower().replace(" ", "_")
                )
                available_columns = [
                    col for col in keep_columns if col in ocod_data.columns
                ]
                ocod_data = ocod_data[available_columns]

        ocod_data = ocod_data.rename(columns=lambda x: x.lower().replace(" ", "_"))

    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {str(e)}")

    # Remove rows with empty addresses
    if "property_address" in ocod_data.columns:
        ocod_data = ocod_data.dropna(subset="property_address")
    else:
        raise ValueError("'property_address' column not found in the data")

    ocod_data.reset_index(drop=True, inplace=True)

    ocod_data["property_address"].str.lower()
    ocod_data.rename(
        columns={
            "country_incorporated_1": "country_incorporated",
            "country_incorporated_(1)": "country_incorporated",
        },
        inplace=True,
    )

    return ocod_data


def ensure_required_columns(df, required_columns, fill_value=None):
    """
    I am not sure this is neeeded any more perhaps it can be removed

    Ensure that the dataframe has all required columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    required_columns : list
        List of column names that must exist
    fill_value : any, default None
        Value to use for missing columns

    Returns:
    --------
    pandas.DataFrame
        DataFrame with all required columns
    """

    missing_columns = []

    for col in required_columns:
        if col not in df.columns:
            df[col] = fill_value
            missing_columns.append(col)

    if missing_columns:
        print(f"Added missing columns: {missing_columns}")

    return df

class AddressNode:
    def __init__(self, entity: Dict[str, Any]):
        self.entity = entity
        self.type = entity['type']
        self.text = entity['text']
        self.start = entity['start']
        self.end = entity['end']
        self.children = []
        self.parent = None
        self.unit_type = None
    
    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self
    
    def get_full_address(self) -> Dict[str, str]:
        """Build complete address by traversing up to root"""
        address = {}
        current = self
        
        # Traverse up the tree collecting all components
        while current:
            address[current.type] = current.text.strip()
            if current.unit_type:
                address['unit_type'] = current.unit_type.text.strip()
            current = current.parent
        
        return address

class AddressGraph:
    """Graph-based parser for hierarchical address entity relationships.
    
    The AddressGraph class constructs a tree structure from named entity recognition
    results to parse complex addresses containing multiple units, buildings, and
    address components. It handles hierarchical relationships between address
    entities and extracts complete address combinations.
    
    The class organizes address entities into a predefined hierarchy:
    - city (root level)
    - postcode  
    - street_name
    - street_number
    - building_name
    - number_filter
    - unit_id
    - unit_type (special handling)
    
    Args:
        entities (List[Dict[str, Any]]): List of named entities extracted from
            address text. Each entity dictionary must contain 'type', 'text',
            'start', and 'end' keys representing the entity type, text content,
            and character positions.
    
    Attributes:
        hierarchy_levels (Dict[str, int]): Mapping of entity types to their
            hierarchy levels for tree construction.
        nodes (List[AddressNode]): List of all address nodes in the graph.
        city_node (Optional[AddressNode]): Reference to the city node if present,
            used as the root of the address tree.
    
    Example:
        >>> entities = [
        ...     {'type': 'unit_type', 'text': 'Flat', 'start': 0, 'end': 4},
        ...     {'type': 'unit_id', 'text': '1A', 'start': 5, 'end': 7},
        ...     {'type': 'street_number', 'text': '25', 'start': 8, 'end': 10},
        ...     {'type': 'street_name', 'text': 'Oak St', 'start': 11, 'end': 17},
        ...     {'type': 'city', 'text': 'London', 'start': 18, 'end': 24}
        ... ]
        >>> graph = AddressGraph(entities)
        >>> addresses_df = graph.get_addresses()
    
    Note:
        The class automatically handles unit_type entities by associating them
        with the nearest preceding unit_id entity within a 30-character distance.
        Multiple complete addresses may be extracted if multiple leaf nodes exist
        in the constructed tree.
    """
    def __init__(self, entities: List[Dict[str, Any]]):
        self.hierarchy_levels = {
            'city': 0,           # Always root
            'postcode': 1,       # Child of city
            'street_name': 2,    # Child of postcode or city
            'street_number': 3,  # Child of street_name
            'building_name': 4,  # Child of street_number or street_name
            'number_filter': 5,  # Child of building_name
            'unit_id': 6,        # Child of number_filter or building_name
            'unit_type': 7       # Special handling
        }
        
        self.nodes = []
        self.city_node = None  # Track the city node separately
        
        self._build_graph(entities)
    
    def _build_graph(self, entities: List[Dict[str, Any]]):
        # Separate unit_types for special handling
        regular_entities = [e for e in entities if e['type'] != 'unit_type']
        unit_types = [e for e in entities if e['type'] == 'unit_type']
        
        # Create nodes for regular entities
        for entity in regular_entities:
            node = AddressNode(entity)
            self.nodes.append(node)
            
            # Track city node separately
            if node.type == 'city':
                self.city_node = node
        
        # Sort nodes by hierarchy level, then by position
        self.nodes.sort(key=lambda x: (self.hierarchy_levels[x.type], x.start))
        
        # Group nodes by hierarchy level
        levels = {}
        for node in self.nodes:
            level = self.hierarchy_levels[node.type]
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
        
        # Connect nodes level by level, but handle city specially
        sorted_levels = sorted(levels.keys())
        
        # If we have a city, make all non-city nodes connect to it eventually
        if self.city_node:
            # Connect other nodes starting from level 1
            for i, current_level in enumerate(sorted_levels[1:], 1):
                parent_level = sorted_levels[i-1]
                
                for child in levels[current_level]:
                    if child.type == 'city':  # Skip city nodes in regular processing
                        continue
                        
                    # Find best parent from previous level
                    best_parent = self._find_best_parent(child, levels[parent_level])
                    if best_parent:
                        best_parent.add_child(child)
                    else:
                        # If no parent found in immediate level, look further up
                        found_parent = False
                        for j in range(i-2, -1, -1):
                            if j == 0:  # Level 0 is city level - use special logic
                                self.city_node.add_child(child)
                                found_parent = True
                                break
                            else:
                                best_parent = self._find_best_parent(child, levels[sorted_levels[j]])
                                if best_parent:
                                    best_parent.add_child(child)
                                    found_parent = True
                                    break
                        
                        # If still no parent found, attach directly to city
                        if not found_parent:
                            self.city_node.add_child(child)
        else:
            # Original logic if no city node
            for i, current_level in enumerate(sorted_levels[1:], 1):
                parent_level = sorted_levels[i-1]
                
                for child in levels[current_level]:
                    best_parent = self._find_best_parent(child, levels[parent_level])
                    if best_parent:
                        best_parent.add_child(child)
                    else:
                        for j in range(i-2, -1, -1):
                            best_parent = self._find_best_parent(child, levels[sorted_levels[j]])
                            if best_parent:
                                best_parent.add_child(child)
                                break
        
        # Handle unit types
        self._connect_unit_types(unit_types)
    
    def _find_best_parent(self, child: AddressNode, potential_parents: List[AddressNode]) -> Optional[AddressNode]:
        """Find the best parent for a child node"""
        # Special handling: never make city a child of anything
        if child.type == 'city':
            return None
            
        valid_parents = []
        
        for parent in potential_parents:
            # Skip if trying to make city a non-root
            if parent.type == 'city':
                # City can be parent of anyone
                valid_parents.append((parent, 0))  # Give city priority with distance 0
            else:
                # For most relationships, parent should come after child in text
                if child.start < parent.start:  
                    distance = parent.start - child.end
                    
                    # Special case: number_filter should only parent the closest preceding unit_id
                    # It may be prudent to rethink the logic a bit as number_filter is basically a child of unit_id
                    # however the positioning is reveresed
                    if parent.type == 'number_filter' and child.type == 'unit_id':
                        # Check if there are any other unit_ids between this child and the parent
                        has_intervening_units = any(
                            node.type == 'unit_id' and 
                            child.end < node.start < parent.start
                            for node in self.nodes
                        )
                        
                        # Only allow connection if no intervening unit_ids and reasonable distance
                        if not has_intervening_units and distance <= 15:
                            valid_parents.append((parent, distance))
                    else:
                        valid_parents.append((parent, distance))
        
        if not valid_parents:
            return None
    
        # Return parent with minimum distance
        return min(valid_parents, key=lambda x: x[1])[0]
    
    def _connect_unit_types(self, unit_types: List[Dict[str, Any]]):
        """Connect unit_type entities to their corresponding unit_id nodes"""
        unit_nodes = [n for n in self.nodes if n.type == 'unit_id']
        
        for unit_node in unit_nodes:
            # Find closest preceding unit_type
            best_unit_type = None
            min_distance = float('inf')
            
            for unit_type_entity in unit_types:
                if unit_type_entity['start'] < unit_node.start:
                    distance = unit_node.start - unit_type_entity['end']
                    if distance < min_distance and distance < 30:
                        min_distance = distance
                        best_unit_type = unit_type_entity
            
            if best_unit_type:
                unit_node.unit_type = AddressNode(best_unit_type)
    
    def get_addresses(self) -> pd.DataFrame:
        """Extract all complete addresses"""
        # Find leaf nodes (nodes with no children)
        leaf_nodes = [node for node in self.nodes if not node.children]
        
        # If no leaf nodes, use all nodes (fallback)
        if not leaf_nodes:
            leaf_nodes = self.nodes
        
        addresses = []
        for leaf in leaf_nodes:
            address = leaf.get_full_address()
            addresses.append(address)
        
        if not addresses:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(addresses)
        
        # Define column order
        column_order = ['unit_type', 'unit_id', 'number_filter', 'building_name',
                       'street_number', 'street_name', 'postcode', 'city']
        
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]
        
        return df
    
    def visualize_graph(self):
        """Visualize the graph structure"""
        # Find root nodes
        roots = [node for node in self.nodes if node.parent is None]
        
        def print_tree(node, level=0):
            indent = "  " * level
            unit_info = f" (unit_type: {node.unit_type.text})" if node.unit_type else ""
            print(f"{indent}{node.type}: '{node.text.strip()}'{unit_info}")
            for child in node.children:
                print_tree(child, level + 1)
        
        print("Address Graph:")
        for root in roots:
            print_tree(root)
        
        # Also show leaf nodes for debugging
        leaves = [node for node in self.nodes if not node.children]
        print(f"\nLeaf nodes: {[(n.type, n.text.strip()) for n in leaves]}")


def parse_addresses_to_dicts(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse entities and return list of dictionaries instead of DataFrame"""
    if not entities:
        return []
    
    graph = AddressGraph(entities)
    
    # Find leaf nodes
    leaf_nodes = [node for node in graph.nodes if not node.children]
    if not leaf_nodes:
        leaf_nodes = graph.nodes
    
    # Convert each leaf to dictionary
    address_dicts = []
    for leaf in leaf_nodes:
        address = leaf.get_full_address()
        address_dicts.append(address)
    
    return address_dicts

def process_addresses(address_data_list: List[Dict]) -> pd.DataFrame:
    """Process a list of address data dictionaries and return structured address DataFrame.
    
    Takes a list of dictionaries containing address entities and metadata, parses each
    address using the AddressGraph class, and returns a consolidated DataFrame with
    all parsed addresses and their associated metadata.
    
    Args:
        address_data_list (List[Dict]): List of dictionaries, each containing:
            - 'entities' (List[Dict]): NER entities with type, text, start, end, confidence
            - 'row_index' (int): Original row index from source data
            - 'datapoint_id' (str/int): Unique identifier for the address record
            - 'original_address' (str): The original unparsed address text
    
    Returns:
        pd.DataFrame: DataFrame with parsed address components and metadata. Columns include:
            - Address components: unit_type, unit_id, number_filter, building_name,
              street_number, street_name, postcode, city (only existing columns included)
            - Metadata: datapoint_id
            Returns empty DataFrame if no addresses could be processed.
    
    Raises:
        None: Exceptions during individual address processing are caught and logged,
              allowing processing to continue with remaining addresses.
    
    Note:
        - Each input address may generate multiple output rows if multiple parsed
          addresses are extracted from the entities
        - Failed address parsing attempts are logged but don't stop overall processing
        - Column order is standardized with address components first, then metadata
    
    Example:
        >>> address_data = [{
        ...     'entities': [{'type': 'street_number', 'text': '123', 'start': 0, 'end': 3}],
        ...     'row_index': 0,
        ...     'datapoint_id': 'addr_001',
        ...     'original_address': '123 Main St'
        ... }]
        >>> df = process_addresses(address_data)
        >>> print(df.columns.tolist())
        ['street_number', 'datapoint_id']
    """
    
    all_rows = []
    
    for address_data in address_data_list:
        try:
            entities = address_data['entities']
            
            if not entities:
                continue
            
            # Get parsed addresses as dictionaries (not DataFrame)
            parsed_addresses = parse_addresses_to_dicts(entities)
            
            # Add metadata to each parsed address
            metadata = {
                'row_index': address_data['row_index'],
                'datapoint_id': address_data['datapoint_id'],
                'property_address': address_data['original_address']
            }
            
            for address_dict in parsed_addresses:
                # Merge parsed address with metadata
                full_row = {**address_dict, **metadata}
                all_rows.append(full_row)
                
        except Exception as e:
            # Log error but continue processing
            print(f"Error processing address {address_data['row_index']}: {e}")
            continue
    
    # Single DataFrame creation at the end
    if not all_rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    
    # Define column order (address columns first, then metadata)
    address_columns = ['unit_type', 'unit_id', 'number_filter', 'building_name',
                      'street_number', 'street_name', 'postcode', 'city']
    metadata_columns = ['datapoint_id']
    
    # Reorder columns (only include columns that exist)
    all_columns = address_columns + metadata_columns
    existing_columns = [col for col in all_columns if col in df.columns]
    df = df[existing_columns]
    
    return df

