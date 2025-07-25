import pandas as pd
import re
import numpy as np
import time
import zipfile
from typing import Optional, List, Callable
import io
# This  module is supposed to contain all the relevant functions for parsing the LabeledS json file


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
            postcode_district_lookup = pd.read_csv(f)[
                ["pcds", "oslaua", "oa11", "lsoa11", "msoa11", "ctry"]
            ]

            # Filter for English and Welsh postcodes
            postcode_district_lookup = postcode_district_lookup[
                (postcode_district_lookup["ctry"] == "E92000001")
                | (postcode_district_lookup["ctry"] == "W92000004")
            ]

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

            # Remove spaces from postcodes
            postcode_district_lookup["postcode2"] = (
                postcode_district_lookup["postcode2"]
                .str.lower()
                .str.replace(r"\s", r"", regex=True)
            )

            # Drop country column
            postcode_district_lookup.drop("ctry", axis=1, inplace=True)

    return postcode_district_lookup


##
##
## Expanding the addresses
##


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


def expand_dataframe_numbers(df2, column_name, print_every=1000, min_count=1):
    # cycles through the dataframe and and expands xx-to-yy formats printing every ith iteration

    # Handle empty dataframe case
    if df2.shape[0] == 0:
        return df2

    temp_list = []
    expand_time = 0
    filter_time = 0
    make_dataframe_time = 0

    for i in range(0, df2.shape[0]):
        start_expand_time = time.time()
        numbers_list = expand_multi_id(df2.loc[i][column_name])
        end_expand_time = time.time()

        numbers_list = filter_contiguous_numbers(
            numbers_list, df2.loc[i]["number_filter"]
        )

        end_filter_time = time.time()

        dataframe_len = len(numbers_list)

        # This prevents large properties counting as several properties
        if dataframe_len > min_count:
            tmp = pd.concat(
                [df2.iloc[i].to_frame().T] * dataframe_len, ignore_index=True
            )

            tmp[column_name] = numbers_list
        else:
            tmp = df2.iloc[i].to_frame().T

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


##
##
##
##
##
##


def identify_multi_addresses(all_entities):
    """
    An important part of the parsing process is knowing which addresses represent multiple properties.
    This function takes the all entities dataframe and returns three lists containing the indexes of
    the nested properties.
    multi_unit_id: The multi properties are units not entire properties aka flats
    multi_property: The properties are nested multiproperties but are not flats
    all_multi_ids: the combination of both.
    """

    xx_to_yy_regex = r"^\d+\s?(?:-|to)\s?\d+$"

    multi_check_df = all_entities[
        [
            "datapoint_id",
            "text",
        ]
    ].drop_duplicates()
    multi_check_df["comma_count"] = multi_check_df["text"].str.count(",")
    multi_check_df["land"] = multi_check_df["text"].str.contains(
        r"^(?:land|plot|airspace|car|parking)", case=False
    )

    multi_check_df["business"] = multi_check_df["text"].str.contains(
        r"cinema|hotel|office|centre|\bpub|holiday\s?inn|travel\s?lodge|business|cafe|^shop| shop|restaurant|home|^stores?\b|^storage\b|company|ltd|limited|plc|retail|leisure|industrial|hall of|trading|commercial|works",
        case=False,
    )

    temp_df = (
        all_entities[["datapoint_id", "label"]]
        .groupby(["datapoint_id", "label"])
        .value_counts()
        .to_frame(name="counts")
        .reset_index()
        .pivot(index="datapoint_id", columns="label", values="counts")
        .fillna(0)
    )

    xx_to_yy_street_counts = (
        all_entities["datapoint_id"][
            all_entities["label_text"].str.contains(xx_to_yy_regex)
            & (all_entities["label"] == "street_number")
        ]
        .to_frame(name="datapoint_id")
        .groupby("datapoint_id")
        .size()
        .to_frame(name="xx_to_yy_street_counts")
    )

    xx_to_yy_unit_counts = (
        all_entities["datapoint_id"][
            all_entities["label_text"].str.contains(xx_to_yy_regex)
            & (all_entities["label"] == "unit_id")
        ]
        .to_frame(name="datapoint_id")
        .groupby("datapoint_id")
        .size()
        .to_frame(name="xx_to_yy_unit_counts")
    )

    multi_check_df = (
        multi_check_df.merge(
            temp_df, how="left", left_on="datapoint_id", right_index=True
        )
        .merge(
            xx_to_yy_street_counts, how="left", left_on="datapoint_id", right_index=True
        )
        .merge(
            xx_to_yy_unit_counts, how="left", left_on="datapoint_id", right_index=True
        )
        .fillna(0)
    )

    # Ensures the neccesary columns are present
    required_columns = ["building_name", "unit_id", "street_number"]
    for col in required_columns:
        if col not in multi_check_df.columns:
            multi_check_df[col] = 0

    del xx_to_yy_street_counts
    del xx_to_yy_unit_counts

    # separate the classes using logical rules
    multi_check_df["class"] = np.select(
        [
            multi_check_df["land"],  # Land/plot addresses are single properties
            multi_check_df[
                "business"
            ],  # Business addresses are typically single properties
            (multi_check_df["building_name"] == 1)
            & (
                multi_check_df["unit_id"] == 0
            ),  # Single building name without units = single property
            (multi_check_df["xx_to_yy_unit_counts"] > 0)
            | (
                multi_check_df["xx_to_yy_street_counts"] > 0
            ),  # Range patterns in unit IDs = multiple units
            multi_check_df["street_number"]
            > 1,  # Multiple street numbers = multiple properties
            multi_check_df["unit_id"] > 1,  # Multiple unit IDs = multiple units
            (multi_check_df["street_number"] <= 1)
            & (multi_check_df["xx_to_yy_street_counts"] == 0)
            & (
                multi_check_df["unit_id"] <= 1
            ),  # Single street number, no ranges, single/no unit = single property
        ],
        [
            "single",
            "single",
            "single",
            "multi",
            "multi",
            "multi",
            "single",
        ],
        default="unknown",  # Fallback for edge cases
    )
    # With the multiaddress dataframe created the required vectors can now be produced

    multi_unit_id = set(
        multi_check_df["datapoint_id"][
            (multi_check_df["class"] == "multi") & (multi_check_df["unit_id"] > 0)
        ].tolist()
    )
    multi_property = set(
        multi_check_df["datapoint_id"][
            (multi_check_df["class"] == "multi") & (multi_check_df["unit_id"] == 0)
        ].tolist()
    )
    all_multi_ids = list(multi_unit_id) + list(multi_property)

    return multi_unit_id, multi_property, all_multi_ids


def spread_address_labels(df, all_multi_ids):
    """
    This function spreads the address dataframe so that each
    label class is its own column
    """
    # Filter and pivot in one step, using datapoint_id as the index
    temp_df = df[df.datapoint_id.isin(all_multi_ids)].copy()
    
    # Pivot with datapoint_id as index to preserve it
    pivoted_df = temp_df.pivot_table(
        index="datapoint_id", 
        columns="label", 
        values="label_text",
        aggfunc='first'  # Handle duplicates if any
    )
    
    # Reset index to make datapoint_id a column again
    pivoted_df = pivoted_df.reset_index()
    
    # Add back the text column
    pivoted_df = pivoted_df.merge(
        temp_df[["datapoint_id", "text"]].drop_duplicates(),
        on="datapoint_id",
        how="left"
    )
    
    return pivoted_df


def add_backfill_blockers(df):
    """
    This places blockers in the spread address dataframe to prevent labels
    being propergates back or forword when not logical.
    As an example if a building is going     to back fill up previous addresses it should not
    back fill past another street as this is highly unlikely to be the same building

    """

    # This indexing should be done using loc
    df.loc[df["street_name"].notnull(), "building_name"] = "block"
    df.loc[df["street_name"].notnull(), "street_number"] = (
        "block"  # for multi-flats inside a common building
    )

    # returns true if current number filter is null and the next row has street_number or unit id is not null
    # prevents number filters propergsating back across roads and unit ids
    number_filter_block = df["number_filter"].isnull() & (
        df["street_number"].shift().notnull() | df["unit_id"].shift().notnull()
    )
    df.loc[number_filter_block, "number_filter"] = "block"

    return df


# This function has been significantly modified from the original
def backfill_address_labels(df):
    """
    Backfilling adds address information in. However, street address should only be back filled for multi addresses.
    I need to work out how to do flat, which may be before or after the unit ID
    Also I don't think this is a very good way of doing it at all.
    """

    # Define column groups for different fill strategies
    backfill_columns = [
        "number_filter",
        "building_name",
        "street_number",
        "postcode",
        "street_name",
        "city",
    ]
    forward_fill_columns = ["unit_type"]

    # Vectorized backfill - more efficient than loops
    for col in backfill_columns:
        if col in df.columns:
            df[col] = df.groupby("datapoint_id")[col].transform(lambda x: x.bfill())

    # Vectorized forward fill
    for col in forward_fill_columns:
        if col in df.columns:
            df[col] = df.groupby("datapoint_id")[col].transform(lambda x: x.ffill())

    return df


def final_parsed_addresses(
    df,
    all_entities,
    multi_property,
    multi_unit_id,
    all_multi_ids,
    expand_addresses=True,
):
    """
    This function creates the final parsed address dataframe.
    It can either expand the multi-addresses in the format xx to yy or not.
    This is because other address parsers are not designed to perform such and expansion
    and so would make such a comparison unfair.
    """
    xx_to_yy_regex = r"^\d+\s?(?:-|to|\/)\s?\d+$"

    expanded_street = df[
        df.datapoint_id.isin(multi_property)
        & df.street_number.str.contains(xx_to_yy_regex)
    ].reset_index()
    expanded_unit_id = df[
        df.datapoint_id.isin(multi_unit_id) & df.unit_id.str.contains(xx_to_yy_regex)
    ].reset_index()

    # Generally expansion is required as it changes the format to 1 address per row
    # N.B. not all expanded addresses are valid. Office blocks are 1 property but can cover multiple street addresses
    # A matching and cleaning process is required to identify what should be expanded and what not
    if expand_addresses:
        expanded_street = expand_dataframe_numbers(
            expanded_street, column_name="street_number"
        )
        expanded_unit_id = expand_dataframe_numbers(
            expanded_unit_id, column_name="unit_id"
        )

    # unit id and street number that does does not have the xx to yy format and so has already been expanded by spreading and backfilling
    expanded_street_simple = df[
        df.datapoint_id.isin(multi_property)
        & (~df.street_number.str.contains(xx_to_yy_regex).fillna(False))
        & (df.street_number != "block")
    ].reset_index()

    expanded_unit_id_simple = df[
        df.datapoint_id.isin(multi_unit_id)
        & (~df.unit_id.str.contains(xx_to_yy_regex).fillna(False))
        & (df.unit_id != "block")
    ].reset_index()

    # remove the multi-addresses
    single_address_only = all_entities[
        ~all_entities["datapoint_id"].isin(all_multi_ids)
    ]
    # remove all but the first instance of a label in the remaining instances
    # this is because for single addresses there should be only a single label for each class
    single_address_only = single_address_only[
        single_address_only["label_id_count"] == 0
    ]
    df2 = single_address_only.pivot(
        index="datapoint_id", columns="label", values="label_text"
    )
    # add the datapoint_id back in for each of joining
    df2 = df2.merge(
        single_address_only[["datapoint_id", "text"]].drop_duplicates(),
        how="left",
        left_on="datapoint_id",
        right_on="datapoint_id",
    )

    full_expanded_data = pd.concat(
        [
            expanded_street,
            expanded_unit_id,
            expanded_street_simple,
            expanded_unit_id_simple,
            df2,
        ]
    )

    return full_expanded_data


# There is a significant change here which is that the column filter_type is renamed number_filter...
# but this would only effect expansions which had that no?
def parsing_and_expansion_process(
    all_entities: pd.DataFrame,
    expand_addresses: bool = False,
    required_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Process address data through parsing and expansion pipeline.

    Combines address identification, label spreading, column validation,
    backfilling, and final address generation into a single workflow.

    Parameters
    ----------
    all_entities : pd.DataFrame
        Input dataframe containing raw address data
    expand_addresses : bool, default False
        If True, expands addresses into multiple records
    required_columns : list of str, optional
        Required output columns. Uses standard address columns if None

    Returns
    -------
    pd.DataFrame
        Processed dataframe with parsed addresses and required columns
    """

    # Define default required columns
    if required_columns is None:
        required_columns = [
            "building_name",
            "street_name",
            "street_number",
            "number_filter",
            "unit_id",
            "unit_type",
            "city",
            "postcode",
        ]

    # Continue with existing logic
    multi_unit_id, multi_property, all_multi_ids = identify_multi_addresses(
        all_entities
    )
    df = spread_address_labels(all_entities, all_multi_ids)

    # The columns are filled with an empty string as at the moment all columns should be strings
    # Ensurinng string prevents errors later when cleaning is performed on street_name and other variables
    # This is not being changed to default behaviour as I may need to implement more significant changes later
    df = ensure_required_columns(df, required_columns, "")

    # Blockers prevent the filling of wrong information. As an example if a building is going to back fill up
    # previous addresses it should not back fill past another street as this is highly unlikely to be the same building
    df = add_backfill_blockers(df)
    df = backfill_address_labels(df)

    df = final_parsed_addresses(
        df,
        all_entities,
        multi_property,
        multi_unit_id,
        all_multi_ids,
        expand_addresses=expand_addresses,
    )

    return df


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


def post_process_expanded_data(expanded_data, ocod_data):
    """
    This function adds in additional meta-data from the ocod dataset and prepares the final expanded dataset to be
    exported for geo-location and classification
    it takes two arguements
    expanded_data is a pandas dataframe produced by the 'final_parsed_addresses' function
    """
    full_expanded_data = expanded_data.merge(
        ocod_data, how="left", left_on="datapoint_id", right_index=True
    )

    full_expanded_data["within_title_id"] = (
        full_expanded_data.groupby("title_number").cumcount() + 1
    )
    full_expanded_data["unique_id"] = [
        str(x) + "-" + str(y)
        for x, y in zip(
            full_expanded_data["title_number"], full_expanded_data["within_title_id"]
        )
    ]

    tmp_df = (
        full_expanded_data[["title_number", "within_title_id"]]
        .groupby("title_number")
        .max("within_title_id")
    ) > 1
    tmp_df.columns = tmp_df.columns.str.replace(
        "within_title_id", "within_larger_title"
    )  # could also be called nested_address
    full_expanded_data = full_expanded_data.merge(
        tmp_df, how="left", left_on="title_number", right_index=True
    )

    full_expanded_data["postcode"] = full_expanded_data["postcode"].str.upper()
    del tmp_df

    # re-order the columns and drop columns that are not needed

    full_expanded_data = full_expanded_data[
        [
            "title_number",
            "within_title_id",
            "unique_id",
            "within_larger_title",
            "tenure",
            "unit_id",
            "unit_type",
            "building_name",
            "street_number",
            "street_name",
            "postcode",
            "city",
            "district",
            "county",
            "region",
            "price_paid",
            "property_address",
            "country_incorporated",
        ]
    ].replace("block", np.NaN)

    return full_expanded_data


def ensure_required_columns(df, required_columns, fill_value=None):
    """
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
