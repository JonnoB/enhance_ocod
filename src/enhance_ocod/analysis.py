"""
Basic analysis functions to create summary statistics of the OCOD timeseries
"""

import pandas as pd
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm


def add_file_metadata(df, filename_stem):
    """Add filename and date columns to a dataframe.

    Extracts date information from filename pattern and adds it to the dataframe.
    Expects filename format ending with '_YYYY_MM'.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to add metadata to
    filename_stem : str
        Filename stem in format '*_YYYY_MM' (e.g., 'OCOD_FULL_2022_01')

    Returns
    -------
    pd.DataFrame
        Copy of input dataframe with added 'filename' and 'date' columns.
        Date is parsed as first day of the month from filename.
    """
    df = df.copy()
    df['filename'] = filename_stem

    # Parse date once from filename
    parts = filename_stem.split('_')
    date_str = f"{parts[-2]}-{parts[-1]}-01"
    df['date'] = pd.to_datetime(date_str)

    return df


def create_summarised_stats(list_of_files, class_var='class2', expansion_threshold=None):
    """Create summary statistics from OCOD files.

    Processes multiple OCOD parquet files to generate summary statistics
    including residential property counts, regional distributions, and
    country of incorporation data.

    Parameters
    ----------
    list_of_files : list of pathlib.Path
        List of file paths to OCOD parquet files to process
    class_var : str, optional
        Column name for property class classification, by default 'class2'
    expansion_threshold : int, optional
        Maximum expansion_size to include. Filters out rows where
        expansion_size >= threshold. If None, no filtering is applied.
        By default None.

    Returns
    -------
    tuple of pd.DataFrame
        Four dataframes containing:
        - total_residential_df : Residential property counts by is_multi flag
        - total_per_region_df : Property counts by region and class
        - total_incorp_df : Property counts by country of incorporation
        - total_resi_lad_df : Residential property counts by LAD and is_multi flag

    Examples
    --------
    >>> files = list(Path('data/ocod').glob('*.parquet'))
    >>> res_df, region_df, incorp_df, lad_df = create_summarised_stats(files)
    >>> # Filter out runaway expansions
    >>> res_df, region_df, incorp_df, lad_df = create_summarised_stats(
    ...     files, expansion_threshold=1000
    ... )
    """
    # Initialize lists
    total_residential_df = []
    total_per_region_df = []
    total_incorp_df = []
    total_resi_lad_df = []

    for target_file in tqdm(list_of_files):
        target_year = pd.read_parquet(target_file)
        # Filter out runaway expansions if threshold is specified
        if expansion_threshold is not None:
            target_year = target_year.loc[target_year['expansion_size'] < expansion_threshold]
        filename_stem = target_file.stem
        
        # Filter residential data once
        residential_data = target_year.loc[target_year[class_var] == 'residential']
        
        # Generate all required groupings
        groupings = {
            'total_residential': residential_data.groupby('is_multi').size().reset_index().rename(columns={0: 'counts'}),
            'total_resi_lad': residential_data.groupby(['is_multi', 'lad11cd']).size().reset_index().rename(columns={0: 'counts'}),
            'total_per_region': target_year.groupby(['region', class_var]).size().reset_index().rename(columns={0: 'counts'}),
            'total_incorp': target_year.groupby('country_incorporated').size().reset_index().rename(columns={0: 'counts'})
        }
        
        # Add metadata and append to respective lists
        total_residential_df.append(add_file_metadata(groupings['total_residential'], filename_stem))
        total_resi_lad_df.append(add_file_metadata(groupings['total_resi_lad'], filename_stem))
        total_per_region_df.append(add_file_metadata(groupings['total_per_region'], filename_stem))
        total_incorp_df.append(add_file_metadata(groupings['total_incorp'], filename_stem))

    # Concatenate all dataframes
    total_residential_df = pd.concat(total_residential_df, ignore_index=True)
    total_per_region_df = pd.concat(total_per_region_df, ignore_index=True)
    total_incorp_df = pd.concat(total_incorp_df, ignore_index=True)
    total_resi_lad_df = pd.concat(total_resi_lad_df, ignore_index=True) 

    return total_residential_df, total_per_region_df, total_incorp_df, total_resi_lad_df



def create_time_series_by_groups(msoa_dwellings, grouping_vars=None,
    class_var='class',
    ocod_path='../data/ocod_history_processed',
    price_paid_path='../data/price_paid_msoa_averages',
    expansion_threshold=None):
    """Create aggregated residential property time series data by groups.

    Processes OCOD (Overseas Companies Ownership Dataset) and price paid data
    to create time series of residential property statistics. Properties without
    MSOA codes are assigned LAD-level average prices to preserve absolute values
    while maintaining representative pricing.

    Parameters
    ----------
    msoa_dwellings : pd.DataFrame
        DataFrame containing MSOA dwelling counts with 'msoa11cd' and
        'dwellings' columns.
    grouping_vars : list of str, optional
        Column names to group by for aggregation (e.g., ['region', 'county']).
        If None, aggregates at MSOA level only. Default is None.
    class_var : str, optional
        Column name used to filter for residential properties.
        Default is 'class2'.
    ocod_path : str or Path, optional
        Path to directory containing OCOD parquet files. Files should be named
        with pattern '*_YYYY_MM.parquet'. Default is '../data/ocod_history_processed'.
    price_paid_path : str or Path, optional
        Path to directory containing price paid parquet files. Files should be
        named 'price_paid_YYYY_MM.parquet'. Default is '../data/price_paid_msoa_averages'.
    expansion_threshold : int, optional
        Maximum expansion_size to include. Filters out rows where
        expansion_size >= threshold. If None, no filtering is applied.
        By default None.
    
    Returns
    -------
    pd.DataFrame
        Time series DataFrame with columns:
        - date : datetime, first day of month
        - year : int, year of data
        - month : int, month of data
        - ocod_mean : int, OCOD count-weighted mean price
        - ocod_median : int, OCOD count-weighted median price
        - dwelling_mean : int, dwelling count-weighted mean price
        - dwelling_median : int, dwelling count-weighted median price
        - ocod_ratio_mean : float, ratio of OCOD to dwelling weighted means
        - ocod_total_counts : int, total count of OCOD residential properties
        - total_dwelling_count : int, total dwelling count
        - total_value_ocod_mean : int, total property value using OCOD weighting
        - total_value_dwelling_mean : int, total property value using dwelling weighting
        - fraction_of_total_value : float, fraction of total value represented by OCOD
        - [grouping_vars] : additional columns if grouping_vars specified
    
    Notes
    -----
    - Only processes residential properties as defined by class_var
    - Properties without MSOA codes are assigned 'UNKNOWN_[LAD_CODE]' identifiers
      and given LAD-level average prices to preserve absolute property values
    - Skips months where either OCOD or price paid data is missing
    - Results are sorted by date and grouping variables
    - Missing dwelling counts are filled with 0
    
    Warnings
    --------
    Prints warnings for:
    - Missing price paid files
    - Months with no residential properties
    - Processing errors for individual files
    - Properties without MSOA codes being assigned LAD averages
    
    Examples
    --------
    >>> # Basic usage - MSOA level aggregation
    >>> result = create_time_series_by_groups(msoa_dwellings)
    
    >>> # Regional aggregation
    >>> result = create_time_series_by_groups(
    ...     msoa_dwellings, 
    ...     grouping_vars=['region']
    ... )
    
    >>> # Multiple grouping with custom paths
    >>> result = create_time_series_by_groups(
    ...     msoa_dwellings,
    ...     grouping_vars=['region', 'county'],
    ...     ocod_path='./data/ocod',
    ...     price_paid_path='./data/prices'
    ... )
    """
    ocod_path = Path(ocod_path)
    price_paid_path = Path(price_paid_path)
    results = []

    # Iterate through OCOD files
    for ocod_file in tqdm(list(ocod_path.glob('*.parquet'))):
        year, month = _extract_year_month(ocod_file)
        price_paid_file = price_paid_path / f'price_paid_{year}_{month:02d}.parquet'
        
        # Skip if price paid file is missing
        if not price_paid_file.exists():
            print(f"Warning: {price_paid_file} not found, skipping...")
            continue
        
        try:
            ocod_df, price_paid_df = _load_and_preprocess_data(ocod_file, price_paid_file, class_var, expansion_threshold)
            
            # Skip if no residential properties
            if ocod_df.empty:
                print(f"Warning: No residential properties for {year}-{month:02d}, skipping...")
                continue
            
            # Perform aggregation
            time_series_data = _aggregate_time_series_data(
                ocod_df, 
                price_paid_df, 
                msoa_dwellings,
                year, 
                month, 
                grouping_vars
            )
            
            results.extend(time_series_data)
            
        except Exception as e:
            print(f"Error processing {ocod_file.name}: {str(e)}")
            continue

    # Create and sort DataFrame
    return _create_final_dataframe(results, grouping_vars)

def _extract_year_month(ocod_file):
    """Extract year and month from OCOD filename.

    Parses filename stem expecting pattern ending with '_YYYY_MM'.

    Parameters
    ----------
    ocod_file : pathlib.Path
        Path object for OCOD file with naming pattern '*_YYYY_MM.parquet'

    Returns
    -------
    tuple of (int, int)
        Year and month as integers (year, month)
    """
    filename_parts = ocod_file.stem.split('_')
    return int(filename_parts[-2]), int(filename_parts[-1])

def _load_and_preprocess_data(ocod_file, price_paid_file, class_var='class2', expansion_threshold=None):
    """Load and preprocess OCOD and price paid data.

    Reads OCOD and price paid parquet files, optionally filters by expansion
    threshold, and extracts residential properties.

    Parameters
    ----------
    ocod_file : pathlib.Path
        Path to OCOD parquet file
    price_paid_file : pathlib.Path
        Path to price paid parquet file
    class_var : str, optional
        Column name for property class classification, by default 'class2'
    expansion_threshold : int, optional
        Maximum expansion_size to include. Filters out rows where
        expansion_size >= threshold. If None, no filtering is applied.
        By default None.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        - ocod_residential : Filtered OCOD data containing only residential properties
        - price_paid_df : Unfiltered price paid data
    """
    ocod_df = pd.read_parquet(ocod_file)
    # Filter out runaway expansions if threshold is specified
    if expansion_threshold is not None:
        ocod_df = ocod_df.loc[ocod_df['expansion_size'] < expansion_threshold]
    price_paid_df = pd.read_parquet(price_paid_file)

    # Filter for residential properties
    ocod_residential = ocod_df.loc[ocod_df[class_var] == 'residential'].copy()

    return ocod_residential, price_paid_df

def _aggregate_time_series_data(ocod_df, price_paid_df, msoa_dwellings, year, month, grouping_vars):
    """Aggregate time series data with optional grouping, handling missing MSOA codes.

    Processes OCOD and price paid data to create aggregated statistics. Properties
    without MSOA codes are assigned LAD-level average prices.

    Parameters
    ----------
    ocod_df : pd.DataFrame
        OCOD data containing residential properties
    price_paid_df : pd.DataFrame
        Price paid data with MSOA-level price information
    msoa_dwellings : pd.DataFrame
        Dwelling counts by MSOA with columns 'msoa11cd' and 'dwellings'
    year : int
        Year for the data
    month : int
        Month for the data
    grouping_vars : list of str or None
        Column names to group by for aggregation. If None, aggregates
        at MSOA level only.

    Returns
    -------
    list of dict
        List of dictionaries containing aggregated statistics for each group,
        including weighted prices, counts, and value calculations.
    """
    
    # Separate properties with and without MSOA codes
    ocod_with_msoa = ocod_df[ocod_df['msoa11cd'].notna()].copy()
    ocod_without_msoa = ocod_df[ocod_df['msoa11cd'].isna()].copy()
    
    # Process properties with MSOA codes normally
    group_cols = ['msoa11cd'] + (grouping_vars or [])
    ocod_grouped = ocod_with_msoa.groupby(group_cols).size().reset_index(name='ocod_total_counts')
    
    # Merge with price data and dwelling data
    df = price_paid_df.merge(ocod_grouped, on='msoa11cd', how='right')
    df = df.merge(msoa_dwellings, on='msoa11cd', how='left')
    df['ocod_total_counts'] = df['ocod_total_counts'].fillna(0).astype(int)
    
    # Handle properties without MSOA codes
    if not ocod_without_msoa.empty and 'lad21cd' in ocod_without_msoa.columns:
        # Calculate LAD-level averages from existing MSOA data
        lad_averages = _calculate_lad_averages(df, grouping_vars)
        
        # Create rows for unknown MSOA properties
        unknown_rows = _create_unknown_msoa_rows(ocod_without_msoa, lad_averages, grouping_vars)
        
        # Append to main dataframe
        if unknown_rows:
            df = pd.concat([df, pd.DataFrame(unknown_rows)], ignore_index=True)
    
    # Create date
    date = datetime(year, month, 1)
    
    # Perform aggregation based on grouping
    return _calculate_aggregations(df, date, year, month, grouping_vars)

def _calculate_lad_averages(df, grouping_vars):
    """Calculate LAD-level price averages for assigning to unknown MSOA properties.

    Computes weighted average prices at the LAD level to be used for properties
    that lack MSOA codes.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing MSOA-level data with 'lad21cd', 'price_mean',
        'price_median', and 'ocod_total_counts' columns
    grouping_vars : list of str or None
        Additional grouping variables beyond LAD code

    Returns
    -------
    dict
        Dictionary mapping group keys (tuples) to dictionaries containing
        'price_mean' and 'price_median' weighted averages. Empty dict if
        'lad21cd' column not present.
    """
    if 'lad21cd' not in df.columns:
        return {}
    
    # Group by LAD (and other grouping vars if present)
    lad_group_cols = ['lad21cd'] + (grouping_vars or [])
    
    lad_averages = {}
    for group_key, group_df in df.groupby(lad_group_cols):
        if isinstance(group_key, str):  # Single LAD grouping
            lad_code = group_key
            group_key = (group_key,)  # Make it a tuple for consistency
        else:
            lad_code = group_key[0]
        
        # Calculate weighted averages within the LAD
        if group_df['ocod_total_counts'].sum() > 0:
            weights = group_df['ocod_total_counts']
            valid_prices = group_df[['price_mean', 'price_median']].notna().all(axis=1)
            
            if valid_prices.any():
                weighted_mean = np.average(group_df.loc[valid_prices, 'price_mean'], 
                                         weights=weights[valid_prices])
                weighted_median = np.average(group_df.loc[valid_prices, 'price_median'], 
                                           weights=weights[valid_prices])
                
                lad_averages[group_key] = {
                    'price_mean': weighted_mean,
                    'price_median': weighted_median
                }
    
    return lad_averages

def _create_unknown_msoa_rows(ocod_without_msoa, lad_averages, grouping_vars):
    """Create rows for properties without MSOA codes using LAD averages.

    Generates data rows for properties lacking MSOA codes by assigning them
    LAD-level average prices and creating synthetic MSOA identifiers.

    Parameters
    ----------
    ocod_without_msoa : pd.DataFrame
        OCOD data for properties without MSOA codes, must contain 'lad21cd'
    lad_averages : dict
        Dictionary mapping LAD group keys to price averages, as returned
        by _calculate_lad_averages
    grouping_vars : list of str or None
        Additional grouping variables beyond LAD code

    Returns
    -------
    list of dict
        List of dictionaries representing rows for unknown MSOA properties,
        with synthetic 'msoa11cd' values ('UNKNOWN_[LAD_CODE]'), LAD average
        prices, and property counts.
    """
    unknown_rows = []
    
    # Group by LAD (and other grouping vars)
    lad_group_cols = ['lad21cd'] + (grouping_vars or [] if grouping_vars else [])
    lad_group_cols = [col for col in lad_group_cols if col in ocod_without_msoa.columns]
    
    for group_key, group_df in ocod_without_msoa.groupby(lad_group_cols):
        if isinstance(group_key, str):
            group_key = (group_key,)  # Make it a tuple for consistency
        
        # Get LAD average prices
        if group_key in lad_averages:
            lad_avg = lad_averages[group_key]
            
            # Create a row for unknown MSOA properties in this LAD
            unknown_row = {
                'msoa11cd': 'UNKNOWN_' + group_key[0],  # Use LAD code in the unknown identifier
                'price_mean': lad_avg['price_mean'],
                'price_median': lad_avg['price_median'],
                'ocod_total_counts': len(group_df),
                'dwellings': 0  # We don't have dwelling counts for unknown areas
            }
            
            # Add grouping variables
            if grouping_vars:
                for i, var in enumerate(grouping_vars):
                    if i + 1 < len(group_key):  # +1 because first element is LAD code
                        unknown_row[var] = group_key[i + 1]
                    elif var in group_df.columns:
                        unknown_row[var] = group_df[var].iloc[0]
            
            unknown_rows.append(unknown_row)
    
    return unknown_rows

def _calculate_aggregations(df, date, year, month, grouping_vars):
    """Calculate aggregation metrics for time series data.

    Computes weighted price statistics, property counts, and value calculations
    for OCOD data, optionally grouped by specified variables.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe with price and property count data
    date : datetime.datetime
        Date for the data point (first day of month)
    year : int
        Year of the data
    month : int
        Month of the data
    grouping_vars : list of str or None
        Column names to group by. If None, treats entire dataframe as one group.

    Returns
    -------
    list of dict
        List of dictionaries containing aggregated metrics including weighted
        prices, counts, total values, and ratios for each group.
    """
    def calculate_metrics(group_df):
        """Calculate metrics for a group.

        Parameters
        ----------
        group_df : pd.DataFrame
            Subset of data for a specific group

        Returns
        -------
        dict
            Dictionary of calculated metrics for the group
        """
        # Fill NaN values with 0 for counts
        group_df['ocod_total_counts'] = group_df['ocod_total_counts'].fillna(0)
        group_df['dwellings'] = group_df['dwellings'].fillna(0)
        
        # Helper function
        def safe_int(value):
            return int(value) if pd.notna(value) and not np.isnan(value) else None
        
        # Weighted by OCOD counts
        if group_df['ocod_total_counts'].sum() > 0:
            ocod_weighted_mean = np.average(group_df['price_mean'], weights=group_df['ocod_total_counts'])
            ocod_weighted_median = np.average(group_df['price_median'], weights=group_df['ocod_total_counts'])
        else:
            ocod_weighted_mean = np.nan
            ocod_weighted_median = np.nan
        
        # Weighted by dwelling counts
        if group_df['dwellings'].sum() > 0:
            dwelling_weighted_mean = np.average(group_df['price_mean'], weights=group_df['dwellings'])
            dwelling_weighted_median = np.average(group_df['price_median'], weights=group_df['dwellings'])
        else:
            dwelling_weighted_mean = np.nan
            dwelling_weighted_median = np.nan
        
        # Total counts and values
        total_ocod_counts = group_df['ocod_total_counts'].sum()
        total_dwelling_count = group_df['dwellings'].sum()
        
        # Handle NaN in calculations
        if pd.notna(ocod_weighted_mean):
            total_value_ocod_mean = total_ocod_counts * ocod_weighted_mean
        else:
            total_value_ocod_mean = np.nan
            
        if pd.notna(dwelling_weighted_mean):
            total_value_dwelling_mean = total_dwelling_count * dwelling_weighted_mean
        else:
            total_value_dwelling_mean = np.nan
        
        # Fraction of total value
        if pd.notna(total_value_ocod_mean) and pd.notna(total_value_dwelling_mean) and total_value_dwelling_mean > 0:
            fraction_of_total_value = total_value_ocod_mean / total_value_dwelling_mean
        else:
            fraction_of_total_value = np.nan
        
        result = {
            'date': date,
            'year': year,
            'month': month,
            'ocod_mean': safe_int(ocod_weighted_mean),
            'ocod_median': safe_int(ocod_weighted_median),
            'dwelling_mean': safe_int(dwelling_weighted_mean),
            'dwelling_median': safe_int(dwelling_weighted_median),
            'ocod_ratio_mean': ocod_weighted_mean / dwelling_weighted_mean if pd.notna(dwelling_weighted_mean) and dwelling_weighted_mean > 0 else np.nan,
            'ocod_total_counts': int(total_ocod_counts),
            'total_dwelling_count': int(total_dwelling_count),
            'total_value_ocod_mean': safe_int(total_value_ocod_mean),
            'total_value_dwelling_mean': safe_int(total_value_dwelling_mean),
            'fraction_of_total_value': fraction_of_total_value
        }
        
        # Add grouping variables if applicable
        if grouping_vars:
            for var in grouping_vars:
                result[var] = group_df[var].iloc[0]
        
        return result

    # If no grouping, treat entire dataframe as one group
    if not grouping_vars:
        return [calculate_metrics(df)]
    
    # Group and calculate metrics
    return [
        calculate_metrics(group_df) 
        for _, group_df in df.groupby(grouping_vars) 
        if not group_df.empty
    ]

def _create_final_dataframe(results, grouping_vars):
    """Create and sort final DataFrame from aggregation results.

    Parameters
    ----------
    results : list of dict
        List of dictionaries containing aggregation results
    grouping_vars : list of str or None
        Column names used for grouping, affects sort order

    Returns
    -------
    pd.DataFrame
        Sorted dataframe with all results. Returns empty dataframe if
        results list is empty. Sorted by date and grouping variables.
    """
    if not results:
        return pd.DataFrame()

    time_series_df = pd.DataFrame(results)

    # Sort columns
    sort_cols = ['date']
    if grouping_vars:
        sort_cols.extend(grouping_vars)

    return time_series_df.sort_values(sort_cols).reset_index(drop=True)

import numpy as np
from scipy import stats

def bootstrap_ratio_test(ratios, n_bootstrap=1000, random_state=42):
    """Vectorized bootstrap test for whether mean ratio is significantly different from 1.

    Performs bootstrap resampling to test if the mean of price ratios differs
    significantly from 1.0 using a two-tailed test at the 0.05 significance level.

    Parameters
    ----------
    ratios : array-like
        Array of price ratios to test
    n_bootstrap : int, optional
        Number of bootstrap samples to generate, by default 1000
    random_state : int, optional
        Random seed for reproducibility, by default 42

    Returns
    -------
    dict
        Dictionary containing:
        - fraction_above_one : float
            Fraction of bootstrap means above 1.0
        - fraction_below_one : float
            Fraction of bootstrap means below 1.0
        - significantly_different : bool
            Whether ratio is significantly different from 1.0 (p < 0.05)
        - mean_ratio : float
            Mean of input ratios
        - ci_lower : float
            Lower bound of 95% confidence interval
        - ci_upper : float
            Upper bound of 95% confidence interval

    Notes
    -----
    Returns NaN values and False for significantly_different if fewer than
    2 ratios are provided.
    """
    ratios = np.array(ratios)
    n = len(ratios)
    
    if n < 2:
        return {
            'fraction_above_one': np.nan,
            'fraction_below_one': np.nan,
            'significantly_different': False,
            'mean_ratio': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan
        }
    
    np.random.seed(random_state)
    
    # Generate bootstrap samples - vectorized
    bootstrap_indices = np.random.choice(n, size=(n_bootstrap, n), replace=True)
    bootstrap_samples = ratios[bootstrap_indices]
    
    # Calculate means for all bootstrap samples at once
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    
    # Calculate results
    fraction_above_one = np.mean(bootstrap_means > 1.0)
    fraction_below_one = np.mean(bootstrap_means < 1.0)
    
    # Check if significantly different from 1 (two-tailed test at 0.05 level)
    significantly_different = (fraction_above_one > 0.975) or (fraction_below_one > 0.975)
    
    mean_ratio = np.mean(ratios)
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    
    return {
        'fraction_above_one': fraction_above_one,
        'fraction_below_one': fraction_below_one,
        'significantly_different': significantly_different,
        'mean_ratio': mean_ratio,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def create_mean_difference_by_groups(grouping_vars,
    ocod_path = '../data/ocod_history_processed',
    price_paid_path = '../data/price_paid_msoa_averages',
    class_var = 'class2',
    expansion_threshold=None,
    start_date=None,
    end_date=None):
    """
    Calculate mean difference price trends for property groups.

    This function processes monthly property data to calculate weighted and
    unweighted price trends for different geographical or categorical groups.
    It compares offshore company property prices with general market prices
    and performs statistical analysis on price ratios. Returns time-aggregated
    summary statistics (one row per group) showing average monthly trends
    across the specified date range or entire dataset.

    Parameters
    ----------
    grouping_vars : list of str
        Column names to group analysis by (e.g., ['localauthority'],
        ['region', 'property_type']). Cannot be None as grouping variables
        are required for the analysis.
    ocod_path : str, optional
        Path to directory containing OCOD (Overseas Companies Ownership Data)
        processed parquet files, by default '../data/ocod_history_processed'
    price_paid_path : str, optional
        Path to directory containing price paid MSOA average parquet files,
        by default '../data/price_paid_msoa_averages'
    class_var : str, optional
        Column name used to filter for residential properties,
        by default 'class2'
    expansion_threshold : int, optional
        Maximum expansion_size to include. Filters out rows where
        expansion_size >= threshold. If None, no filtering is applied.
        By default None.
    start_date : datetime, optional
        Start date for analysis (inclusive). Only process files from this
        date onwards. If None, includes all files from the beginning.
        Example: pd.to_datetime('2020-01-01') or datetime(2020, 1, 1).
        By default None.
    end_date : datetime, optional
        End date for analysis (inclusive). Only process files up to this
        date. If None, includes all files up to the end.
        Example: pd.to_datetime('2023-12-31') or datetime(2023, 12, 31).
        By default None.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing mean difference calculations for each group with
        columns:
        - mean_weighted_difference : float
            Average monthly change in weighted mean prices
        - mean_unweighted_difference : float  
            Average monthly change in unweighted mean prices
        - mean_price_ratio : float
            Mean ratio of weighted to unweighted prices
        - ratio_fraction_above_one : float
            Fraction of price ratios above 1.0
        - ratio_fraction_below_one : float
            Fraction of price ratios below 1.0
        - ratio_significantly_different : bool
            Whether price ratio is significantly different from 1.0
        - ratio_ci_lower, ratio_ci_upper : float
            Confidence interval bounds for price ratio
        - n_periods : int
            Number of time periods used in calculation
        - n_ratio_observations : int
            Number of valid price ratio observations
        - [grouping_vars] : various
            Original grouping variable columns
    
    Raises
    ------
    ValueError
        If grouping_vars is None
    
    """
    if grouping_vars is None:
        raise ValueError("grouping_vars cannot be None. Please provide grouping variables.")
    
    ocod_path = Path(ocod_path)
    price_paid_path = Path(price_paid_path)

    # List to store monthly results
    monthly_results = []

    # Get all parquet files from ocod directory
    ocod_files = sorted(list(ocod_path.glob('*.parquet')))

    for ocod_file in tqdm(ocod_files):
        # Extract year and month from filename
        filename_parts = ocod_file.stem.split('_')
        year = int(filename_parts[-2])
        month = int(filename_parts[-1])

        # Filter by date range if specified
        file_date = datetime(year, month, 1)
        if start_date is not None and file_date < start_date:
            continue
        if end_date is not None and file_date > end_date:
            continue

        # Create corresponding price_paid filename
        price_paid_file = price_paid_path / f'price_paid_{year}_{month:02d}.parquet'
        
        # Check if corresponding price_paid file exists
        if not price_paid_file.exists():
            continue
        
        try:
            # Read the data
            ocod_df = pd.read_parquet(ocod_file)
            # Filter out runaway expansions if threshold is specified
            if expansion_threshold is not None:
                ocod_df = ocod_df.loc[ocod_df['expansion_size'] < expansion_threshold]
            price_paid_df = pd.read_parquet(price_paid_file)

            # Filter for residential properties
            ocod_residential = ocod_df.loc[ocod_df[class_var]=='residential'].copy()
            
            # Skip if no residential properties
            if ocod_residential.empty:
                continue
            
            # Group and count by msoa11cd + grouping variables
            group_cols = ['msoa11cd'] + grouping_vars
            ocod_grouped = ocod_residential.groupby(group_cols).size().reset_index().rename(columns={0:'counts'})
            
            # Merge with price data
            df = price_paid_df.merge(ocod_grouped, on='msoa11cd')
            
            # Skip if no data after merge
            if df.empty:
                continue
            
            # Create date as first day of the month
            date = datetime(year, month, 1)
            
            # Group by grouping variables and calculate weighted/unweighted means
            # Use scalar for single grouping var, list for multiple to avoid tuple issues
            group_by_cols = grouping_vars[0] if len(grouping_vars) == 1 else grouping_vars
            
            for group_values, group_df in df.groupby(group_by_cols):
                if group_df.empty:
                    continue
                
                # Weighted mean (by offshore property counts)
                ocod_weighted_mean = np.average(group_df['price_mean'], weights=group_df['counts'])
                
                # Unweighted mean (treating all MSOAs equally)
                ocod_unweighted_mean = group_df['price_mean'].mean()
                
                # Calculate price ratio
                price_ratio = ocod_weighted_mean / ocod_unweighted_mean if ocod_unweighted_mean != 0 else np.nan
                
                # Create result dictionary
                result = {
                    'date': date,
                    'year': year,
                    'month': month,
                    'ocod_weighted_mean': ocod_weighted_mean,
                    'ocod_unweighted_mean': ocod_unweighted_mean,
                    'price_ratio': price_ratio
                }
                
                # Add grouping variables to result
                if len(grouping_vars) == 1:
                    result[grouping_vars[0]] = group_values
                else:
                    for i, var in enumerate(grouping_vars):
                        result[var] = group_values[i]
                
                monthly_results.append(result)
                
        except Exception as e:
            print(f"Error processing {ocod_file.name}: {str(e)}")
            continue

    # Create DataFrame from monthly results
    monthly_df = pd.DataFrame(monthly_results)
    
    if monthly_df.empty:
        return pd.DataFrame()
    
    # Sort by date and grouping variables
    sort_cols = ['date'] + grouping_vars
    monthly_df = monthly_df.sort_values(sort_cols).reset_index(drop=True)
    
    # Calculate mean differences (trends) for each group
    results = []
    
    # Use scalar for single grouping var, list for multiple to avoid tuple issues
    group_by_cols = grouping_vars[0] if len(grouping_vars) == 1 else grouping_vars
    
    for group_values, group_df in monthly_df.groupby(group_by_cols):
        if len(group_df) < 2:  # Need at least 2 points to calculate difference
            continue
            
        group_df = group_df.sort_values('date')
        
        # Calculate differences between consecutive months
        weighted_diff = group_df['ocod_weighted_mean'].diff().dropna()
        unweighted_diff = group_df['ocod_unweighted_mean'].diff().dropna()
        
        if len(weighted_diff) == 0:
            continue
        
        # Calculate mean difference (average trend)
        mean_weighted_diff = weighted_diff.mean()
        mean_unweighted_diff = unweighted_diff.mean()
        
        # Calculate price ratio statistics
        valid_ratios = group_df['price_ratio'].dropna()
        
        # Bootstrap test for price ratio
        bootstrap_results = bootstrap_ratio_test(valid_ratios)

        # Create result dictionary
        result = {
            'mean_weighted_difference': mean_weighted_diff,
            'mean_unweighted_difference': mean_unweighted_diff,
            'mean_price_ratio': bootstrap_results['mean_ratio'],
            'ratio_fraction_above_one': bootstrap_results['fraction_above_one'],
            'ratio_fraction_below_one': bootstrap_results['fraction_below_one'],
            'ratio_significantly_different': bootstrap_results['significantly_different'],
            'ratio_ci_lower': bootstrap_results['ci_lower'],
            'ratio_ci_upper': bootstrap_results['ci_upper'],
            'n_periods': len(weighted_diff),
            'n_ratio_observations': len(valid_ratios)
        }
        
        # Add grouping variables to result
        if len(grouping_vars) == 1:
            result[grouping_vars[0]] = group_values
        else:
            for i, var in enumerate(grouping_vars):
                result[var] = group_values[i]
        
        results.append(result)
    
    return pd.DataFrame(results)