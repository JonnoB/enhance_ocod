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
    """Helper function to add filename and date columns to a dataframe"""
    df = df.copy()
    df['filename'] = filename_stem
    
    # Parse date once from filename
    parts = filename_stem.split('_')
    date_str = f"{parts[-2]}-{parts[-1]}-01"
    df['date'] = pd.to_datetime(date_str)
    
    return df


def create_summarised_stats(list_of_files, class_var = 'class2'):
    # Initialize lists
    total_residential_df = []
    total_per_region_df = []
    total_incorp_df = []
    total_resi_lad_df = []

    for target_file in tqdm(list_of_files):
        target_year = pd.read_parquet(target_file)
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
    class_var = 'class2',
    ocod_path = '../data/ocod_history_processed', 
    price_paid_path = '../data/price_paid_msoa_averages' ):
    """
    Create time series with optional grouping variables and dwelling data
    
    Args:
        msoa_dwellings (pd.DataFrame): DataFrame with MSOA dwelling counts
        grouping_vars (list or None): List of column names to group by (e.g., ['region'])
    
    Returns:
        pandas.DataFrame: Time series data with aggregated statistics
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
            ocod_df, price_paid_df = _load_and_preprocess_data(ocod_file, price_paid_file, class_var)
            
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
    """Extract year and month from filename"""
    filename_parts = ocod_file.stem.split('_')
    return int(filename_parts[-2]), int(filename_parts[-1])

def _load_and_preprocess_data(ocod_file, price_paid_file, class_var = 'class2'):
    """Load and preprocess OCOD and price paid data"""
    ocod_df = pd.read_parquet(ocod_file)
    price_paid_df = pd.read_parquet(price_paid_file)
    
    # Filter for residential properties
    ocod_residential = ocod_df.loc[ocod_df[class_var] == 'residential'].copy()
    
    return ocod_residential, price_paid_df

def _aggregate_time_series_data(ocod_df, price_paid_df, msoa_dwellings, year, month, grouping_vars):
    """Aggregate time series data with optional grouping"""
    # Determine grouping columns
    group_cols = ['msoa11cd'] + (grouping_vars or [])
    
    # Group and count offshore properties
    ocod_grouped = ocod_df.groupby(group_cols).size().reset_index(name='ocod_total_counts')
    
    # Merge with price data and dwelling data
    df = price_paid_df.merge(ocod_grouped, on='msoa11cd', how='right')
    df = df.merge(msoa_dwellings, on='msoa11cd', how='left')
    df['ocod_total_counts'] = df['ocod_total_counts'].fillna(0).astype(int)
    
    # Create date
    date = datetime(year, month, 1)
    
    # Perform aggregation based on grouping
    return _calculate_aggregations(df, date, year, month, grouping_vars)

def _calculate_aggregations(df, date, year, month, grouping_vars):
    """Calculate aggregation metrics"""
    def calculate_metrics(group_df):
        """Calculate metrics for a group"""
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
    """Create and sort final DataFrame"""
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
    """
    Vectorized bootstrap test for whether mean ratio is significantly different from 1
    
    Args:
        ratios (array-like): Array of price ratios
        n_bootstrap (int): Number of bootstrap samples
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Contains fractions above/below 1, mean_ratio, confidence intervals, and significance
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
    class_var = 'class2'):
    """
    Calculate mean difference price trends for property groups.
    
    This function processes monthly property data to calculate weighted and 
    unweighted price trends for different geographical or categorical groups.
    It compares offshore company property prices with general market prices
    and performs statistical analysis on price ratios.
    
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
        
        # Create corresponding price_paid filename
        price_paid_file = price_paid_path / f'price_paid_{year}_{month:02d}.parquet'
        
        # Check if corresponding price_paid file exists
        if not price_paid_file.exists():
            continue
        
        try:
            # Read the data
            ocod_df = pd.read_parquet(ocod_file)
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