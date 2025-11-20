"""
compare_bez_cochlea_mainscript.py
Centralized data loading utilities for BEZ2018 and Cochlea model comparisons

⚠️ IMPORTANT - BEZ DATA LOADING:
---------------------------------
There are TWO functions for loading BEZ data:

1. load_bez_psth_data() - Loads psth_data_128fibers.mat
   ✅ Has individual runs (run dimension exists)
   ✅ Use for: All regression functions, run-by-run analysis
   
2. load_matlab_data() - Loads bez_acrossruns_psths.mat  
   ⚠️ NO individual runs (averaged across runs)
   ⚠️ Cannot use with: regress_cochlea_vs_bez_all_runs(), 
                       regress_all_bez_run_pairs(),
                       regress_bez_run_vs_run()

RECOMMENDATION: Use load_bez_psth_data() for regression analysis!

STRUCTURE:
----------
This module uses a hybrid design pattern with private engines and public API:

LEVEL 1: Core Utilities (Lines ~12-750)
    - convert_to_numeric()
    - load_matlab_data()
    - extract_parameters()
    - validate_psth_data()
    - values_match()
    - find_matching_value()
    - compute_single_regression()
    - extract_mean_rates_from_nested_dict()

LEVEL 2: Data Processing (Lines ~750-850)
    - match_data_points() - Fuzzy matching between datasets
    
LEVEL 3: Private Engines (Lines ~850-1100)
    - _get_num_runs() - Helper for counting runs
    - _regression_analysis() - Injected regression analysis
    - _correlation_analysis() - Injected correlation analysis
    - _batch_process_runs() - Generic multi-run processor
    - _analyze_by_groups() - Generic grouped analysis engine

LEVEL 4: Public API - Simple function names (Lines ~1100+)
    Single Comparisons:
        - regress_cochlea_vs_bez_single_run()
        - regress_bez_run_vs_run()
    
    Batch Comparisons:
        - regress_cochlea_vs_bez_all_runs()
        - regress_all_bez_run_pairs()
    
    Grouped Analysis:
        - regress_by_fiber_type()
        - regress_by_cf()
        - regress_by_db()
        - correlate_by_fiber_type()

USAGE EXAMPLES:
---------------
# Load BEZ data with individual runs (RECOMMENDED)
>>> bez_data, params = load_bez_psth_data()  # Has run dimension
>>> _, cochlea_data = load_matlab_data()     # Get cochlea only
>>> cochlea = mean_across_time_psth_cochlea(cochlea_data, params)
>>> bez = mean_across_time_psth_bez(bez_data, params)

# Basic comparison: Cochlea vs BEZ Run 0
>>> result = regress_cochlea_vs_bez_single_run(cochlea, bez, run_idx=0)
>>> print(f"Correlation: r = {result['r_value'].iloc[0]:.3f}")

# All runs at once
>>> all_runs = regress_cochlea_vs_bez_all_runs(cochlea, bez)
>>> print(f"Run 0: r = {all_runs[0]['r_value'].iloc[0]:.3f}")

# Group by fiber type
>>> by_fiber = regress_by_fiber_type(cochlea, bez, run2_idx=0)
>>> print(f"HSR: r = {by_fiber['hsr']['r_value'].iloc[0]:.3f}")
"""
import numpy as np
from scipy.io import loadmat
from pathlib import Path
from scipy.stats import linregress, pearsonr 
import pandas as pd


def convert_to_numeric(arr):
    """
    Convert MATLAB string arrays to numeric numpy arrays
    
    Parameters:
        arr: Input array (may be string, numeric, or MATLAB array)
    
    Returns:
        np.ndarray: Numeric array as float64
    """
    arr = np.atleast_1d(arr).flatten()
    if arr.dtype.kind in ['U', 'S']:  # Unicode or byte strings
        return np.array([float(str(x)) for x in arr])
    else:
        return arr.astype(float)
    
def load_matlab_data(bez_dir=None, cochlea_dir=None):
    """
    Load BEZ and Cochlea auditory nerve model data from MATLAB files
    
    Parameters:
        bez_dir (Path, optional): Directory containing BEZ PSTH data
        cochlea_dir (Path, optional): Directory containing Cochlea PSTH data
    
    Returns:
        tuple: (bez_data, cochlea_data) dictionaries
    
    Raises:
        FileNotFoundError: If required .mat files are not found
    """
    # Use defaults if not specified
    if bez_dir is None:
        bez_dir = Path("//subcorticalSTRF/BEZ2018_meanrate/results/processed_data")
    if cochlea_dir is None:
        cochlea_dir = Path("//subcorticalSTRF/cochlea_meanrate/out/condition_psths")
    
    # Load BEZ model data
    bez_file = bez_dir / 'bez_acrossruns_psths.mat'
    if not bez_file.exists():
        raise FileNotFoundError(f'BEZ model PSTH data not found at: {bez_file}')
    
    print(f"Loading BEZ data from: {bez_file}")
    try:
        bez_data = loadmat(str(bez_file), struct_as_record=False, squeeze_me=True)
        print("✓ BEZ data loaded successfully")
    except Exception as e:
        print(f"Error loading BEZ data: {e}")
        raise
    
    # Load cochlea model data
    cochlea_file = cochlea_dir / 'cochlea_psths.mat'
    if not cochlea_file.exists():
        raise FileNotFoundError(f'Cochlea model PSTH data not found at: {cochlea_file}')
    
    print(f"Loading cochlea data from: {cochlea_file}")
    try:
        cochlea_data = loadmat(str(cochlea_file), struct_as_record=False, squeeze_me=True)
        print("✓ Cochlea data loaded successfully")
    except Exception as e:
        print(f"Error loading cochlea data: {e}")
        raise
    
    return bez_data, cochlea_data



def extract_parameters(data, model_type="bez", verbose=True):
    """
    Extract experimental parameters from loaded MATLAB data
    
    Parameters:
        data (dict): Loaded MATLAB data structure
        model_type (str): Model identifier for logging ("bez" or "cochlea")
        verbose (bool): Print diagnostic information
    
    Returns:
        tuple: (cfs, frequencies, dbs) as sorted numpy arrays
            - cfs: Characteristic frequencies (Hz)
            - frequencies: Stimulus tone frequencies (Hz)
            - dbs: Sound pressure levels (dB SPL)
    """
    if verbose:
        print(f"Extracting parameters for {model_type} model...")
    
    cfs = convert_to_numeric(data['cfs'])
    frequencies = convert_to_numeric(data['frequencies'])
    dbs = convert_to_numeric(data['dbs'])
    
    if verbose:
        print(f"  CFs: {len(cfs)} values, range: {cfs.min():.1f} - {cfs.max():.1f} Hz")
        print(f"  Frequencies: {len(frequencies)} values, range: {frequencies.min():.1f} - {frequencies.max():.1f} Hz")
        print(f"  dB levels: {dbs}")
    
    return np.sort(cfs), np.sort(frequencies), np.sort(dbs)

def matlab_double_to_np_array(matlab_double_obj):
    """
    Convert MATLAB double object to numpy array
    
    Parameters:
        matlab_double_obj: MATLAB double object from matlab.engine
    
    Returns:
        np.ndarray: Squeezed numpy array
    """
    np_array = np.array(matlab_double_obj).squeeze()
    return np_array

def validate_psth_data(psth_data, expected_length=200, fiber_type=None, 
                      cf_val=None, freq_val=None, db_val=None, run_idx=None):
    """
    Validate PSTH data array for convolution.

    Parameters:
    - psth_data: PSTH array to validate
    - expected_length: Expected length of PSTH (default: 200 for 200ms)
    - fiber_type, cf_val, freq_val, db_val, run_idx: Parameters for error reporting

    Returns:
    - is_valid: Boolean indicating if data is valid
    - processed_data: Processed PSTH array (flattened and cleaned)
    """
    try:
        if psth_data is None:
            return False, None

        # Convert to numpy array and flatten
        if isinstance(psth_data, np.ndarray):
            psth_array = np.atleast_1d(psth_data).flatten()
        else:
            psth_array = np.array(psth_data).flatten()

        # Check length
        if len(psth_array) != expected_length:
            if any(x is not None for x in [fiber_type, cf_val, freq_val, db_val, run_idx]):
                print(f"Warning: Unexpected PSTH length {len(psth_array)} (expected {expected_length}) "
                      f"for {fiber_type or 'unknown'} CF={cf_val}Hz, Freq={freq_val}Hz, "
                      f"dB={db_val}, Run={run_idx}")
            return False, None

        # Check for finite values
        if not np.any(np.isfinite(psth_array)):
            if any(x is not None for x in [fiber_type, cf_val, freq_val, db_val, run_idx]):
                print(f"Warning: No finite values for {fiber_type or 'unknown'} "
                      f"CF={cf_val}Hz, Freq={freq_val}Hz, dB={db_val}, Run={run_idx}")
            return False, None

        return True, psth_array

    except Exception as e:
        if any(x is not None for x in [fiber_type, cf_val, freq_val, db_val, run_idx]):
            print(f"Warning: Error validating PSTH data for {fiber_type or 'unknown'} "
                  f"CF={cf_val}Hz, Freq={freq_val}Hz, dB={db_val}, Run={run_idx}: {e}")
        return False, None
    

def load_bez_psth_data():
    """
    Load BEZ model PSTH data from MATLAB file psth_data_128fibers.mat.
    The data contains hsr_all, lsr_all, msr_all as 4D cell arrays.
    This will give me individual run PSTHs for each fiber type.
    Returns:
    - bez_data: Dictionary containing PSTH data for all fiber types
    - parameters: Dictionary with CFs, frequencies, and dB levels
    """
    # Load BEZ model data from psth_data_128fibers.mat
    # Use proper path resolution
    current_dir = Path(__file__).parent
    
    # Try multiple possible locations
    possible_paths = [
        # Relative to convolution_HRF directory
        current_dir.parent / 'BEZ2018_meanrate' / 'results' / 
        'processed_data' / 'psth_data_128fibers.mat',
        
        # Absolute path from project root
        Path('/home/ekim/PycharmProjects/phd_firstyear/subcorticalSTRF/BEZ2018_meanrate/results/processed_data/psth_data_128fibers.mat'),
        
        # Try from audio_periph_models
        Path('/home/ekim/audio_periph_models/BEZ2018a_model/results/processed_data/psth_data_128fibers.mat'),
    ]
    
    # Find the first existing path
    bez_file = None
    for path in possible_paths:
        if path.exists():
            bez_file = path
            break
    
    if bez_file is None:
        raise FileNotFoundError(
            f"Could not find psth_data_128fibers.mat in any of:\n" +
            '\n'.join(f"  - {p}" for p in possible_paths) +
            "\n\nPlease check the file location."
        )

    print(f"Loading BEZ PSTH data from: {bez_file}")
    bez_data = loadmat(
        str(bez_file), struct_as_record=False, squeeze_me=True
    )
    print("✓ BEZ PSTH data loaded successfully")

    # Extract parameters

    parameters = {
        'cfs': np.sort(convert_to_numeric(bez_data['cfs'])),
        'frequencies': np.sort(convert_to_numeric(bez_data['frequencies'])),
        'dbs': np.sort(convert_to_numeric(bez_data['dbs']))
    }

    print(f"Parameters extracted:")
    print(f"  CFs: {len(parameters['cfs'])} values")
    print(f"  Frequencies: {len(parameters['frequencies'])} values")
    print(f"  dB levels: {parameters['dbs']}")

    # Verify the data structure
    print(f"Data structure:")
    print(f"  hsr_all shape: {bez_data['hsr_all'].shape}")
    print(f"  msr_all shape: {bez_data['msr_all'].shape}")
    print(f"  lsr_all shape: {bez_data['lsr_all'].shape}")

    return bez_data, parameters


def _compute_mean_rate_single_condition(psth_cell, fiber_type, cf_val, freq_val, 
                                        db_val, run_idx, expected_length=200):
    """
    Helper function to compute mean rate for a single PSTH condition.
    
    Parameters:
    -----------
    psth_cell : array-like
        PSTH data for a single condition
    fiber_type, cf_val, freq_val, db_val, run_idx : 
        Parameters for error reporting
    expected_length : int
        Expected PSTH length (default: 200)
    
    Returns:
    --------
    float : Mean firing rate, or np.nan if invalid
    """
    try:
        # Validate PSTH data
        is_valid, psth_array = validate_psth_data(
            psth_cell, expected_length=expected_length,
            fiber_type=fiber_type, cf_val=cf_val,
            freq_val=freq_val, db_val=db_val, run_idx=run_idx
        )
        
        if is_valid:
            return np.mean(psth_array)
        else:
            return np.nan
            
    except Exception as e:
        run_str = f"Run={run_idx}" if run_idx is not None else "single instance"
        print(f"Error processing PSTH for {fiber_type} CF={cf_val}Hz, "
              f"Freq={freq_val}Hz, dB={db_val}, {run_str}: {e}")
        return np.nan


def mean_across_time_psth(psth_data, parameters, fiber_types=['hsr', 'msr', 'lsr'], 
                          has_runs=True, model_name="unknown"):
    """
    Compute mean firing rate across time for PSTH data.
    
    Handles both multi-run data (BEZ) and single-instance data (Cochlea).
    This function uses iterate_over_psth_conditions with operation_compute_mean_rate.
    
    Parameters:
    -----------
    psth_data : dict
        Dictionary with keys 'lsr_all', 'msr_all', 'hsr_all' containing PSTH arrays
    parameters : dict
        Dictionary with 'cfs', 'frequencies', 'dbs' arrays
    fiber_types : list
        List of fiber types to process (default: ['hsr', 'msr', 'lsr'])
    has_runs : bool
        If True, expects 4D arrays [CF, Freq, dB, Run]
        If False, expects 3D arrays [CF, Freq, dB] (default: True)
    model_name : str
        Name of model for logging purposes
    
    Returns:
    --------
    mean_psth : dict
        Nested dictionary structure:
        - With runs: [fiber_type][cf][freq][db][run] = mean_rate
        - Without runs: [fiber_type][cf][freq][db] = mean_rate
    """
    # Use the higher-order function with the mean rate operation
    return iterate_over_psth_conditions(
        psth_data, parameters, operation_compute_mean_rate,
        fiber_types=fiber_types, has_runs=has_runs, model_name=model_name
    )


def mean_across_time_psth_bez(bez_data, parameters, fiber_types=['hsr', 'msr', 'lsr']):
    """
    Compute mean PSTH across time for BEZ model data (with multiple runs).
    
    Parameters:
    -----------
    bez_data : dict
        BEZ model PSTH data with 4D arrays [CF, Freq, dB, Run]
    parameters : dict
        Dictionary with 'cfs', 'frequencies', 'dbs' arrays
    fiber_types : list
        List of fiber types to process
    
    Returns:
    --------
    mean_psth : dict
        Nested dictionary [fiber_type][cf][freq][db][run] = mean_rate
    """
    return mean_across_time_psth(
        bez_data, parameters, fiber_types, 
        has_runs=True, model_name="BEZ"
    )


def mean_across_time_psth_cochlea(cochlea_data, parameters, fiber_types=['hsr', 'msr', 'lsr']):
    """
    Compute mean PSTH across time for Cochlea model data (single instance per condition).
    
    Parameters:
    -----------
    cochlea_data : dict
        Cochlea model PSTH data with 3D arrays [CF, Freq, dB]
    parameters : dict
        Dictionary with 'cfs', 'frequencies', 'dbs' arrays
    fiber_types : list
        List of fiber types to process
    
    Returns:
    --------
    mean_psth : dict
        Nested dictionary [fiber_type][cf][freq][db] = mean_rate
    """
    return mean_across_time_psth(
        cochlea_data, parameters, fiber_types,
        has_runs=False, model_name="Cochlea"
    )


### These are the new functions, added after the initial ones above ### 
### November 5th, 2025 
### Higher-order functions to pass custom operations for PSTH processing
###
def iterate_over_psth_conditions(psth_data, parameters, operation_func,
                                 fiber_types=['hsr', 'msr', 'lsr'],
                                   has_runs=True, model_name="unknown"):
    """ 
    Generic iterator over all PSTH conditions, applies a custom operation function to each PSTH.

    This is a "higer-order function" it takes another function as an argument
    and calls it for each PSTH condition.

    Parameters:
    -----------
    psth_data : dict
        Dictionary with keys 'lsr_all', 'msr_all', 'hsr_all' containing PSTH arrays
    parameters : dict
        Dictionary with 'cfs', 'frequencies', 'dbs' arrays
    operation_func : callable
        FUnction to apple to each PSTH.
        Must accept: (psth_array, fiber_type, cf_val, freq_val, db_val, run_idx)
        Must return a single value (float, array, etc.)
    fiber_types : list
        List of fiber types to process (default: ['hsr', 'msr', 'lsr'])
    has_runs : bool
        If True, expects 4D arrays [CF, Freq, dB, Run]
        If False, expects 3D arrays [CF, Freq, dB] (default: True)
    model_name : str
        Name of model for logging purposes
        (default: "unknown")
    
    Returns:
    --------
    results : dict
        Nested dictionary structure:
        - With runs: [fiber_type][cf][freq][db][run] = operation_result
        - Without runs: [fiber_type][cf][freq][db] = operation_result

    Example:
    --------
    >>> def my_operation(psth, fiber_type, cf_val, freq_val, db_val, run_idx):
    ...     return np.mean(psth)  # Example operation
    >>>
    >>> results = iterate_over_psth_conditions(data, params, my_operation)

    """

    # Get dimensions from one of the fiber type arrays
    sample_data = psth_data[fiber_types[0] + '_all']

    if has_runs:
        n_cfs, n_freqs, n_dbs, n_runs = sample_data.shape
        print(f"{model_name} data dimensions: CFs={n_cfs}, Freqs={n_freqs}, "
              f"dBs={n_dbs}, Runs={n_runs}")
    else:
        n_cfs, n_freqs, n_dbs = sample_data.shape
        n_runs = 1 # Treat as single "run"
        print(f"{model_name} data dimensions: CFs={n_cfs}, Freqs={n_freqs}, "
              f"dBs={n_dbs}")
    
    # Extract PSTH data for each fiber type
    fiber_data = {ft: psth_data[f'{ft}_all'] for ft in fiber_types}

    # Iterate and apply operation
    results = {}

    for fiber_type in fiber_types:
        print(f" Processing {model_name} {fiber_type} fibers...")
        results[fiber_type] = {}
        psth_array = fiber_data[fiber_type]
        # Process each CF
        for cf_idx, cf_val in enumerate(parameters['cfs']):
            results[fiber_type][cf_val] = {}
            # Process each tone frequency
            for freq_idx, freq_val in enumerate(parameters['frequencies']):
                results[fiber_type][cf_val][freq_val] = {}
                # Process each dB level
                for db_idx, db_val in enumerate(parameters['dbs']):

                    if has_runs:
                        # Multi-run data: create dict for runs 
                        results[fiber_type][cf_val][freq_val][db_val] = {}

                        for run_idx in range(n_runs):
                            psth = psth_array[cf_idx, freq_idx, db_idx, run_idx]

                            # CALL THE CUSTOM OPERATION
                            result = operation_func(
                                psth, fiber_type, cf_val,
                                freq_val, db_val, run_idx
                            )
                            results[fiber_type][cf_val][freq_val][db_val][run_idx] = result
                    else:
                        # Single-instance data: store value directly
                        psth = psth_array[cf_idx, freq_idx, db_idx]

                        # CALL THE CUSTOM OPERATION
                        result = operation_func(
                            psth, fiber_type, cf_val,
                            freq_val, db_val, None
                        )
                        results[fiber_type][cf_val][freq_val][db_val] = result
    return results

# ============================================================================
# OPERATION FUNCTIONS
# ============================================================================

def operation_compute_mean_rate(psth, fiber_type, cf_val, freq_val, db_val, run_idx):
    """
    Operation: Compute mean firing rate from PSTH data.
    This is a simple operation function that can be passed to iterate_over_psth_conditions.
    """

    is_valid, psth_array = validate_psth_data(
        psth, expected_length=200,
        fiber_type=fiber_type, cf_val=cf_val,
        freq_val=freq_val, db_val=db_val, run_idx=run_idx
    )

    if is_valid:
        return np.mean(psth_array)
    else:
        return np.nan
    
# ==========================================================================
# LINEAR REGRESSION AND CORRELATION UTILITIES
# ==========================================================================

def values_match(val1, val2, rtol=1e-5, atol=1e-8):
    """
    Check if two floating-point values match within tolerance.
    
    Uses numpy's isclose() logic: |val1 - val2| <= (atol + rtol * |val2|)
    
    Parameters:
    -----------
    val1, val2 : float
        Values to compare
    rtol : float
        Relative tolerance (default: 1e-5, means 0.001% difference allowed)
    atol : float
        Absolute tolerance (default: 1e-8)
    
    Returns:
    --------
    bool : True if values match within tolerance
    
    Examples:
    ---------
    >>> values_match(125.0, 125.00000001)  # True
    >>> values_match(125.0, 125.1)         # False
    >>> values_match(0.0, 1e-9)            # True (within absolute tolerance)
    """
    return np.isclose(val1, val2, rtol=rtol, atol=atol)

def find_matching_value(target, value_list, rtol=1e-5, atol=1e-8):
    """
    Find the first value in value_list that matches target within tolerance.

    This is useful when you have slightly different floating-point values of
    the same conceptual value (e.g., characteristic frequency).

    Parameters:
    -----------
    target : float
        Target value to match
    value_list : list of float
        List of values to search
    rtol : float
        Relative tolerance (default: 1e-5)
    atol : float
        Absolute tolerance (default: 1e-8)

    Returns:
    --------
    float : The first matching value from value_list or None if not found
    """
    for val in value_list:
        if values_match(target, val, rtol=rtol, atol=atol):
            return val
    return None
def compute_single_regression(x_data, y_data, condition_info=None):
    """
    Compute linear regression between two arrays.
    
    This is the ATOMIC function - does one regression calculation.
    All other functions build on this.
    
    Parameters:
    -----------
    x_data : array-like
        X values (e.g., BEZ mean rates for one run)
    y_data : array-like
        Y values (e.g., Cochlea mean rates)
    condition_info : dict, optional
        Information about the condition (for labeling results)
        Keys: 'fiber_type', 'cf', 'freq', 'db', 'run_idx', etc.
    
    Returns:
    --------
    results : dict
        Dictionary with regression statistics:
        - 'slope': Regression slope
        - 'intercept': Y-intercept
        - 'r_value': Correlation coefficient
        - 'r_squared': R² value
        - 'p_value': Statistical significance
        - 'std_err': Standard error of estimate
        - 'n_points': Number of data points
        - 'condition': condition_info dict (if provided)
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean = np.array(x_data)[valid_mask]
    y_clean = np.array(y_data)[valid_mask]
    
    if len(x_clean) < 2:
        # Not enough points for regression
        return {
            'slope': np.nan,
            'intercept': np.nan,
            'r_value': np.nan,
            'r_squared': np.nan,
            'p_value': np.nan,
            'std_err': np.nan,
            'n_points': len(x_clean),
            'condition': condition_info
        }
    
    # Compute regression
    slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'std_err': std_err,
        'n_points': len(x_clean),
        'condition': condition_info
    }

def extract_data_at_db(mean_psth_dict, db_level, fiber_types=None, rtol=1e-5, atol=1e-8):
    """
    Extract all data at a specific dB level across all fiber types, CFs, and frequencies.
    
    This is useful for analyzing responses at a single sound pressure level.
    
    Parameters:
    -----------
    mean_psth_dict : dict
        Nested dictionary [fiber_type][cf][freq][db][run] = mean_rate
    db_level : float
        The dB SPL level to extract (e.g., 50, 60, 70, 80)
    fiber_types : list, optional
        Which fiber types to include (default: ['hsr', 'msr', 'lsr'])
    rtol, atol : float
        Fuzzy matching tolerances
    
    Returns:
    --------
    db_data : dict
        Nested dictionary [fiber_type][cf][freq][run] = mean_rate
        (dB dimension removed, only data at specified dB level)
    """
    if fiber_types is None:
        fiber_types = ['hsr', 'msr', 'lsr']
    
    db_data = {}
    
    for fiber_type in fiber_types:
        if fiber_type not in mean_psth_dict:
            continue
            
        db_data[fiber_type] = {}
        fiber_dict = mean_psth_dict[fiber_type]
        
        for cf, cf_data in fiber_dict.items():
            db_data[fiber_type][cf] = {}
            
            for freq, freq_data in cf_data.items():
                # Find matching dB level using fuzzy matching
                matched_db = find_matching_value(db_level, freq_data.keys(), rtol, atol)
                
                if matched_db is not None:
                    # Copy the data at this dB level
                    db_data[fiber_type][cf][freq] = freq_data[matched_db]
    
    return db_data


def extract_mean_rates_from_nested_dict(mean_psth_dict, fiber_type, cf=None, 
                                        freq=None, db=None, run_idx=None,
                                        rtol=1e-5, atol=1e-8):
    """
    Extract mean rates from nested dictionary structure with optional filtering.
    
    Uses FUZZY MATCHING for floating-point comparisons to handle precision issues.
    
    Parameters:
    -----------
    mean_psth_dict : dict
        Nested dictionary [fiber_type][cf][freq][db][run] = mean_rate
        or [fiber_type][cf][freq][db] = mean_rate (without runs)
    fiber_type : str
        Which fiber type to extract ('hsr', 'msr', 'lsr')
    cf, freq, db, run_idx : optional
        If provided, filter to specific values (uses fuzzy matching for floats)
        If None, include all values for that dimension
    rtol, atol : float
        Relative and absolute tolerances for fuzzy matching
    
    Returns:
    --------
    rates : list or float
        List of mean rates matching the filters
    metadata : list of dicts
        Metadata for each rate (cf, freq, db, run_idx values)
    """
    rates = []
    metadata = []
    
    fiber_data = mean_psth_dict[fiber_type]
    
    # Iterate through all conditions
    for cf_key, cf_data in fiber_data.items():
        # Fuzzy match for CF
        if cf is not None and not values_match(cf_key, cf, rtol=rtol, atol=atol):
            continue
            
        for freq_key, freq_data in cf_data.items():
            # Fuzzy match for frequency
            if freq is not None and not values_match(freq_key, freq, rtol=rtol, atol=atol):
                continue
                
            for db_key, db_data in freq_data.items():
                # Fuzzy match for dB
                if db is not None and not values_match(db_key, db, rtol=rtol, atol=atol):
                    continue
                
                # Check if data has runs (dict) or is single value (float)
                if isinstance(db_data, dict):
                    # Has runs
                    for run_key, rate_value in db_data.items():
                        if run_idx is not None and run_key != run_idx:
                            continue
                        
                        rates.append(rate_value)
                        metadata.append({
                            'fiber_type': fiber_type,
                            'cf': cf_key,
                            'freq': freq_key,
                            'db': db_key,
                            'run_idx': run_key
                        })
                else:
                    # Single value (no runs)
                    if run_idx is None:  # Only include if not filtering by run
                        rates.append(db_data)
                        metadata.append({
                            'fiber_type': fiber_type,
                            'cf': cf_key,
                            'freq': freq_key,
                            'db': db_key,
                            'run_idx': None
                        })
    
    return rates, metadata


# ===================================================================
# LEVEL 2: DATA MATCHING AND PREPARATION
# ===================================================================

def match_data_points(data1_means, data2_means, fiber_type=None, cf=None, 
                      freq=None, db=None, run1_idx=None, run2_idx=None,
                      rtol=1e-5, atol=1e-8, verbose=False):
    """
    Match data points between two datasets using fuzzy matching
    
    Extracts rates from both datasets and pairs them based on matching
    experimental conditions (CF, frequency, dB). Handles floating-point
    precision differences using configurable tolerances.
    
    Parameters:
        data1_means: Nested dict from mean_across_time_psth (first dataset)
        data2_means: Nested dict from mean_across_time_psth (second dataset)
        fiber_type: Filter by fiber type ('hsr', 'msr', 'lsr') or None for all
        cf, freq, db: Filter by specific values or None for all
        run1_idx: Run index for data1 (None if data1 has no runs)
        run2_idx: Run index for data2 (None if data2 has no runs)
        rtol: Relative tolerance for fuzzy matching (default: 1e-5 = 0.001%)
        atol: Absolute tolerance for fuzzy matching (default: 1e-8)
        verbose: Print diagnostic information
    
    Returns:
        tuple: (matched_rates, matched_metadata)
            matched_rates: List of dicts with 'data1_rate' and 'data2_rate'
            matched_metadata: List of dicts with condition info for each match
    
    Example:
        >>> cochlea = mean_across_time_psth_cochlea(...)
        >>> bez = mean_across_time_psth_bez(...)
        >>> rates, meta = match_data_points(cochlea, bez, fiber_type='hsr')
        >>> print(f"Matched {len(rates)} HSR data points")
    """
    # Extract rates from both datasets
    rates1, meta1 = extract_mean_rates_from_nested_dict(
        data1_means, fiber_type, cf, freq, db, run1_idx, rtol, atol
    )
    rates2, meta2 = extract_mean_rates_from_nested_dict(
        data2_means, fiber_type, cf, freq, db, run2_idx, rtol, atol
    )
    
    if verbose:
        print(f"Data1: {len(rates1)} points, Data2: {len(rates2)} points")
    
    # Match points based on conditions
    matched_rates = []
    matched_metadata = []
    
    for i, meta_i in enumerate(meta1):
        for j, meta_j in enumerate(meta2):
            # Check if conditions match (with fuzzy matching)
            if (meta_i['fiber_type'] == meta_j['fiber_type'] and
                values_match(meta_i['cf'], meta_j['cf'], rtol, atol) and
                values_match(meta_i['freq'], meta_j['freq'], rtol, atol) and
                values_match(meta_i['db'], meta_j['db'], rtol, atol)):
                
                matched_rates.append({
                    'data1_rate': rates1[i],
                    'data2_rate': rates2[j]
                })
                matched_metadata.append({
                    'fiber_type': meta_i['fiber_type'],
                    'cf': meta_i['cf'],
                    'freq': meta_i['freq'],
                    'db': meta_i['db'],
                    'run1_idx': meta_i['run_idx'],
                    'run2_idx': meta_j['run_idx']
                })
    
    if verbose:
        print(f"Matched {len(matched_rates)} data points")
    
    return matched_rates, matched_metadata


# ===================================================================
# PRIVATE ENGINES - Flexible core implementations
# ===================================================================

def _get_num_runs(nested_dict):
    """
    Helper: Determine number of runs in a nested dict
    
    Assumes structure: nested_dict[fiber_type][cf][freq][db][run_idx]
    """
    # Navigate to the deepest level to find runs
    for fiber_type, cf_data in nested_dict.items():
        for cf, freq_data in cf_data.items():
            for freq, db_data in freq_data.items():
                for db, run_data in db_data.items():
                    if isinstance(run_data, dict):
                        return len(run_data)
    return 0  # No runs found


def _regression_analysis(matched_rates, matched_metadata):
    """
    PRIVATE: Perform linear regression and correlation on matched data
    
    Parameters:
        matched_rates: List of dicts with 'data1_rate' and 'data2_rate'
        matched_metadata: List of dicts with condition information
    
    Returns:
        pd.DataFrame: Single row with regression statistics
    """
    if len(matched_rates) == 0:
        return pd.DataFrame({
            'r_value': [np.nan],
            'slope': [np.nan],
            'intercept': [np.nan],
            'p_value': [np.nan],
            'stderr': [np.nan],
            'n_points': [0]
        })
    
    x_data = np.array([pt['data1_rate'] for pt in matched_rates])
    y_data = np.array([pt['data2_rate'] for pt in matched_rates])
    
    # Compute both regression and correlation
    slope, intercept, r_value, p_value, stderr = linregress(x_data, y_data)
    r_corr, p_corr = pearsonr(x_data, y_data)
    
    return pd.DataFrame({
        'r_value': [r_value],
        'r_pearson': [r_corr],  # Pearson correlation (should match r_value)
        'slope': [slope],
        'intercept': [intercept],
        'p_value': [p_value],
        'p_pearson': [p_corr],
        'stderr': [stderr],
        'n_points': [len(x_data)]
    })


def _correlation_analysis(matched_rates, matched_metadata):
    """
    PRIVATE: Perform correlation analysis only (no regression)
    
    Parameters:
        matched_rates: List of dicts with 'data1_rate' and 'data2_rate'
        matched_metadata: List of dicts with condition information
    
    Returns:
        pd.DataFrame: Single row with correlation statistics
    """
    if len(matched_rates) == 0:
        return pd.DataFrame({
            'r_value': [np.nan],
            'p_value': [np.nan],
            'n_points': [0]
        })
    
    x_data = np.array([pt['data1_rate'] for pt in matched_rates])
    y_data = np.array([pt['data2_rate'] for pt in matched_rates])
    
    r_value, p_value = pearsonr(x_data, y_data)
    
    return pd.DataFrame({
        'r_value': [r_value],
        'p_value': [p_value],
        'n_points': [len(x_data)]
    })


def _batch_process_runs(data1_means, data2_means, run1_idx, run2_func,
                        rtol=1e-5, atol=1e-8, verbose=True):
    """
    PRIVATE: Generic engine for processing multiple runs
    
    This flexible core processes all runs from data2 against a single run
    (or all data) from data1.
    
    Parameters:
        data1_means: First dataset (nested dict)
        data2_means: Second dataset with runs (nested dict)
        run1_idx: Run index for data1 (None if no runs in data1)
        run2_func: Function to call for each run2
            Signature: func(data1, data2, run1_idx, run2_idx, rtol, atol) -> DataFrame
        rtol, atol: Fuzzy matching tolerances
        verbose: Print progress messages
    
    Returns:
        dict: {run_idx: DataFrame} for each run in data2
    """
    num_runs = _get_num_runs(data2_means)
    
    if num_runs == 0:
        if verbose:
            print("Warning: No runs found in data2")
        return {}
    
    results = {}
    
    if verbose:
        print(f"Processing {num_runs} runs...")
    
    for run_idx in range(num_runs):
        if verbose:
            print(f"  Run {run_idx}...", end=' ')
        
        try:
            df = run2_func(data1_means, data2_means, run1_idx, run_idx, rtol, atol)
            results[run_idx] = df
            
            if verbose and 'r_value' in df.columns:
                r_val = df['r_value'].iloc[0]
                n_pts = df['n_points'].iloc[0]
                print(f"✓ r = {r_val:.3f}, n = {int(n_pts)}")
            elif verbose:
                print("✓")
                
        except Exception as e:
            if verbose:
                print(f"✗ Error: {e}")
            results[run_idx] = None
    
    return results


def _analyze_by_groups(data1_means, data2_means, groupby, run1_idx=None, 
                       run2_idx=None, analysis_func=None, rtol=1e-5, 
                       atol=1e-8, verbose=True):
    """
    PRIVATE: Generic engine for grouped analysis
    
    Performs separate analyses for each unique value in the groupby field(s).
    
    Parameters:
        data1_means: First dataset (nested dict)
        data2_means: Second dataset (nested dict)
        groupby: Field to group by ('fiber_type', 'cf', 'freq', or 'db')
        run1_idx: Run index for data1 (None if no runs)
        run2_idx: Run index for data2 (None if no runs)
        analysis_func: Function to analyze each group
            Signature: func(matched_rates, matched_metadata) -> DataFrame
            If None, uses _regression_analysis
        rtol, atol: Fuzzy matching tolerances
        verbose: Print progress
    
    Returns:
        dict: {group_value: DataFrame} for each unique group value
    """
    if analysis_func is None:
        analysis_func = _regression_analysis
    
    # Determine unique values for the groupby field
    unique_values = set()
    
    # Extract from data1
    for fiber_type, cf_data in data1_means.items():
        if groupby == 'fiber_type':
            unique_values.add(fiber_type)
            continue
        for cf, freq_data in cf_data.items():
            if groupby == 'cf':
                unique_values.add(cf)
                continue
            for freq, db_data in freq_data.items():
                if groupby == 'freq':
                    unique_values.add(freq)
                    continue
                for db, run_data in db_data.items():
                    if groupby == 'db':
                        unique_values.add(db)
    
    unique_values = sorted(unique_values)
    
    if verbose:
        print(f"Analyzing by {groupby}: {len(unique_values)} groups")
    
    results = {}
    
    for group_val in unique_values:
        if verbose:
            print(f"  {groupby} = {group_val}...", end=' ')
        
        try:
            # Set filter based on groupby
            filters = {
                'fiber_type': group_val if groupby == 'fiber_type' else None,
                'cf': group_val if groupby == 'cf' else None,
                'freq': group_val if groupby == 'freq' else None,
                'db': group_val if groupby == 'db' else None,
                'run1_idx': run1_idx,
                'run2_idx': run2_idx,
                'rtol': rtol,
                'atol': atol
            }
            
            # Match data for this group
            matched_rates, matched_meta = match_data_points(
                data1_means, data2_means, **filters
            )
            
            # Analyze this group
            df = analysis_func(matched_rates, matched_meta)
            results[group_val] = df
            
            if verbose and 'r_value' in df.columns:
                r_val = df['r_value'].iloc[0]
                n_pts = df['n_points'].iloc[0]
                print(f"✓ r = {r_val:.3f}, n = {int(n_pts)}")
            elif verbose:
                print("✓")
                
        except Exception as e:
            if verbose:
                print(f"✗ Error: {e}")
            results[group_val] = None
    
    return results


# ===================================================================
# PUBLIC API - Simple, clear function names (LEVEL 3 & 4)
# ===================================================================

def regress_cochlea_vs_bez_single_run(cochlea_means, bez_means, run_idx=0,
                                       rtol=1e-5, atol=1e-8, verbose=False):
    """
    Compare Cochlea model to a single BEZ run using linear regression
    
    Matches all data points (across all CFs, frequencies, dBs, fiber types)
    between Cochlea and one BEZ run, then performs linear regression.
    
    Parameters:
        cochlea_means: Nested dict from mean_across_time_psth_cochlea()
        bez_means: Nested dict from mean_across_time_psth_bez()
        run_idx: Which BEZ run to compare (default: 0)
        rtol: Relative tolerance for fuzzy matching (default: 1e-5 = 0.001%)
        atol: Absolute tolerance for fuzzy matching (default: 1e-8)
        verbose: Print diagnostic information
    
    Returns:
        pd.DataFrame: Single row with regression statistics
            Columns: r_value, r_pearson, slope, intercept, p_value, 
                    p_pearson, stderr, n_points
    
    Example:
        >>> cochlea = mean_across_time_psth_cochlea(cochlea_data, params)
        >>> bez = mean_across_time_psth_bez(bez_data, params)
        >>> result = regress_cochlea_vs_bez_single_run(cochlea, bez, run_idx=0)
        >>> print(f"Correlation: r = {result['r_value'].iloc[0]:.3f}")
        >>> print(f"Regression: y = {result['slope'].iloc[0]:.2f}x + {result['intercept'].iloc[0]:.2f}")
    """
    matched_rates, matched_meta = match_data_points(
        cochlea_means, bez_means,
        run1_idx=None,  # Cochlea has no runs
        run2_idx=run_idx,
        rtol=rtol,
        atol=atol,
        verbose=verbose
    )
    
    return _regression_analysis(matched_rates, matched_meta)


def regress_cochlea_vs_bez_all_runs(cochlea_means, bez_means, rtol=1e-5, 
                                      atol=1e-8, verbose=True):
    """
    Compare Cochlea model to all BEZ runs using linear regression
    
    Convenience wrapper that processes all BEZ runs automatically.
    Under the hood, calls regress_cochlea_vs_bez_single_run() for each run.
    
    Parameters:
        cochlea_means: Nested dict from mean_across_time_psth_cochlea()
        bez_means: Nested dict from mean_across_time_psth_bez()
        rtol: Relative tolerance for fuzzy matching (default: 1e-5)
        atol: Absolute tolerance for fuzzy matching (default: 1e-8)
        verbose: Print progress messages
    
    Returns:
        dict: {run_idx: DataFrame} containing regression results for each run
              Each DataFrame has columns: r_value, slope, intercept, p_value, etc.
    
    Example:
        >>> results = regress_cochlea_vs_bez_all_runs(cochlea, bez)
        Processing 10 runs...
          Run 0... ✓ r = 0.943, n = 432
          Run 1... ✓ r = 0.938, n = 432
          ...
        >>> print(f"Run 0: r = {results[0]['r_value'].iloc[0]:.3f}")
    """
    def process_single_run(data1, data2, run1_idx, run2_idx, rtol, atol):
        return regress_cochlea_vs_bez_single_run(data1, data2, run2_idx, rtol, atol)
    
    return _batch_process_runs(
        cochlea_means, bez_means, 
        run1_idx=None,
        run2_func=process_single_run,
        rtol=rtol, 
        atol=atol, 
        verbose=verbose
    )


def regress_bez_run_vs_run(bez_means, run_i, run_j, rtol=1e-5, atol=1e-8, verbose=False):
    """
    Compare two BEZ runs to each other using linear regression
    
    Assesses run-to-run consistency within the BEZ model.
    
    Parameters:
        bez_means: Nested dict from mean_across_time_psth_bez()
        run_i: First run index
        run_j: Second run index
        rtol: Relative tolerance for fuzzy matching (default: 1e-5)
        atol: Absolute tolerance for fuzzy matching (default: 1e-8)
        verbose: Print diagnostic information
    
    Returns:
        pd.DataFrame: Single row with regression statistics
    
    Example:
        >>> bez = mean_across_time_psth_bez(bez_data, params)
        >>> result = regress_bez_run_vs_run(bez, run_i=0, run_j=1)
        >>> print(f"Run 0 vs Run 1: r = {result['r_value'].iloc[0]:.3f}")
    """
    matched_rates, matched_meta = match_data_points(
        bez_means, bez_means,
        run1_idx=run_i,
        run2_idx=run_j,
        rtol=rtol,
        atol=atol,
        verbose=verbose
    )
    
    return _regression_analysis(matched_rates, matched_meta)


def regress_all_bez_run_pairs(bez_means, rtol=1e-5, atol=1e-8, verbose=True):
    """
    Compare all pairs of BEZ runs to assess run-to-run consistency
    
    Computes regression between every combination of BEZ runs.
    For 10 runs, this generates 45 comparisons: (0,1), (0,2), ..., (8,9)
    
    Parameters:
        bez_means: Nested dict from mean_across_time_psth_bez()
        rtol: Relative tolerance for fuzzy matching (default: 1e-5)
        atol: Absolute tolerance for fuzzy matching (default: 1e-8)
        verbose: Print progress
    
    Returns:
        dict: {(run_i, run_j): DataFrame} for each run pair
              Keys are tuples like (0, 1), (0, 2), (1, 2), etc.
    
    Example:
        >>> bez = mean_across_time_psth_bez(bez_data, params)
        >>> pairs = regress_all_bez_run_pairs(bez)
        Comparing 45 run pairs...
          Run 0 vs Run 1... ✓ r = 0.987, n = 432
          Run 0 vs Run 2... ✓ r = 0.982, n = 432
          ...
        >>> print(f"Best consistency: {max(p['r_value'].iloc[0] for p in pairs.values()):.3f}")
    """
    from itertools import combinations
    
    num_runs = _get_num_runs(bez_means)
    run_pairs = list(combinations(range(num_runs), 2))
    
    if verbose:
        print(f"Comparing {len(run_pairs)} run pairs...")
    
    results = {}
    
    for run_i, run_j in run_pairs:
        if verbose:
            print(f"  Run {run_i} vs Run {run_j}...", end=' ')
        
        try:
            df = regress_bez_run_vs_run(bez_means, run_i, run_j, rtol, atol)
            results[(run_i, run_j)] = df
            
            if verbose:
                r_val = df['r_value'].iloc[0]
                n_pts = df['n_points'].iloc[0]
                print(f"✓ r = {r_val:.3f}, n = {int(n_pts)}")
                
        except Exception as e:
            if verbose:
                print(f"✗ Error: {e}")
            results[(run_i, run_j)] = None
    
    return results


def regress_by_fiber_type(data1_means, data2_means, run1_idx=None, 
                           run2_idx=None, rtol=1e-5, atol=1e-8, verbose=True):
    """
    Separate regressions for each fiber type (HSR, MSR, LSR)
    
    Convenience wrapper for the common case of grouping by fiber type.
    Useful for checking if model agreement differs across fiber types.
    
    Parameters:
        data1_means: First dataset (e.g., Cochlea)
        data2_means: Second dataset (e.g., BEZ)
        run1_idx: Run index for data1 (None if no runs)
        run2_idx: Run index for data2 (None if no runs)
        rtol: Relative tolerance for fuzzy matching (default: 1e-5)
        atol: Absolute tolerance for fuzzy matching (default: 1e-8)
        verbose: Print progress
    
    Returns:
        dict: {'hsr': DataFrame, 'msr': DataFrame, 'lsr': DataFrame}
    
    Example:
        >>> cochlea = mean_across_time_psth_cochlea(...)
        >>> bez = mean_across_time_psth_bez(...)
        >>> by_fiber = regress_by_fiber_type(cochlea, bez, run2_idx=0)
        Analyzing by fiber_type: 3 groups
          fiber_type = hsr... ✓ r = 0.956, n = 144
          fiber_type = msr... ✓ r = 0.921, n = 144
          fiber_type = lsr... ✓ r = 0.887, n = 144
        >>> print(by_fiber['hsr']['r_value'].iloc[0])
        0.956
    """
    return _analyze_by_groups(
        data1_means, data2_means,
        groupby='fiber_type',
        run1_idx=run1_idx,
        run2_idx=run2_idx,
        analysis_func=_regression_analysis,
        rtol=rtol,
        atol=atol,
        verbose=verbose
    )


def regress_by_cf(data1_means, data2_means, run1_idx=None, run2_idx=None,
                  rtol=1e-5, atol=1e-8, verbose=True):
    """
    Separate regressions for each characteristic frequency (CF)
    
    Useful for checking if model agreement varies across the tonotopic axis.
    
    Parameters:
        data1_means, data2_means: Datasets to compare
        run1_idx, run2_idx: Run indices (None if no runs)
        rtol, atol: Fuzzy matching tolerances
        verbose: Print progress
    
    Returns:
        dict: {125.0: DataFrame, 250.0: DataFrame, ...} for each CF
    
    Example:
        >>> by_cf = regress_by_cf(cochlea, bez, run2_idx=0)
        Analyzing by cf: 16 groups
          cf = 125.0... ✓ r = 0.943, n = 27
          cf = 250.0... ✓ r = 0.938, n = 27
          ...
    """
    return _analyze_by_groups(
        data1_means, data2_means,
        groupby='cf',
        run1_idx=run1_idx,
        run2_idx=run2_idx,
        analysis_func=_regression_analysis,
        rtol=rtol,
        atol=atol,
        verbose=verbose
    )


def regress_by_db(data1_means, data2_means, run1_idx=None, run2_idx=None,
                  rtol=1e-5, atol=1e-8, verbose=True):
    """
    Separate regressions for each sound level (dB SPL)
    
    Useful for checking if model agreement varies with sound intensity.
    
    Parameters:
        data1_means, data2_means: Datasets to compare
        run1_idx, run2_idx: Run indices (None if no runs)
        rtol, atol: Fuzzy matching tolerances
        verbose: Print progress
    
    Returns:
        dict: {30.0: DataFrame, 40.0: DataFrame, ...} for each dB level
    
    Example:
        >>> by_db = regress_by_db(cochlea, bez, run2_idx=0)
        Analyzing by db: 3 groups
          db = 30.0... ✓ r = 0.912, n = 144
          db = 50.0... ✓ r = 0.945, n = 144
          db = 70.0... ✓ r = 0.958, n = 144
    """
    return _analyze_by_groups(
        data1_means, data2_means,
        groupby='db',
        run1_idx=run1_idx,
        run2_idx=run2_idx,
        analysis_func=_regression_analysis,
        rtol=rtol,
        atol=atol,
        verbose=verbose
    )


def correlate_by_fiber_type(data1_means, data2_means, run1_idx=None, 
                             run2_idx=None, rtol=1e-5, atol=1e-8, verbose=True):
    """
    Compute correlation (not regression) for each fiber type
    
    Same as regress_by_fiber_type() but returns only correlation statistics
    (no slope/intercept). Useful when you only care about agreement strength,
    not the scaling relationship.
    
    Parameters:
        data1_means, data2_means: Datasets to compare
        run1_idx, run2_idx: Run indices (None if no runs)
        rtol, atol: Fuzzy matching tolerances
        verbose: Print progress
    
    Returns:
        dict: {'hsr': DataFrame, 'msr': DataFrame, 'lsr': DataFrame}
              Each DataFrame has columns: r_value, p_value, n_points
              (no slope/intercept)
    
    Example:
        >>> corr = correlate_by_fiber_type(cochlea, bez, run2_idx=0)
        >>> print(f"HSR correlation: {corr['hsr']['r_value'].iloc[0]:.3f}")
    """
    return _analyze_by_groups(
        data1_means, data2_means,
        groupby='fiber_type',
        run1_idx=run1_idx,
        run2_idx=run2_idx,
        analysis_func=_correlation_analysis,  # Different analysis!
        rtol=rtol,
        atol=atol,
        verbose=verbose
    )


# ===================================================================
# VISUALIZATION FUNCTIONS
# ===================================================================

def plot_regression_scatter(cochlea_means, bez_means, run_idx=0, 
                            db_level=None, rtol=1e-5, atol=1e-8, 
                            title=None, figsize=(8, 8), save_path=None):
    """
    Plot scatter plot with regression line: Cochlea vs BEZ mean rates
    
    Creates a simple scatter plot showing the relationship between Cochlea
    and BEZ firing rates with the linear regression line overlaid.
    
    ⚠️ IMPORTANT: Analysis is conducted WITHIN a single dB level only!
    
    Parameters:
        cochlea_means: Nested dict from mean_across_time_psth_cochlea()
        bez_means: Nested dict from mean_across_time_psth_bez()
        run_idx: BEZ run index to plot (default: 0 for mean across runs)
        db_level: Specific dB level to analyze (e.g., 70.0). If None, uses all dB levels.
                  RECOMMENDED: Always specify a dB level for meaningful comparisons.
        rtol: Relative tolerance for fuzzy matching (default: 1e-5)
        atol: Absolute tolerance for fuzzy matching (default: 1e-8)
        title: Plot title (default: auto-generated)
        figsize: Figure size as (width, height) tuple (default: (8, 8))
        save_path: Path to save figure (default: None, displays instead)
    
    Returns:
        fig, ax: Matplotlib figure and axis objects
    
    Example:
        >>> cochlea = mean_across_time_psth_cochlea(cochlea_data, params)
        >>> bez = mean_across_time_psth_bez(bez_data, params)
        >>> 
        >>> # Analysis at specific dB level (RECOMMENDED)
        >>> fig, ax = plot_regression_scatter(cochlea, bez, run_idx=0, db_level=70.0)
        >>> plt.show()
        >>> 
        >>> # Or across all dB levels (may mix different response regimes)
        >>> fig, ax = plot_regression_scatter(cochlea, bez, run_idx=0, db_level=None)
    """
    import matplotlib.pyplot as plt
    
    # Filter to specific dB level if requested
    if db_level is not None:
        cochlea_means = extract_data_at_db(cochlea_means, db_level, rtol=rtol, atol=atol)
        bez_means = extract_data_at_db(bez_means, db_level, rtol=rtol, atol=atol)
    
    # Get matched data points
    matched_rates, matched_meta = match_data_points(
        cochlea_means, bez_means,
        run2_idx=run_idx,
        rtol=rtol,
        atol=atol,
        verbose=False
    )
    
    # Extract x and y data
    x = np.array([pt['data1_rate'] for pt in matched_rates])  # Cochlea
    y = np.array([pt['data2_rate'] for pt in matched_rates])  # BEZ
    
    # Compute regression
    result = regress_cochlea_vs_bez_single_run(
        cochlea_means, bez_means, run_idx, rtol, atol
    )
    
    slope = result['slope'].iloc[0]
    intercept = result['intercept'].iloc[0]
    r_value = result['r_value'].iloc[0]
    p_value = result['p_value'].iloc[0]
    n_points = int(result['n_points'].iloc[0])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(x, y, alpha=0.5, s=30, color='steelblue', edgecolors='navy', linewidth=0.5)
    
    # Regression line
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, 
            label=f'Regression: y = {slope:.3f}x + {intercept:.2f}')
    
    # Identity line (y = x)
    identity_line = np.array([min(x.min(), y.min()), max(x.max(), y.max())])
    ax.plot(identity_line, identity_line, 'k--', linewidth=1.5, alpha=0.7,
            label='Identity (y = x)')
    
    # Labels and title
    ax.set_xlabel('Cochlea Mean Rate (Hz)', fontsize=12)
    ax.set_ylabel('BEZ Mean Rate (Hz)', fontsize=12)
    
    if title is None:
        db_str = f' at {db_level} dB' if db_level is not None else ' (all dB levels)'
        title = f'Cochlea vs BEZ Run {run_idx}{db_str}\nr = {r_value:.3f}, p = {p_value:.2e}, n = {n_points}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Legend and grid
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Equal aspect for easier interpretation
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    return fig, ax


def plot_slope_distribution(cochlea_means, bez_means, 
                            rtol=1e-5, atol=1e-8,
                            bins=20, title=None, figsize=(10, 6), save_path=None):
    """
    Plot distribution of regression slopes across all BEZ runs
    
    Computes regression for Cochlea vs each BEZ run and displays
    the distribution of slopes as a histogram.
    
    Parameters:
        cochlea_means: Nested dict from mean_across_time_psth_cochlea()
        bez_means: Nested dict from mean_across_time_psth_bez()
        rtol: Relative tolerance for fuzzy matching (default: 1e-5)
        atol: Absolute tolerance for fuzzy matching (default: 1e-8)
        bins: Number of histogram bins (default: 20)
        title: Plot title (default: auto-generated)
        figsize: Figure size as (width, height) tuple (default: (10, 6))
        save_path: Path to save figure (default: None, displays instead)
    
    Returns:
        fig, ax: Matplotlib figure and axis objects
        slopes: List of slopes for each run
    
    Example:
        >>> cochlea = mean_across_time_psth_cochlea(cochlea_data, params)
        >>> bez = mean_across_time_psth_bez(bez_data, params)
        >>> fig, ax, slopes = plot_slope_distribution(cochlea, bez)
        >>> plt.show()
        
        >>> # Check slopes
        >>> print(f"Mean slope: {np.mean(slopes):.3f}")
        >>> print(f"Std slope: {np.std(slopes):.3f}")
    """
    import matplotlib.pyplot as plt
    
    # Get regression results for all runs
    print("Computing regressions for all runs...")
    all_runs = regress_cochlea_vs_bez_all_runs(
        cochlea_means, bez_means, rtol=rtol, atol=atol, verbose=False
    )
    
    # Extract slopes
    slopes = []
    run_indices = []
    for run_idx, df in sorted(all_runs.items()):
        if df is not None and not np.isnan(df['slope'].iloc[0]):
            slopes.append(df['slope'].iloc[0])
            run_indices.append(run_idx)
    
    slopes = np.array(slopes)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histogram
    n, bins_edges, patches = ax.hist(slopes, bins=bins, color='steelblue', 
                                      alpha=0.7, edgecolor='navy', linewidth=1.5)
    
    # Add mean line
    mean_slope = np.mean(slopes)
    ax.axvline(mean_slope, color='red', linestyle='--', linewidth=2.5,
               label=f'Mean = {mean_slope:.3f}')
    
    # Add std shading
    std_slope = np.std(slopes)
    ax.axvspan(mean_slope - std_slope, mean_slope + std_slope, 
               alpha=0.2, color='red', label=f'±1 SD = {std_slope:.3f}')
    
    # Reference line at slope = 1 (perfect agreement)
    ax.axvline(1.0, color='green', linestyle=':', linewidth=2,
               label='Perfect scaling (slope = 1)')
    
    # Labels and title
    ax.set_xlabel('Regression Slope', fontsize=12)
    ax.set_ylabel('Frequency (Number of Runs)', fontsize=12)
    
    if title is None:
        title = (f'Distribution of Regression Slopes\n'
                f'Cochlea vs BEZ Runs (n = {len(slopes)})\n'
                f'Mean = {mean_slope:.3f} ± {std_slope:.3f}')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Legend and grid
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    # Print statistics
    print(f"\nSlope Statistics:")
    print(f"  Mean: {mean_slope:.4f}")
    print(f"  Std:  {std_slope:.4f}")
    print(f"  Min:  {slopes.min():.4f} (Run {run_indices[np.argmin(slopes)]})")
    print(f"  Max:  {slopes.max():.4f} (Run {run_indices[np.argmax(slopes)]})")
    print(f"  Median: {np.median(slopes):.4f}")
    
    return fig, ax, slopes


def plot_regression_by_fiber_type(cochlea_means, bez_means, run_idx=0,
                                   db_level=None, rtol=1e-5, atol=1e-8,
                                   figsize=(18, 6), save_path=None):
    """
    Plot regression scatter plots separately for each fiber type
    
    Creates three side-by-side scatter plots (HSR, MSR, LSR) showing the
    relationship between Cochlea and BEZ for each fiber type separately.
    
    ⚠️ IMPORTANT: Analysis is conducted WITHIN a single dB level only!
    
    Parameters:
        cochlea_means: Nested dict from mean_across_time_psth_cochlea()
        bez_means: Nested dict from mean_across_time_psth_bez()
        run_idx: BEZ run index to plot (default: 0)
        db_level: Specific dB level to analyze (e.g., 70.0). If None, uses all dB levels.
                  RECOMMENDED: Always specify a dB level for meaningful comparisons.
        rtol: Relative tolerance for fuzzy matching (default: 1e-5)
        atol: Absolute tolerance for fuzzy matching (default: 1e-8)
        figsize: Figure size as (width, height) tuple (default: (18, 6))
        save_path: Path to save figure (default: None, displays instead)
    
    Returns:
        fig: Matplotlib figure object
        axes: Array of 3 axis objects [hsr_ax, msr_ax, lsr_ax]
        results: Dict with regression stats for each fiber type
    
    Example:
        >>> cochlea = mean_across_time_psth_cochlea(cochlea_data, params)
        >>> bez = mean_across_time_psth_bez(bez_data, params)
        >>> 
        >>> # Analysis at specific dB level (RECOMMENDED)
        >>> fig, axes, results = plot_regression_by_fiber_type(cochlea, bez, db_level=70.0)
        >>> print(f"HSR slope at 70 dB: {results['hsr']['slope']:.3f}")
        >>> plt.show()
    """
    import matplotlib.pyplot as plt
    
    # Filter to specific dB level if requested
    if db_level is not None:
        cochlea_means = extract_data_at_db(cochlea_means, db_level, rtol=rtol, atol=atol)
        bez_means = extract_data_at_db(bez_means, db_level, rtol=rtol, atol=atol)
    
    fiber_types = ['hsr', 'msr', 'lsr']
    colors = {'hsr': 'darkred', 'msr': 'darkgreen', 'lsr': 'darkblue'}
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    results = {}
    
    for ax, fiber_type in zip(axes, fiber_types):
        # Get matched data for this fiber type
        matched_rates, matched_meta = match_data_points(
            cochlea_means, bez_means,
            fiber_type=fiber_type,
            run2_idx=run_idx,
            rtol=rtol,
            atol=atol,
            verbose=False
        )
        
        if len(matched_rates) == 0:
            ax.text(0.5, 0.5, f'No data for {fiber_type.upper()}',
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Extract x and y data
        x = np.array([pt['data1_rate'] for pt in matched_rates])
        y = np.array([pt['data2_rate'] for pt in matched_rates])
        
        # Compute regression
        slope, intercept, r_value, p_value, stderr = linregress(x, y)
        
        # Store results
        results[fiber_type] = {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value,
            'stderr': stderr,
            'n_points': len(x)
        }
        
        # Scatter plot
        ax.scatter(x, y, alpha=0.6, s=40, 
                  color=colors[fiber_type],
                  edgecolors='black', linewidth=0.5)
        
        # Regression line
        x_line = np.array([x.min(), x.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=2.5,
               label=f'y = {slope:.3f}x + {intercept:.2f}')
        
        # Identity line (y = x)
        identity_min = min(x.min(), y.min())
        identity_max = max(x.max(), y.max())
        identity_line = np.array([identity_min, identity_max])
        ax.plot(identity_line, identity_line, 'k--', linewidth=1.5, alpha=0.6,
               label='y = x')
        
        # Formatting
        ax.set_xlabel('Cochlea Mean Rate (Hz)', fontsize=11)
        ax.set_ylabel('BEZ Mean Rate (Hz)', fontsize=11)
        ax.set_title(f'{fiber_type.upper()}\n'
                    f'r = {r_value:.3f}, p = {p_value:.2e}\n'
                    f'n = {len(x)}',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    # Overall title
    db_str = f' at {db_level} dB' if db_level is not None else ' (all dB levels)'
    fig.suptitle(f'Cochlea vs BEZ Run {run_idx}{db_str} - Regression by Fiber Type',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    # Print statistics
    print(f"\nRegression Statistics by Fiber Type (Run {run_idx}):")
    print("-" * 60)
    for fiber_type in fiber_types:
        if fiber_type in results:
            r = results[fiber_type]
            print(f"{fiber_type.upper()}: slope={r['slope']:.4f}, "
                  f"r={r['r_value']:.3f}, n={r['n_points']}")
    
    return fig, axes, results


def plot_slope_distribution_by_fiber_type(cochlea_means, bez_means,
                                          rtol=1e-5, atol=1e-8,
                                          bins=15, figsize=(18, 6), save_path=None):
    """
    Plot slope distributions separately for each fiber type
    
    Creates three side-by-side histograms showing the distribution of
    regression slopes across BEZ runs for each fiber type.
    
    Parameters:
        cochlea_means: Nested dict from mean_across_time_psth_cochlea()
        bez_means: Nested dict from mean_across_time_psth_bez()
        rtol: Relative tolerance for fuzzy matching (default: 1e-5)
        atol: Absolute tolerance for fuzzy matching (default: 1e-8)
        bins: Number of histogram bins (default: 15)
        figsize: Figure size as (width, height) tuple (default: (18, 6))
        save_path: Path to save figure (default: None, displays instead)
    
    Returns:
        fig: Matplotlib figure object
        axes: Array of 3 axis objects [hsr_ax, msr_ax, lsr_ax]
        slopes_dict: Dict with slopes array for each fiber type
    
    Example:
        >>> cochlea = mean_across_time_psth_cochlea(cochlea_data, params)
        >>> bez = mean_across_time_psth_bez(bez_data, params)
        >>> fig, axes, slopes = plot_slope_distribution_by_fiber_type(cochlea, bez)
        >>> print(f"HSR mean slope: {np.mean(slopes['hsr']):.3f}")
        >>> plt.show()
    """
    import matplotlib.pyplot as plt
    
    fiber_types = ['hsr', 'msr', 'lsr']
    colors = {'hsr': 'darkred', 'msr': 'darkgreen', 'lsr': 'darkblue'}
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    slopes_dict = {}
    
    # Get number of runs
    num_runs = _get_num_runs(bez_means)
    
    print(f"Computing regressions for {num_runs} runs across fiber types...")
    
    for ax, fiber_type in zip(axes, fiber_types):
        slopes = []
        
        # Compute regression for each run
        for run_idx in range(num_runs):
            matched_rates, _ = match_data_points(
                cochlea_means, bez_means,
                fiber_type=fiber_type,
                run2_idx=run_idx,
                rtol=rtol,
                atol=atol,
                verbose=False
            )
            
            if len(matched_rates) == 0:
                continue
            
            x = np.array([pt['data1_rate'] for pt in matched_rates])
            y = np.array([pt['data2_rate'] for pt in matched_rates])
            
            slope, _, _, _, _ = linregress(x, y)
            slopes.append(slope)
        
        slopes = np.array(slopes)
        slopes_dict[fiber_type] = slopes
        
        if len(slopes) == 0:
            ax.text(0.5, 0.5, f'No data for {fiber_type.upper()}',
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Histogram
        n, bins_edges, patches = ax.hist(slopes, bins=bins, 
                                         color=colors[fiber_type],
                                         alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Mean line
        mean_slope = np.mean(slopes)
        ax.axvline(mean_slope, color='red', linestyle='--', linewidth=2.5,
                  label=f'Mean = {mean_slope:.3f}')
        
        # Std shading
        std_slope = np.std(slopes)
        ax.axvspan(mean_slope - std_slope, mean_slope + std_slope,
                  alpha=0.2, color='red', label=f'±1 SD = {std_slope:.3f}')
        
        # Reference line at slope = 1
        ax.axvline(1.0, color='green', linestyle=':', linewidth=2,
                  label='Slope = 1')
        
        # Formatting
        ax.set_xlabel('Regression Slope', fontsize=11)
        ax.set_ylabel('Frequency (# of Runs)', fontsize=11)
        ax.set_title(f'{fiber_type.upper()}\n'
                    f'Mean = {mean_slope:.3f} ± {std_slope:.3f}\n'
                    f'n = {len(slopes)} runs',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Overall title
    fig.suptitle('Slope Distribution by Fiber Type\nCochlea vs BEZ Runs',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    # Print statistics
    print(f"\nSlope Statistics by Fiber Type:")
    print("-" * 60)
    for fiber_type in fiber_types:
        if fiber_type in slopes_dict and len(slopes_dict[fiber_type]) > 0:
            slopes = slopes_dict[fiber_type]
            print(f"{fiber_type.upper()}: mean={np.mean(slopes):.4f}, "
                  f"std={np.std(slopes):.4f}, "
                  f"range=[{slopes.min():.4f}, {slopes.max():.4f}]")
    
    return fig, axes, slopes_dict


# ===================================================================
# ADVANCED VISUALIZATION WITH SEABORN
# ===================================================================

def plot_regression_seaborn_cf_brightness(cochlea_means, bez_means, run_idx=0,
                                           db_level=None, rtol=1e-5, atol=1e-8,
                                           figsize=(14, 10), save_path=None):
    """
    Advanced seaborn scatter plot with CF as hue and frequency difference as brightness.
    
    Creates publication-quality plots where:
    - Color (hue) represents Characteristic Frequency (perceptually uniform)
    - Brightness (alpha) represents distance from CF (brighter = at CF, dimmer = far from CF)
    
    ⚠️ IMPORTANT: Analysis is conducted WITHIN a single dB level only!
    
    Parameters:
    -----------
    cochlea_means : dict
        Nested dict from mean_across_time_psth_cochlea()
    bez_means : dict
        Nested dict from mean_across_time_psth_bez()
    run_idx : int
        BEZ run index to plot (default: 0)
    db_level : float
        Specific dB level to analyze (REQUIRED for meaningful comparisons)
    rtol, atol : float
        Fuzzy matching tolerances
    figsize : tuple
        Figure size (width, height) in inches (default: (14, 10))
    save_path : str or None
        Path to save figure. If provided, saves as PNG and PDF.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object containing all 3 subplots
    axes : array of matplotlib.axes.Axes
        Array of 3 axis objects [hsr_ax, msr_ax, lsr_ax]
    stats : dict
        Statistics for each fiber type (slope, R², etc.)
    
    Example:
    --------
    >>> cochlea = mean_across_time_psth_cochlea(cochlea_data, params)
    >>> bez = mean_across_time_psth_bez(bez_data, params)
    >>> fig, axes, stats = plot_regression_seaborn_cf_brightness(
    ...     cochlea, bez, run_idx=0, db_level=70.0
    ... )
    >>> print(f"HSR slope: {stats['hsr']['slope']:.3f}")
    >>> plt.show()
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError("This function requires seaborn. Install with: pip install seaborn")
    
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    
    # Set seaborn style
    sns.set_theme(style="whitegrid")
    
    if db_level is None:
        raise ValueError("db_level must be specified for meaningful CF-frequency analysis")
    
    # Filter to specific dB level
    cochlea_db = extract_data_at_db(cochlea_means, db_level, rtol=rtol, atol=atol)
    bez_db = extract_data_at_db(bez_means, db_level, rtol=rtol, atol=atol)
    
    fiber_types = ['hsr', 'msr', 'lsr']
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    stats = {}
    
    # Get unique CFs (same across all fiber types)
    sample_fiber = fiber_types[0]
    unique_cfs = sorted(cochlea_db[sample_fiber].keys())
    n_cfs = len(unique_cfs)
    
    # Create perceptually uniform color palette for CFs
    cf_palette = sns.color_palette("Spectral", n_colors=n_cfs)
    cf_color_map = {cf: cf_palette[i] for i, cf in enumerate(unique_cfs)}
    
    print(f"Creating seaborn plots with {n_cfs} CFs at {db_level} dB...")
    
    for ax, fiber_type in zip(axes, fiber_types):
        # Collect all data points for this fiber type
        data_points = []
        
        fiber_cochlea = cochlea_db[fiber_type]
        fiber_bez = bez_db[fiber_type]
        
        for cf_val in fiber_cochlea.keys():
            if cf_val not in fiber_bez:
                continue
            
            for freq_val in fiber_cochlea[cf_val].keys():
                if freq_val not in fiber_bez[cf_val]:
                    continue
                
                cochlea_rate = fiber_cochlea[cf_val][freq_val]
                bez_rate = fiber_bez[cf_val][freq_val][run_idx]
                
                # Calculate frequency difference for brightness
                freq_diff = freq_val - cf_val
                
                data_points.append({
                    'cf': cf_val,
                    'freq': freq_val,
                    'freq_diff': freq_diff,
                    'cochlea_rate': cochlea_rate,
                    'bez_rate': bez_rate
                })
        
        if len(data_points) == 0:
            print(f"  {fiber_type.upper()}: No data points")
            continue
        
        # Convert to arrays for processing
        cf_vals = np.array([p['cf'] for p in data_points])
        freq_diffs = np.array([p['freq_diff'] for p in data_points])
        cochlea_rates = np.array([p['cochlea_rate'] for p in data_points])
        bez_rates = np.array([p['bez_rate'] for p in data_points])
        
        # Calculate alpha values based on absolute frequency difference
        # Closer to CF = brighter (higher alpha), farther = dimmer (lower alpha)
        abs_freq_diff = np.abs(freq_diffs)
        max_abs_diff = abs_freq_diff.max() if abs_freq_diff.max() > 0 else 1.0
        
        # Exponential scaling for more pronounced brightness differences (0.1 to 1.0 range)
        normalized_diff = abs_freq_diff / max_abs_diff
        alpha_vals = 0.1 + 0.9 * (1 - normalized_diff)**2
        
        # Plot points grouped by CF with brightness
        for cf_val in unique_cfs:
            cf_mask = cf_vals == cf_val
            if not np.any(cf_mask):
                continue
            
            cf_bez = bez_rates[cf_mask]
            cf_cochlea = cochlea_rates[cf_mask]
            cf_alphas = alpha_vals[cf_mask]
            
            # Plot each point with its specific alpha
            for i in range(len(cf_bez)):
                ax.scatter(cf_bez[i], cf_cochlea[i],
                          c=[cf_color_map[cf_val]], alpha=cf_alphas[i],
                          s=80, edgecolors='black', linewidths=0.5)
        
        # Remove NaN and infinite values for regression
        valid_mask = np.isfinite(bez_rates) & np.isfinite(cochlea_rates)
        clean_bez = bez_rates[valid_mask]
        clean_cochlea = cochlea_rates[valid_mask]
        
        # Compute regression
        if len(clean_bez) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(clean_bez, clean_cochlea)
            
            # Add regression line
            x_range = np.linspace(clean_bez.min(), clean_bez.max(), 100)
            y_fit = slope * x_range + intercept
            ax.plot(x_range, y_fit, 'k-', linewidth=2, alpha=0.8, label='Linear fit')
            
            stats[fiber_type] = {
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_err': std_err,
                'n_points': len(clean_bez)
            }
        else:
            stats[fiber_type] = {
                'slope': np.nan,
                'intercept': np.nan,
                'r_value': np.nan,
                'r_squared': np.nan,
                'p_value': np.nan,
                'std_err': np.nan,
                'n_points': 0
            }
        
        # Add diagonal reference line (perfect match)
        min_val = min(clean_bez.min(), clean_cochlea.min()) if len(clean_bez) > 0 else 0
        max_val = max(clean_bez.max(), clean_cochlea.max()) if len(clean_bez) > 0 else 1
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', 
               linewidth=1, alpha=0.5, label='Perfect match')
        
        # Add statistics text
        if not np.isnan(stats[fiber_type]['slope']):
            stats_text = (f"R² = {stats[fiber_type]['r_squared']:.3f}\n"
                         f"Slope = {stats[fiber_type]['slope']:.3f}\n"
                         f"n = {stats[fiber_type]['n_points']}")
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='white', 
                           alpha=0.9, edgecolor='gray'))
        
        # Formatting
        ax.set_xlabel('BEZ Mean Rate (Hz)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cochlea Mean Rate (Hz)', fontsize=11, fontweight='bold')
        ax.set_title(f'{fiber_type.upper()} Fibers', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        print(f"  {fiber_type.upper()}: {len(data_points)} points, "
              f"slope={stats[fiber_type]['slope']:.3f}, "
              f"R²={stats[fiber_type]['r_squared']:.3f}")
    
    # Create custom legend for CF colors and brightness
    legend_entries = []
    
    # Sample a subset of CFs for legend (every 3rd to avoid crowding)
    sample_cfs = unique_cfs[::max(1, len(unique_cfs)//6)]
    for cf_val in sample_cfs:
        legend_entries.append(Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=cf_color_map[cf_val], markersize=6,
                                    markeredgecolor='black', markeredgewidth=0.5,
                                    label=f'CF {cf_val:.0f} Hz'))
    
    # Add brightness explanation
    legend_entries.extend([
        Line2D([0], [0], marker='o', color='w',
              markerfacecolor='gray', alpha=0.1, markersize=8,
              markeredgecolor='black', markeredgewidth=0.5,
              label='Far from CF (dim)'),
        Line2D([0], [0], marker='o', color='w',
              markerfacecolor='gray', alpha=1.0, markersize=8,
              markeredgecolor='black', markeredgewidth=0.5,
              label='At CF (bright)')
    ])
    
    # Add legend to the side
    legend = fig.legend(handles=legend_entries, bbox_to_anchor=(1.02, 0.5),
                       loc='center left', fontsize=8, frameon=True,
                       fancybox=True, shadow=True, framealpha=0.9,
                       title='Color: CF\nBrightness: |f-CF|')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    
    # Overall title
    fig.suptitle(f'Cochlea vs BEZ Run {run_idx} at {db_level} dB\n'
                f'Color = Characteristic Frequency, Brightness = Distance from CF',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    return fig, axes, stats


# =============================================================================
# BEZ RUN CORRELATION ANALYSIS
# =============================================================================

def compute_bez_run_pairwise_correlations(bez_means, db_level=None, fiber_types=None):
    """
    Compute pair-wise Pearson correlations between all BEZ runs.
    
    This function helps determine if BEZ runs are similar enough to justify averaging.
    Compares all possible pairs of runs (e.g., run 0 vs run 1, run 0 vs run 2, etc.)
    across all CF and frequency combinations.
    
    Parameters:
    -----------
    bez_means : dict
        Nested dictionary from mean_across_time_psth_bez()
        Structure: [fiber_type][cf][freq][db][run] = mean_rate
    db_level : float, optional
        If specified, only analyze this dB level. If None, analyze all dB levels separately
    fiber_types : list, optional
        List of fiber types to analyze (e.g., ['hsr', 'msr', 'lsr'])
        If None, analyzes all available fiber types
    
    Returns:
    --------
    results : dict
        Nested dictionary with correlation results:
        - If db_level specified: [fiber_type][run_pair] = correlation_value
        - If db_level is None: [fiber_type][db][run_pair] = correlation_value
        Also includes 'summary' key with mean/std statistics
    
    Example:
    --------
    >>> correlations = compute_bez_run_pairwise_correlations(bez_means, db_level=60.0)
    >>> print(correlations['hsr'][(0, 1)])  # Correlation between run 0 and run 1 for HSR
    >>> print(correlations['summary']['hsr']['mean'])  # Mean correlation across all run pairs
    """
    from scipy.stats import pearsonr
    import itertools
    
    if fiber_types is None:
        fiber_types = ['hsr', 'msr', 'lsr']
    
    # Filter to specific dB level if requested
    if db_level is not None:
        bez_db = extract_data_at_db(bez_means, db_level, fiber_types=fiber_types)
        db_levels = [db_level]
    else:
        bez_db = bez_means
        # Get all dB levels from the first fiber type
        first_fiber = list(bez_means.keys())[0]
        first_cf = list(bez_means[first_fiber].keys())[0]
        first_freq = list(bez_means[first_fiber][first_cf].keys())[0]
        db_levels = list(bez_means[first_fiber][first_cf][first_freq].keys())
    
    results = {}
    
    for fiber in fiber_types:
        if fiber not in bez_db:
            continue
            
        results[fiber] = {}
        
        # Determine which structure we're working with
        if db_level is not None:
            # Structure: [cf][freq][run]
            db_dict = {db_level: bez_db[fiber]}
        else:
            # Structure: [cf][freq][db][run]
            # We need to organize by dB level
            db_dict = {}
            for db in db_levels:
                db_dict[db] = extract_data_at_db(bez_means, db)[fiber]
        
        # Process each dB level
        for db in db_dict.keys():
            fiber_data = db_dict[db]
            
            # Collect all runs' data
            runs_data = {}  # run_idx -> list of mean rates
            
            for cf in fiber_data.keys():
                for freq in fiber_data[cf].keys():
                    if db_level is not None:
                        # Data is at [run] level
                        run_values = fiber_data[cf][freq]
                    else:
                        # Data is at [db][run] level
                        run_values = fiber_data[cf][freq]
                    
                    # Collect values for each run
                    for run_idx, mean_rate in run_values.items():
                        if run_idx not in runs_data:
                            runs_data[run_idx] = []
                        runs_data[run_idx].append(mean_rate)
            
            # Convert to arrays
            run_indices = sorted(runs_data.keys())
            run_arrays = {idx: np.array(runs_data[idx]) for idx in run_indices}
            
            # Compute pair-wise correlations
            correlations = {}
            for run_i, run_j in itertools.combinations(run_indices, 2):
                # Ensure same length (should always be true)
                if len(run_arrays[run_i]) == len(run_arrays[run_j]):
                    r, p = pearsonr(run_arrays[run_i], run_arrays[run_j])
                    correlations[(run_i, run_j)] = {
                        'r': r,
                        'p': p,
                        'n': len(run_arrays[run_i])
                    }
            
            # Store results
            if db_level is not None:
                results[fiber] = correlations
            else:
                results[fiber][db] = correlations
    
    # Compute summary statistics
    results['summary'] = {}
    
    for fiber in fiber_types:
        if fiber not in results:
            continue
            
        results['summary'][fiber] = {}
        
        if db_level is not None:
            # Single dB level
            all_r = [v['r'] for v in results[fiber].values()]
            results['summary'][fiber] = {
                'mean': np.mean(all_r),
                'std': np.std(all_r),
                'min': np.min(all_r),
                'max': np.max(all_r),
                'n_pairs': len(all_r)
            }
        else:
            # Multiple dB levels
            for db in results[fiber].keys():
                all_r = [v['r'] for v in results[fiber][db].values()]
                results['summary'][fiber][db] = {
                    'mean': np.mean(all_r),
                    'std': np.std(all_r),
                    'min': np.min(all_r),
                    'max': np.max(all_r),
                    'n_pairs': len(all_r)
                }
    
    return results


def compute_cochlea_vs_bez_runs_correlations(cochlea_means, bez_means, db_level, fiber_types=None):
    """
    Compute correlations between Cochlea and each individual BEZ run.
    
    This allows comparison of Cochlea-BEZ agreement vs BEZ run-to-run agreement.
    
    Parameters:
    -----------
    cochlea_means : dict
        Nested dictionary from mean_across_time_psth_cochlea()
    bez_means : dict
        Nested dictionary from mean_across_time_psth_bez()
    db_level : float
        dB level to analyze
    fiber_types : list, optional
        List of fiber types to analyze. If None, uses ['hsr', 'msr', 'lsr']
    
    Returns:
    --------
    results : dict
        Structure: [fiber_type][run_idx] = {'r': correlation, 'p': p_value, 'n': n_points}
        Also includes 'summary' with statistics across runs
    
    Example:
    --------
    >>> corr = compute_cochlea_vs_bez_runs_correlations(cochlea_means, bez_means, 60.0)
    >>> print(corr['hsr'][0]['r'])  # Correlation between Cochlea and BEZ run 0 for HSR
    >>> print(corr['summary']['hsr']['mean'])  # Mean Cochlea-BEZ correlation across runs
    """
    from scipy.stats import pearsonr
    
    if fiber_types is None:
        fiber_types = ['hsr', 'msr', 'lsr']
    
    # Filter to specific dB level
    cochlea_db = extract_data_at_db(cochlea_means, db_level, fiber_types=fiber_types)
    bez_db = extract_data_at_db(bez_means, db_level, fiber_types=fiber_types)
    
    results = {}
    
    for fiber in fiber_types:
        if fiber not in cochlea_db or fiber not in bez_db:
            continue
        
        results[fiber] = {}
        
        # Get all available runs
        first_cf = list(bez_db[fiber].keys())[0]
        first_freq = list(bez_db[fiber][first_cf].keys())[0]
        run_indices = list(bez_db[fiber][first_cf][first_freq].keys())
        
        # For each run, collect matched data points
        for run_idx in run_indices:
            cochlea_vals = []
            bez_vals = []
            
            for cf in cochlea_db[fiber].keys():
                if cf not in bez_db[fiber]:
                    continue
                    
                for freq in cochlea_db[fiber][cf].keys():
                    if freq not in bez_db[fiber][cf]:
                        continue
                    
                    cochlea_vals.append(cochlea_db[fiber][cf][freq])
                    bez_vals.append(bez_db[fiber][cf][freq][run_idx])
            
            # Compute correlation
            if len(cochlea_vals) > 0:
                r, p = pearsonr(cochlea_vals, bez_vals)
                results[fiber][run_idx] = {
                    'r': r,
                    'p': p,
                    'n': len(cochlea_vals)
                }
    
    # Compute summary statistics
    results['summary'] = {}
    
    for fiber in fiber_types:
        if fiber not in results:
            continue
        
        all_r = [v['r'] for v in results[fiber].values()]
        results['summary'][fiber] = {
            'mean': np.mean(all_r),
            'std': np.std(all_r),
            'min': np.min(all_r),
            'max': np.max(all_r),
            'n_runs': len(all_r)
        }
    
    return results


def plot_correlation_comparison(bez_pairwise, cochlea_bez, fiber_types=None, 
                                  save_path=None, figsize=None):
    """
    Plot comparison of BEZ run-to-run correlations vs Cochlea-BEZ correlations.
    
    Creates side-by-side violin/box plots to visualize:
    1. Distribution of pair-wise BEZ run correlations (run-to-run variability)
    2. Distribution of Cochlea vs individual BEZ run correlations (model agreement)
    
    Parameters:
    -----------
    bez_pairwise : dict
        Results from compute_bez_run_pairwise_correlations()
    cochlea_bez : dict
        Results from compute_cochlea_vs_bez_runs_correlations()
    fiber_types : list, optional
        Fiber types to plot. If None, uses ['hsr', 'msr', 'lsr']
        For population analysis, use ['population']
    save_path : str, optional
        Path to save figure
    figsize : tuple, optional
        Figure size (width, height). Auto-determined based on number of fiber types
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    
    if fiber_types is None:
        fiber_types = ['hsr', 'msr', 'lsr']
    
    # Auto-determine figure size if not specified
    if figsize is None:
        n_types = len(fiber_types)
        if n_types == 1:
            figsize = (7, 5)
        elif n_types == 3:
            figsize = (15, 5)
        else:
            figsize = (5 * n_types, 5)
    
    # Create subplots
    if len(fiber_types) == 1:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, len(fiber_types), figsize=figsize)
        if not isinstance(axes, np.ndarray):
            axes = [axes]
    
    for idx, fiber in enumerate(fiber_types):
        ax = axes[idx]
        
        # Extract BEZ pair-wise correlations
        bez_corrs = [v['r'] for v in bez_pairwise[fiber].values()]
        
        # Extract Cochlea-BEZ correlations
        coch_corrs = [v['r'] for v in cochlea_bez[fiber].values()]
        
        # Create box plots
        bp = ax.boxplot([bez_corrs, coch_corrs], 
                        labels=['BEZ run pairs', 'Cochlea vs BEZ'],
                        widths=0.6,
                        patch_artist=True,
                        showfliers=True,
                        showmeans=True)
        
        # Color the boxes
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        # Add individual points
        x1 = np.random.normal(1, 0.04, len(bez_corrs))
        x2 = np.random.normal(2, 0.04, len(coch_corrs))
        ax.scatter(x1, bez_corrs, alpha=0.4, s=30, c='blue', edgecolors='black', linewidth=0.5)
        ax.scatter(x2, coch_corrs, alpha=0.4, s=30, c='red', edgecolors='black', linewidth=0.5)
        
        # Add mean lines
        ax.axhline(np.mean(bez_corrs), color='blue', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(np.mean(coch_corrs), color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Labels
        ax.set_ylabel('Pearson Correlation (r)', fontsize=11, fontweight='bold')
        
        # Title based on fiber type
        if fiber == 'population':
            title = 'Weighted Population'
        else:
            title = f'{fiber.upper()} Fibers'
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        textstr = f'BEZ pairs: {np.mean(bez_corrs):.3f}±{np.std(bez_corrs):.3f}\n'
        textstr += f'Cochlea-BEZ: {np.mean(coch_corrs):.3f}±{np.std(coch_corrs):.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=props)
    
    # Title
    if len(fiber_types) == 1 and fiber_types[0] == 'population':
        title = 'Weighted Population: BEZ Run-to-Run vs Cochlea-BEZ Agreement'
    else:
        title = 'BEZ Run-to-Run Variability vs Cochlea-BEZ Agreement'
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Correlation comparison saved to: {save_path}")
    
    return fig, axes if len(axes) > 1 else (fig, ax)


def save_correlation_statistics(bez_pairwise, cochlea_bez, db_level, output_dir='model_comparison_PSTH_final'):
    """
    Save correlation statistics to JSON and text files for use in LaTeX documents.
    
    Parameters:
    -----------
    bez_pairwise : dict
        Results from compute_bez_run_pairwise_correlations()
    cochlea_bez : dict
        Results from compute_cochlea_vs_bez_runs_correlations()
    db_level : float
        The dB level analyzed
    output_dir : str
        Directory to save statistics files
    
    Returns:
    --------
    output_files : dict
        Dictionary with paths to created files
    """
    import json
    from pathlib import Path
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for JSON
    json_data = {
        'db_level': db_level,
        'bez_run_pairwise': {},
        'cochlea_vs_bez': {},
        'summary': {}
    }
    
    fiber_types = ['hsr', 'msr', 'lsr']
    
    # Extract BEZ pair-wise correlations
    for fiber in fiber_types:
        if fiber in bez_pairwise:
            # Convert tuple keys to strings for JSON serialization
            json_data['bez_run_pairwise'][fiber] = {
                f"run_{pair[0]}_vs_run_{pair[1]}": {
                    'r': float(v['r']),
                    'p': float(v['p']),
                    'n': int(v['n'])
                }
                for pair, v in bez_pairwise[fiber].items()
            }
            
            # Add summary
            if fiber in bez_pairwise['summary']:
                json_data['bez_run_pairwise'][f'{fiber}_summary'] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in bez_pairwise['summary'][fiber].items()
                }
    
    # Extract Cochlea-BEZ correlations
    for fiber in fiber_types:
        if fiber in cochlea_bez:
            json_data['cochlea_vs_bez'][fiber] = {
                f"run_{run_idx}": {
                    'r': float(v['r']),
                    'p': float(v['p']),
                    'n': int(v['n'])
                }
                for run_idx, v in cochlea_bez[fiber].items()
            }
            
            # Add summary
            if fiber in cochlea_bez['summary']:
                json_data['cochlea_vs_bez'][f'{fiber}_summary'] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in cochlea_bez['summary'][fiber].items()
                }
    
    # Create comparison summary
    for fiber in fiber_types:
        if fiber in bez_pairwise['summary'] and fiber in cochlea_bez['summary']:
            json_data['summary'][fiber] = {
                'bez_mean_r': float(bez_pairwise['summary'][fiber]['mean']),
                'bez_std_r': float(bez_pairwise['summary'][fiber]['std']),
                'cochlea_mean_r': float(cochlea_bez['summary'][fiber]['mean']),
                'cochlea_std_r': float(cochlea_bez['summary'][fiber]['std']),
                'difference': float(bez_pairwise['summary'][fiber]['mean'] - 
                                   cochlea_bez['summary'][fiber]['mean'])
            }
    
    # Save JSON file
    json_file = output_path / f'correlation_stats_{int(db_level)}dB.json'
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Create LaTeX-friendly text file
    txt_file = output_path / f'correlation_stats_{int(db_level)}dB.txt'
    with open(txt_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"CORRELATION STATISTICS at {db_level} dB\n")
        f.write("="*80 + "\n\n")
        
        # BEZ run pair-wise correlations
        f.write("BEZ RUN PAIR-WISE CORRELATIONS\n")
        f.write("-"*80 + "\n")
        for fiber in fiber_types:
            if fiber in bez_pairwise['summary']:
                s = bez_pairwise['summary'][fiber]
                f.write(f"{fiber.upper()}: mean={s['mean']:.4f}, std={s['std']:.4f}, ")
                f.write(f"min={s['min']:.4f}, max={s['max']:.4f}, n_pairs={s['n_pairs']}\n")
        
        f.write("\n")
        
        # Cochlea-BEZ correlations
        f.write("COCHLEA vs BEZ CORRELATIONS\n")
        f.write("-"*80 + "\n")
        for fiber in fiber_types:
            if fiber in cochlea_bez['summary']:
                s = cochlea_bez['summary'][fiber]
                f.write(f"{fiber.upper()}: mean={s['mean']:.4f}, std={s['std']:.4f}, ")
                f.write(f"min={s['min']:.4f}, max={s['max']:.4f}, n_runs={s['n_runs']}\n")
        
        f.write("\n")
        
        # Comparison summary
        f.write("COMPARISON SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Fiber':<8} {'BEZ Mean':<12} {'Cochlea Mean':<12} {'Difference':<12}\n")
        f.write("-"*80 + "\n")
        for fiber in fiber_types:
            if fiber in json_data['summary']:
                s = json_data['summary'][fiber]
                f.write(f"{fiber.upper():<8} {s['bez_mean_r']:<12.4f} ")
                f.write(f"{s['cochlea_mean_r']:<12.4f} {s['difference']:<+12.4f}\n")
    
    # Create LaTeX table
    tex_file = output_path / f'correlation_table_{int(db_level)}dB.tex'
    with open(tex_file, 'w') as f:
        f.write("% LaTeX table for correlation statistics\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("  \\centering\n")
        f.write(f"  \\caption{{Correlation statistics at {int(db_level)} dB}}\n")
        f.write("  \\begin{tabular}{lccc}\n")
        f.write("    \\hline\n")
        f.write("    Fiber Type & BEZ Run Pairs & Cochlea vs BEZ & Difference \\\\\n")
        f.write("    \\hline\n")
        
        for fiber in fiber_types:
            if fiber in json_data['summary']:
                s = json_data['summary'][fiber]
                f.write(f"    {fiber.upper()} & ")
                f.write(f"${s['bez_mean_r']:.3f} \\pm {s['bez_std_r']:.3f}$ & ")
                f.write(f"${s['cochlea_mean_r']:.3f} \\pm {s['cochlea_std_r']:.3f}$ & ")
                f.write(f"${s['difference']:+.3f}$ \\\\\n")
        
        f.write("    \\hline\n")
        f.write("  \\end{tabular}\n")
        f.write(f"  \\label{{tab:correlation_{int(db_level)}dB}}\n")
        f.write("\\end{table}\n")
    
    output_files = {
        'json': str(json_file),
        'txt': str(txt_file),
        'tex': str(tex_file)
    }
    
    print(f"\n✓ Correlation statistics saved to {output_dir}/")
    print(f"  - JSON: {json_file.name}")
    print(f"  - Text: {txt_file.name}")
    print(f"  - LaTeX: {tex_file.name}")
    
    return output_files


# =============================================================================
# WEIGHTED POPULATION ANALYSIS
# =============================================================================

def compute_weighted_fiber_sum_psth(psth_data, params, weights=None, normalize=True):
    """
    Compute weighted sum of fiber type PSTHs to create population response.
    
    This combines HSR, MSR, and LSR responses with physiologically-motivated ratios
    to simulate the population-level auditory nerve response.
    
    ⚠️ IMPORTANT: This function handles MATLAB object arrays from load_bez_psth_data()
    The input arrays are 4D object arrays (CF × freq × dB × run) where each element
    contains a 1D PSTH time series array.
    
    Parameters:
    -----------
    psth_data : dict
        Dictionary with 'hsr_all', 'msr_all', 'lsr_all' arrays
        - Shape (n_cf, n_freq, n_db, n_runs), dtype=object
        - Each element contains 1D array (n_time,) with PSTH time series
        - For Cochlea: shape (n_cf, n_freq, n_db, n_time), dtype=float (not object)
    params : dict
        Parameters dictionary with 'cfs', 'frequencies', 'dbs'
    weights : dict, optional
        Weights for each fiber type. Default: {'hsr': 0.65, 'msr': 0.25, 'lsr': 0.15}
    normalize : bool, optional
        If True, enforce that weights sum to 1.0. Default: True
    
    Returns:
    --------
    weighted_psth : dict
        Nested dictionary structure: [cf][freq][db][run] = weighted_psth_array (1D)
        Or [cf][freq][db] = weighted_psth_array (1D) for Cochlea without runs
    weights_used : dict
        The actual weights used (after normalization if applicable)
    
    Example:
    --------
    >>> weighted, weights = compute_weighted_fiber_sum_psth(bez_data, params)
    >>> print(type(weighted))  # dict
    >>> print(weighted[125.0][125.0][50.0][0].shape)  # (200,) - PSTH time series
    """
    # Default physiological ratios
    if weights is None:
        weights = {'hsr': 0.65, 'msr': 0.25, 'lsr': 0.15}
    
    # Validate weights
    fiber_types = ['hsr', 'msr', 'lsr']
    for fiber in fiber_types:
        if fiber not in weights:
            raise ValueError(f"Missing weight for fiber type: {fiber}")
    
    # Normalize weights to sum to 1.0
    if normalize:
        weight_sum = sum(weights[f] for f in fiber_types)
        if not np.isclose(weight_sum, 1.0):
            print(f"⚠ Normalizing weights (sum was {weight_sum:.4f})")
            weights = {f: weights[f] / weight_sum for f in fiber_types}
    else:
        weight_sum = sum(weights[f] for f in fiber_types)
        if not np.isclose(weight_sum, 1.0):
            print(f"⚠ Warning: Weights sum to {weight_sum:.4f}, not 1.0")
    
    print(f"Computing weighted population PSTH...")
    print(f"  Weights: HSR={weights['hsr']:.3f}, MSR={weights['msr']:.3f}, LSR={weights['lsr']:.3f}")
    
    # Check if data has run dimension (BEZ) or not (Cochlea)
    sample_shape = psth_data['hsr_all'].shape
    has_runs = (len(sample_shape) == 4 and psth_data['hsr_all'].dtype == object)
    
    if has_runs:
        print(f"  Detected BEZ data with runs: shape {sample_shape}")
    else:
        print(f"  Detected Cochlea data without runs: shape {sample_shape}")
    
    # Initialize weighted PSTH dictionary
    weighted_psth = {}
    
    # Iterate through all conditions
    for cf_idx, cf in enumerate(params['cfs']):
        weighted_psth[cf] = {}
        
        for freq_idx, freq in enumerate(params['frequencies']):
            weighted_psth[cf][freq] = {}
            
            for db_idx, db in enumerate(params['dbs']):
                
                if has_runs:
                    # BEZ: 4D object array - iterate through runs
                    weighted_psth[cf][freq][db] = {}
                    n_runs = sample_shape[3]
                    
                    for run_idx in range(n_runs):
                        # Get PSTH time series for each fiber type (these are 1D arrays)
                        hsr_psth = psth_data['hsr_all'][cf_idx, freq_idx, db_idx, run_idx]
                        msr_psth = psth_data['msr_all'][cf_idx, freq_idx, db_idx, run_idx]
                        lsr_psth = psth_data['lsr_all'][cf_idx, freq_idx, db_idx, run_idx]
                        
                        # Compute weighted sum of PSTH time series
                        weighted_pop_psth = (
                            weights['hsr'] * hsr_psth +
                            weights['msr'] * msr_psth +
                            weights['lsr'] * lsr_psth
                        )
                        
                        weighted_psth[cf][freq][db][run_idx] = weighted_pop_psth
                else:
                    # Cochlea: 3D or 4D numeric array - no run dimension
                    # Extract PSTH for this condition (should be 1D time series)
                    hsr_psth = psth_data['hsr_all'][cf_idx, freq_idx, db_idx]
                    msr_psth = psth_data['msr_all'][cf_idx, freq_idx, db_idx]
                    lsr_psth = psth_data['lsr_all'][cf_idx, freq_idx, db_idx]
                    
                    # Ensure 1D (squeeze if needed)
                    hsr_psth = np.squeeze(hsr_psth)
                    msr_psth = np.squeeze(msr_psth)
                    lsr_psth = np.squeeze(lsr_psth)
                    
                    # Compute weighted sum
                    weighted_pop_psth = (
                        weights['hsr'] * hsr_psth +
                        weights['msr'] * msr_psth +
                        weights['lsr'] * lsr_psth
                    )
                    
                    weighted_psth[cf][freq][db] = weighted_pop_psth
    
    print(f"✓ Computed weighted population PSTH")
    if has_runs:
        print(f"  Structure: [cf][freq][db][run] = 1D PSTH array")
        print(f"  {len(params['cfs'])} CFs × {len(params['frequencies'])} freqs × {len(params['dbs'])} dBs × {sample_shape[3]} runs")
    else:
        print(f"  Structure: [cf][freq][db] = 1D PSTH array")
        print(f"  {len(params['cfs'])} CFs × {len(params['frequencies'])} freqs × {len(params['dbs'])} dBs")
    
    return weighted_psth, weights
    
    return weighted_psth, weights


def compute_weighted_fiber_sum_meanrates(mean_data, weights=None, normalize=True):
    """
    Compute weighted sum of mean firing rates to create population response.
    
    This operates on already-computed mean rates (after temporal averaging).
    
    Parameters:
    -----------
    mean_data : dict
        Nested dictionary from mean_across_time_psth_bez() or mean_across_time_psth_cochlea()
        Structure: [fiber_type][cf][freq][db][run] = mean_rate (BEZ)
                  [fiber_type][cf][freq][db] = mean_rate (Cochlea)
    weights : dict, optional
        Weights for each fiber type. Default: {'hsr': 0.65, 'msr': 0.25, 'lsr': 0.15}
    normalize : bool, optional
        If True, enforce that weights sum to 1.0. Default: True
    
    Returns:
    --------
    weighted_means : dict
        Nested dictionary with 'population' key:
        {'population': {cf: {freq: {db: {run: value}}}}} for BEZ
        {'population': {cf: {freq: {db: value}}}} for Cochlea
    weights_used : dict
        The actual weights used
    
    Example:
    --------
    >>> weighted, weights = compute_weighted_fiber_sum_meanrates(bez_means)
    >>> # Access like: weighted['population'][cf][freq][db][run]
    """
    # Default physiological ratios
    if weights is None:
        weights = {'hsr': 0.65, 'msr': 0.23, 'lsr': 0.12}
    
    # Validate weights
    fiber_types = ['hsr', 'msr', 'lsr']
    for fiber in fiber_types:
        if fiber not in weights:
            raise ValueError(f"Missing weight for fiber type: {fiber}")
        if fiber not in mean_data:
            raise ValueError(f"Missing fiber type in data: {fiber}")
    
    # Normalize weights to sum to 1.0
    if normalize:
        weight_sum = sum(weights[f] for f in fiber_types)
        if not np.isclose(weight_sum, 1.0):
            print(f"⚠ Normalizing weights (sum was {weight_sum:.4f})")
            weights = {f: weights[f] / weight_sum for f in fiber_types}
    
    # Initialize result with 'population' key
    weighted_means = {'population': {}}
    
    # Determine data structure (BEZ vs Cochlea)
    # Check if there's a 'run' dimension
    sample_fiber = fiber_types[0]
    sample_cf = list(mean_data[sample_fiber].keys())[0]
    sample_freq = list(mean_data[sample_fiber][sample_cf].keys())[0]
    sample_db = list(mean_data[sample_fiber][sample_cf][sample_freq].keys())[0]
    sample_value = mean_data[sample_fiber][sample_cf][sample_freq][sample_db]
    
    has_runs = isinstance(sample_value, dict)
    
    # Iterate through all conditions
    for cf in mean_data[sample_fiber].keys():
        weighted_means['population'][cf] = {}
        
        for freq in mean_data[sample_fiber][cf].keys():
            weighted_means['population'][cf][freq] = {}
            
            for db in mean_data[sample_fiber][cf][freq].keys():
                if has_runs:
                    # BEZ data with runs
                    weighted_means['population'][cf][freq][db] = {}
                    
                    # Get run indices from first fiber type
                    run_indices = mean_data[sample_fiber][cf][freq][db].keys()
                    
                    for run_idx in run_indices:
                        weighted_sum = sum(
                            weights[fiber] * mean_data[fiber][cf][freq][db][run_idx]
                            for fiber in fiber_types
                        )
                        weighted_means['population'][cf][freq][db][run_idx] = weighted_sum
                else:
                    # Cochlea data without runs
                    weighted_sum = sum(
                        weights[fiber] * mean_data[fiber][cf][freq][db]
                        for fiber in fiber_types
                    )
                    weighted_means['population'][cf][freq][db] = weighted_sum
    
    data_type = "BEZ (with runs)" if has_runs else "Cochlea"
    print(f"✓ Computed weighted population mean rates ({data_type})")
    print(f"  Weights: HSR={weights['hsr']:.3f}, MSR={weights['msr']:.3f}, LSR={weights['lsr']:.3f}")
    
    return weighted_means, weights


def save_weighted_population_data(weighted_psth, params, weights, 
                                    filename='weighted_population_psth.mat',
                                    output_dir='model_comparison_PSTH_final'):
    """
    Save weighted population PSTH data to disk for caching.
    
    Parameters:
    -----------
    weighted_psth : ndarray
        Weighted population PSTH array
    params : dict
        Parameters dictionary with 'cfs', 'frequencies', 'dbs'
    weights : dict
        Weights used for combination
    filename : str
        Output filename
    output_dir : str
        Directory to save file
    
    Returns:
    --------
    filepath : str
        Full path to saved file
    """
    from scipy.io import savemat
    from pathlib import Path
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    
    # Prepare data for saving
    save_dict = {
        'population_psth': weighted_psth,
        'cfs': params['cfs'],
        'frequencies': params['frequencies'],
        'dbs': params['dbs'],
        'weight_hsr': weights['hsr'],
        'weight_msr': weights['msr'],
        'weight_lsr': weights['lsr']
    }
    
    # Save to MATLAB format
    savemat(str(filepath), save_dict)
    
    file_size_mb = filepath.stat().st_size / 1024 / 1024
    print(f"✓ Saved weighted population PSTH to: {filepath}")
    print(f"  File size: {file_size_mb:.2f} MB")
    
    return str(filepath)


def load_weighted_population_data(filename='weighted_population_psth.mat',
                                    output_dir='model_comparison_PSTH_final'):
    """
    Load cached weighted population PSTH data from disk.
    
    Parameters:
    -----------
    filename : str
        Input filename
    output_dir : str
        Directory containing file
    
    Returns:
    --------
    weighted_psth : ndarray
        Weighted population PSTH array
    params : dict
        Parameters dictionary
    weights : dict
        Weights used for combination
    """
    from scipy.io import loadmat
    from pathlib import Path
    
    filepath = Path(output_dir) / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Weighted population data not found: {filepath}")
    
    # Load from MATLAB format
    data = loadmat(str(filepath))
    
    weighted_psth = data['population_psth']
    params = {
        'cfs': data['cfs'].flatten(),
        'frequencies': data['frequencies'].flatten(),
        'dbs': data['dbs'].flatten()
    }
    weights = {
        'hsr': float(data['weight_hsr']),
        'msr': float(data['weight_msr']),
        'lsr': float(data['weight_lsr'])
    }
    
    file_size_mb = filepath.stat().st_size / 1024 / 1024
    print(f"✓ Loaded weighted population PSTH from: {filepath}")
    print(f"  Shape: {weighted_psth.shape}")
    print(f"  Weights: HSR={weights['hsr']:.3f}, MSR={weights['msr']:.3f}, LSR={weights['lsr']:.3f}")
    print(f"  File size: {file_size_mb:.2f} MB")
    
    return weighted_psth, params, weights


def prepare_weighted_population_analysis(bez_means, cochlea_means, 
                                          weights=None, normalize=True):
    """
    Convenience function to prepare weighted population data for both BEZ and Cochlea.
    
    Parameters:
    -----------
    bez_means : dict
        BEZ mean rates from mean_across_time_psth_bez()
    cochlea_means : dict
        Cochlea mean rates from mean_across_time_psth_cochlea()
    weights : dict, optional
        Weights for fiber types
    normalize : bool, optional
        Normalize weights to sum to 1.0
    
    Returns:
    --------
    bez_weighted : dict
        BEZ weighted population: {'population': {cf: {freq: {db: {run: value}}}}}
    cochlea_weighted : dict
        Cochlea weighted population: {'population': {cf: {freq: {db: value}}}}
    weights_used : dict
        Weights actually used
    """
    print("="*70)
    print("PREPARING WEIGHTED POPULATION ANALYSIS")
    print("="*70)
    
    # Compute BEZ weighted population
    print("\n1. Computing BEZ weighted population...")
    bez_weighted, weights_used = compute_weighted_fiber_sum_meanrates(
        bez_means, weights=weights, normalize=normalize
    )
    
    # Compute Cochlea weighted population
    print("\n2. Computing Cochlea weighted population...")
    cochlea_weighted, _ = compute_weighted_fiber_sum_meanrates(
        cochlea_means, weights=weights_used, normalize=False  # Already normalized
    )
    
    print("\n" + "="*70)
    print("✓ Weighted population data prepared")
    print("="*70)
    
    return bez_weighted, cochlea_weighted, weights_used
