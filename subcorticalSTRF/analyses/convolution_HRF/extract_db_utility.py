#!/usr/bin/env python3
"""
Utility functions for extracting and reorganizing BEZ convolved data by dB 
levels.
Created for simplifying data structure from 
[fiber_type][cf][freq][db][run] to [fiber_type][cf][freq][run].

Usage:
    from extract_db_utility import (
        extract_conditions_by_db,
        load_extracted_db_data,
        extract_cochlea_conditions_by_db
    )
"""

import pickle
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from convolution_utils import generate_timestamp

def load_and_combine_bez_files(file_pattern_or_files, fiber_types=['hsr', 'msr', 'lsr']):
    combined_data = {}
    if isinstance(file_pattern_or_files, str):
        # Pattern based loading
        file_pattern = file_pattern_or_files
        for fiber_type in fiber_types:
            filepath = file_pattern.format(fiber_type=fiber_type)
            filepath = Path(filepath)

            if filepath.exists():
                print(f"Loading {fiber_type.upper()} data from: {filepath}")
                with open(filepath, 'rb') as f:
                    fiber_data = pickle.load(f)

                    # Add this fiber type's data to combined structure
                    combined_data[fiber_type] = fiber_data

                print(f"  ✓ Loaded {fiber_type.upper()} data")

            else:
                print(f"  ✗ File not found for {fiber_type.upper()}: {filepath}")
    elif isinstance(file_pattern_or_files, dict):
        # Dictionary based loading
        for fiber_type, filepath in file_pattern_or_files.items():
            filepath = Path(filepath)

            if filepath.exists():
                print(f"Loading {fiber_type.upper()} data from: {filepath}")
                with open(filepath, 'rb') as f:
                    fiber_data = pickle.load(f)

                    # Add this fiber type's data to combined structure
                    combined_data[fiber_type] = fiber_data

                print(f"  ✓ Loaded {fiber_type.upper()} data")

            else:
                print(f"  ✗ File not found for {fiber_type.upper()}: {filepath}")
    print(f"\nCombined data structure created with fiber types: {list(combined_data.keys())}")
    # Quick summary of combined data
    total_conditions = 0
    for fiber_type in combined_data:
        fiber_conditions = 0
        for cf in combined_data[fiber_type]:
            for freq in combined_data[fiber_type][cf]:
                fiber_conditions += len(combined_data[fiber_type][cf][freq])
        print(f"  {fiber_type.upper()}: {fiber_conditions} conditions")
        total_conditions += fiber_conditions

    print(f"Total conditions across all fiber types: {total_conditions}")
    return combined_data

def extract_conditions_by_db(
    bez_data, target_db, save_to_file=None, 
    output_dir='results/extracted'
):
    """
    Extract all conditions with a specific dB level from BEZ data across 
    all fiber types.
    Creates a new data structure organized as 
    [fiber_type][cf_val][freq_val] containing all runs for the specified 
    dB level.

    Parameters:
    - bez_data: Original BEZ data dictionary with structure 
      [fiber_type][cf][freq][db][run] or [fiber_type][cf][freq][db]
    - target_db: The dB level to extract (e.g., 60.0)
    - save_to_file: If True, saves the extracted data to a pickle file
    - output_dir: Directory to save the file

    Returns:
    - extracted_data: Dictionary with structure 
      [fiber_type][cf_val][freq_val] containing run data
    - extraction_summary: Dictionary with statistics about extraction
    """
    print(f"Extracting all conditions with {target_db} dB level...")

    extracted_data = {}
    extraction_summary = {
        'target_db': target_db,
        'total_conditions': 0,
        'total_valid_runs': 0,
        'fiber_type_summary': {}
    }

    for fiber_type in bez_data:
        print(f"\nProcessing {fiber_type.upper()} fibers...")
        extracted_data[fiber_type] = {}

        fiber_conditions = 0
        fiber_valid_runs = 0

        for cf_val in bez_data[fiber_type]:
            extracted_data[fiber_type][cf_val] = {}

            for freq_val in bez_data[fiber_type][cf_val]:
                # Check if target dB exists for this condition
                if target_db in bez_data[fiber_type][cf_val][freq_val]:
                    runs_data = (
                        bez_data[fiber_type][cf_val][freq_val][target_db]
                    )

                    # Handle two cases:
                    # 1. Dictionary of runs (BEZ): {run_idx: array}
                    # 2. Single array (Cochlea): array
                    
                    if isinstance(runs_data, dict):
                        # BEZ format with multiple runs
                        valid_runs = {}
                        valid_count = 0

                        for run_idx, run_data in runs_data.items():
                            if run_data is not None:
                                valid_runs[run_idx] = run_data
                                valid_count += 1

                        # Only include condition if it has valid runs
                        if valid_count > 0:
                            extracted_data[fiber_type][cf_val][
                                freq_val
                            ] = valid_runs
                            fiber_conditions += 1
                            fiber_valid_runs += valid_count

                            print(
                                f"  ✓ CF={cf_val}Hz, "
                                f"Freq={freq_val}Hz: "
                                f"{valid_count} valid runs"
                            )
                        else:
                            print(
                                f"  ✗ CF={cf_val}Hz, "
                                f"Freq={freq_val}Hz: "
                                f"No valid runs"
                            )
                    
                    elif isinstance(runs_data, np.ndarray):
                        # Cochlea format with single response
                        if np.any(np.isfinite(runs_data)):
                            extracted_data[fiber_type][cf_val][
                                freq_val
                            ] = runs_data
                            fiber_conditions += 1
                            fiber_valid_runs += 1
                            
                            print(
                                f"  ✓ CF={cf_val}Hz, "
                                f"Freq={freq_val}Hz: "
                                f"1 response"
                            )
                        else:
                            print(
                                f"  ✗ CF={cf_val}Hz, "
                                f"Freq={freq_val}Hz: "
                                f"No finite values"
                            )
                    else:
                        print(
                            f"  ✗ CF={cf_val}Hz, Freq={freq_val}Hz: "
                            f"Unexpected data type: {type(runs_data)}"
                        )

            # Remove empty CF entries
            if not extracted_data[fiber_type][cf_val]:
                del extracted_data[fiber_type][cf_val]

        # Remove empty fiber type entries
        if not extracted_data[fiber_type]:
            del extracted_data[fiber_type]
            print(f"  No conditions found for {fiber_type.upper()}")
        else:
            extraction_summary['fiber_type_summary'][fiber_type] = {
                'conditions': fiber_conditions,
                'valid_runs': fiber_valid_runs
            }
            print(
                f"  {fiber_type.upper()} summary: {fiber_conditions} "
                f"conditions, {fiber_valid_runs} valid runs"
            )

    # Update total summary
    extraction_summary['total_conditions'] = sum(
        summary['conditions'] for summary in extraction_summary['fiber_type_summary'].values()
    )
    extraction_summary['total_valid_runs'] = sum(
        summary['valid_runs'] for summary in extraction_summary['fiber_type_summary'].values()
    )

    print(f"\n=== Extraction Summary ===")
    print(f"Target dB level: {target_db}")
    print(f"Total conditions extracted: {extraction_summary['total_conditions']}")
    print(f"Total valid runs: {extraction_summary['total_valid_runs']}")

    for fiber_type, summary in extraction_summary['fiber_type_summary'].items():
        print(f"  {fiber_type.upper()}: {summary['conditions']} conditions, {summary['valid_runs']} runs")

    # Save to file if requested
    if save_to_file:
        timestamp = generate_timestamp()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create filename
        filename = f"BEZ_extracted_{target_db}dB_{timestamp}.pkl"
        filepath = output_path / filename

        # Prepare data to save
        save_data = {
            'extracted_data': extracted_data,
            'extraction_summary': extraction_summary,
            'original_structure': '[fiber_type][cf_val][freq_val][run_idx]',
            'extraction_timestamp': timestamp
        }

        # Save to pickle file
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"\n✓ Extracted data saved to: {filepath}")
        return extracted_data, extraction_summary, filepath

    return extracted_data, extraction_summary


def extract_from_separate_files(file_pattern_or_files, target_db, fiber_types=['hsr', 'msr', 'lsr'],
                                save_to_file=None, output_dir='results/extracted'):
    """
    Load separate BEZ files by fiber type, combine them, and extract specific dB level.

    Parameters:
    - file_pattern_or_files: Either pattern string or dict of fiber_type->filepath
    - target_db: The dB level to extract
    - fiber_types: List of fiber types to process
    - save_to_file: If True, saves the extracted data
    - output_dir: Directory to save the file

    Returns:
    - extracted_data: Dictionary with extracted data
    - extraction_summary: Summary statistics
    - filepath: Path to saved file (if save_to_file=True)
    """
    print("=== Loading and Combining Separate BEZ Files ===")

    # Step 1: Load and combine all fiber type files
    combined_data = load_and_combine_bez_files(file_pattern_or_files, fiber_types)

    if not combined_data:
        raise ValueError("No data was successfully loaded from the specified files")

    print(f"\n=== Extracting {target_db} dB Conditions ===")

    # Step 2: Extract the specific dB level
    if save_to_file:
        extracted_data, extraction_summary, filepath = extract_conditions_by_db(
            combined_data, target_db, save_to_file=True, output_dir=output_dir
        )
        return extracted_data, extraction_summary, filepath
    else:
        extracted_data, extraction_summary = extract_conditions_by_db(
            combined_data, target_db, save_to_file=False
        )

        return extracted_data, extraction_summary


def get_available_db_levels(file_pattern_or_files, fiber_types=['hsr', 'msr', 'lsr']):
    """
    Check what dB levels are available across all fiber type files.

    Parameters:
    - file_pattern_or_files: Either pattern string or dict of fiber_type->filepath
    - fiber_types: List of fiber types to check

    Returns:
    - db_summary: Dictionary showing available dB levels per fiber type
    """
    combined_data = load_and_combine_bez_files(file_pattern_or_files, fiber_types)

    db_summary = {}
    all_db_levels = set()

    for fiber_type in combined_data:
        fiber_db_levels = set()
        for cf in combined_data[fiber_type]:
            for freq in combined_data[fiber_type][cf]:
                fiber_db_levels.update(combined_data[fiber_type][cf][freq].keys())

        db_summary[fiber_type] = sorted(list(fiber_db_levels))
        all_db_levels.update(fiber_db_levels)

    db_summary['all_available'] = sorted(list(all_db_levels))

    print("Available dB levels by fiber type:")
    for fiber_type in ['hsr', 'msr', 'lsr']:
        if fiber_type in db_summary:
            print(f"  {fiber_type.upper()}: {db_summary[fiber_type]}")

    print(f"All available dB levels: {db_summary['all_available']}")

    return db_summary

def load_extracted_db_data(filepath):
    """
    Load previously extracted dB-specific data from pickle file.

    Parameters:
    - filepath: Path to the extracted data pickle file

    Returns:
    - extracted_data: The extracted data dictionary
    - extraction_summary: Summary of the extraction
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'rb') as f:
        saved_data = pickle.load(f)

    print(f"Loaded extracted data from: {filepath}")
    print(f"Original extraction timestamp: {saved_data['extraction_timestamp']}")
    print(f"Data structure: {saved_data['original_structure']}")

    extraction_summary = saved_data['extraction_summary']
    print(f"\nExtracted data summary:")
    print(f"  Target dB: {extraction_summary['target_db']}")
    print(f"  Total conditions: {extraction_summary['total_conditions']}")
    print(f"  Total valid runs: {extraction_summary['total_valid_runs']}")

    return saved_data['extracted_data'], saved_data['extraction_summary']


def extract_cochlea_conditions_by_db(
    cochlea_data,
    target_db,
    save_to_file=None,
    output_dir='results/extracted'
):
    """
    Extract all conditions with a specific dB level from Cochlea PSTH data
    across all fiber types.
    Creates a new data structure organized as 
    [fiber_type][cf_val][freq_val] containing the PSTH for the specified dB 
    level.

    Parameters:
    - cochlea_data: Loaded Cochlea MATLAB data structure
    - target_db: The dB level to extract (e.g., 60.0)
    - save_to_file: If True, saves the extracted data to a pickle file
    - output_dir: Directory to save the file

    Returns:
    - extracted_data: Dictionary with structure 
      [fiber_type][cf_val][freq_val] containing PSTH data
    - extraction_summary: Dictionary with statistics about the extraction
    - filepath: Path to saved file (if save_to_file=True)
    """
    print(f"Extracting Cochlea conditions with {target_db} dB level...")

    # Extract parameters from cochlea data
    cfs = np.atleast_1d(convert_to_numeric(cochlea_data['cfs']))
    frequencies = np.atleast_1d(
        convert_to_numeric(cochlea_data['frequencies'])
    )
    dbs = np.atleast_1d(convert_to_numeric(cochlea_data['dbs']))
    
    print(f"Data dimensions:")
    print(f"  CFs: {len(cfs)} values: {cfs}")
    print(f"  Frequencies: {len(frequencies)} values: {frequencies}")
    print(f"  dB levels: {len(dbs)} values: {dbs}")

    # Find the dB index using floating-point tolerant comparison
    db_indices = np.where(np.isclose(dbs, target_db, atol=0.1))[0]
    if len(db_indices) == 0:
        raise ValueError(
            f"Target dB {target_db} not found in data. "
            f"Available dB levels: {dbs}"
        )
    db_idx = db_indices[0]
    print(f"  Target dB {target_db} found at index {db_idx}")

    # Extract PSTH data for each fiber type
    cochlea_rates = {
        'lsr': cochlea_data['lsr_all'],
        'msr': cochlea_data['msr_all'],
        'hsr': cochlea_data['hsr_all']
    }

    extracted_data = {}
    extraction_summary = {
        'target_db': target_db,
        'total_conditions': 0,
        'total_valid_conditions': 0,
        'fiber_type_summary': {}
    }

    fiber_types = ['hsr', 'msr', 'lsr']

    for fiber_type in fiber_types:
        print(f"\nProcessing Cochlea {fiber_type.upper()} fibers...")
        extracted_data[fiber_type] = {}

        psth_data = cochlea_rates[fiber_type]
        print(f"  PSTH data shape: {psth_data.shape}")
        
        fiber_conditions = 0
        fiber_valid_conditions = 0

        # Process each CF
        for cf_idx, cf_val in enumerate(cfs):
            extracted_data[fiber_type][cf_val] = {}

            # Process each tone frequency
            for freq_idx, freq_val in enumerate(frequencies):
                try:
                    # Extract PSTH for this CF, frequency, and target dB
                    # MATLAB indexing: (cf_idx, freq_idx, db_idx)
                    psth_cell = psth_data[cf_idx, freq_idx, db_idx]
                    
                    # Handle MATLAB cell array - extract the actual data
                    if hasattr(psth_cell, 'flatten'):
                        psth = psth_cell.flatten()
                    elif isinstance(psth_cell, np.ndarray):
                        psth = psth_cell.flatten()
                    else:
                        # Try to convert to array
                        psth = np.array(psth_cell).flatten()
                    
                    fiber_conditions += 1

                    # Check for valid (finite) values
                    if len(psth) > 0 and np.any(np.isfinite(psth)):
                        extracted_data[fiber_type][cf_val][freq_val] = psth
                        fiber_valid_conditions += 1
                        
                        if cf_idx == 0 and freq_idx == 0:
                            print(
                                f"  Sample PSTH shape for "
                                f"CF={cf_val}Hz, Freq={freq_val}Hz: "
                                f"{psth.shape}"
                            )
                    else:
                        print(
                            f"  Warning: No finite values for "
                            f"CF={cf_val}Hz, Freq={freq_val}Hz"
                        )

                except (IndexError, ValueError, AttributeError) as e:
                    print(
                        f"  Warning: Skipping CF={cf_val}Hz, "
                        f"Freq={freq_val}Hz due to: {e}"
                    )
                    continue

            # Remove empty CF entries
            if not extracted_data[fiber_type][cf_val]:
                del extracted_data[fiber_type][cf_val]

        # Remove empty fiber type entries
        if not extracted_data[fiber_type]:
            del extracted_data[fiber_type]
            print(f"  No conditions found for {fiber_type.upper()}")
        else:
            extraction_summary['fiber_type_summary'][fiber_type] = {
                'conditions': fiber_conditions,
                'valid_conditions': fiber_valid_conditions
            }
            print(
                f"  {fiber_type.upper()} summary: {fiber_conditions} "
                f"conditions, {fiber_valid_conditions} valid"
            )

    # Update total summary
    extraction_summary['total_conditions'] = sum(
        summary['conditions']
        for summary in extraction_summary['fiber_type_summary'].values()
    )
    extraction_summary['total_valid_conditions'] = sum(
        summary['valid_conditions']
        for summary in extraction_summary['fiber_type_summary'].values()
    )

    print(f"\n=== Cochlea Extraction Summary ===")
    print(f"Target dB level: {target_db}")
    print(
        f"Total conditions extracted: "
        f"{extraction_summary['total_conditions']}"
    )
    print(
        f"Total valid conditions: "
        f"{extraction_summary['total_valid_conditions']}"
    )

    for fiber_type, summary in (
        extraction_summary['fiber_type_summary'].items()
    ):
        print(
            f"  {fiber_type.upper()}: {summary['conditions']} conditions, "
            f"{summary['valid_conditions']} valid"
        )

    # Save to file if requested
    if save_to_file:
        timestamp = generate_timestamp()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create filename without timestamp to match BEZ convention
        filename = f"Cochlea_extracted_{target_db}dB.pkl"
        filepath = output_path / filename

        # Prepare data to save
        save_data = {
            'extracted_data': extracted_data,
            'extraction_summary': extraction_summary,
            'original_structure': '[fiber_type][cf_val][freq_val]',
            'extraction_timestamp': timestamp,
            'model': 'cochlea',
            'parameters': {
                'cfs': cfs.tolist(),
                'frequencies': frequencies.tolist(),
                'dbs': dbs.tolist()
            }
        }

        # Save to pickle file
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"\n✓ Cochlea extracted data saved to: {filepath}")
        return extracted_data, extraction_summary, filepath

    return extracted_data, extraction_summary


def extract_cochlea_from_mat_file(
    mat_filepath,
    target_db,
    save_to_file=True,
    output_dir='results/extracted'
):
    """
    Load Cochlea PSTH data from MATLAB file and extract specific dB level.

    Parameters:
    - mat_filepath: Path to Cochlea PSTH MATLAB file
    - target_db: The dB level to extract
    - save_to_file: If True, saves the extracted data
    - output_dir: Directory to save the file

    Returns:
    - extracted_data: Dictionary with extracted data
    - extraction_summary: Summary statistics
    - filepath: Path to saved file (if save_to_file=True)
    """
    print("=== Loading Cochlea PSTH Data from MATLAB File ===")

    mat_filepath = Path(mat_filepath)
    if not mat_filepath.exists():
        raise FileNotFoundError(f"MATLAB file not found: {mat_filepath}")

    print(f"Loading from: {mat_filepath}")
    cochlea_data = loadmat(
        str(mat_filepath),
        struct_as_record=False,
        squeeze_me=True
    )
    print("✓ Cochlea PSTH data loaded successfully")

    # Extract the specific dB level
    print(f"\n=== Extracting {target_db} dB Conditions ===")

    if save_to_file:
        extracted_data, extraction_summary, filepath = (
            extract_cochlea_conditions_by_db(
                cochlea_data,
                target_db,
                save_to_file=True,
                output_dir=output_dir
            )
        )
        return extracted_data, extraction_summary, filepath
    else:
        extracted_data, extraction_summary = (
            extract_cochlea_conditions_by_db(
                cochlea_data,
                target_db,
                save_to_file=False
            )
        )
        return extracted_data, extraction_summary


def get_available_cochlea_db_levels(mat_filepath):
    """
    Check what dB levels are available in Cochlea MATLAB file.

    Parameters:
    - mat_filepath: Path to Cochlea PSTH MATLAB file

    Returns:
    - db_levels: Sorted array of available dB levels
    """
    mat_filepath = Path(mat_filepath)
    if not mat_filepath.exists():
        raise FileNotFoundError(f"MATLAB file not found: {mat_filepath}")

    print(f"Loading Cochlea data from: {mat_filepath}")
    cochlea_data = loadmat(
        str(mat_filepath),
        struct_as_record=False,
        squeeze_me=True
    )

    dbs = np.sort(convert_to_numeric(cochlea_data['dbs']))

    print("Available dB levels in Cochlea data:")
    print(f"  {dbs}")

    return dbs


def convert_to_numeric(array):
    """
    Convert a MATLAB-style array to a numeric numpy array.
    Handles scalars, lists, and nested MATLAB structures.
    
    This function recursively processes MATLAB data structures to extract
    numeric values, compatible with both BEZ and Cochlea model data formats.
    
    Parameters:
    - array: The input from MATLAB (scalar, list, array, or nested struct)

    Returns:
    - numeric_array: A 1D numpy array with numeric values
    """
    # Try simple conversion first for basic numpy arrays
    if isinstance(array, np.ndarray):
        # If it's already a numeric array, just flatten and return
        if array.dtype.kind in ['i', 'f', 'u']:  # int, float, unsigned
            return array.flatten()
        # Handle object dtype arrays
        elif array.dtype == object:
            result = []
            for item in array.flat:
                result.extend(convert_to_numeric(item).flatten())
            return np.array(result, dtype=float)
    
    # Handle scalar values
    if np.isscalar(array):
        return np.array([float(array)])
    
    # Handle lists and tuples
    if isinstance(array, (list, tuple)):
        try:
            # Try direct conversion
            return np.array(array, dtype=float).flatten()
        except (ValueError, TypeError):
            # If that fails, recursively process each element
            result = []
            for item in array:
                result.extend(convert_to_numeric(item).flatten())
            return np.array(result, dtype=float)
    
    # Handle MATLAB structs with _item attribute
    if hasattr(array, '_item'):
        return convert_to_numeric(array._item)
    
    # Last resort: try to convert to float
    try:
        return np.array([float(array)])
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Cannot convert to numeric array. Type: {type(array)}, "
            f"Value: {array}"
        ) from e


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract specific dB level conditions from BEZ or "
                    "Cochlea files"
    )
    parser.add_argument(
        '--model',
        choices=['bez', 'cochlea'],
        default='bez',
        help='Model type to process'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        help='File pattern with {fiber_type} placeholder (for BEZ)'
    )
    parser.add_argument(
        '--mat-file',
        type=str,
        help='Path to Cochlea MATLAB file'
    )
    parser.add_argument(
        '--db',
        type=float,
        required=True,
        help='Target dB level to extract'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/extracted',
        help='Output directory'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save extracted data to file'
    )
    parser.add_argument(
        '--fiber-types',
        nargs='+',
        default=['hsr', 'msr', 'lsr'],
        help='Fiber types to process (for BEZ)'
    )

    args = parser.parse_args()

    if args.model == 'bez':
        if not args.pattern:
            parser.error("--pattern required for BEZ model")

        extracted_data, summary = extract_from_separate_files(
            file_pattern_or_files=args.pattern,
            target_db=args.db,
            fiber_types=args.fiber_types,
            save_to_file=args.save,
            output_dir=args.output_dir
        )
    elif args.model == 'cochlea':
        if not args.mat_file:
            parser.error("--mat-file required for Cochlea model")

        if args.save:
            extracted_data, summary, filepath = (
                extract_cochlea_from_mat_file(
                    args.mat_file,
                    args.db,
                    save_to_file=True,
                    output_dir=args.output_dir
                )
            )
        else:
            extracted_data, summary = extract_cochlea_from_mat_file(
                args.mat_file,
                args.db,
                save_to_file=False
            )

    print("\n✓ Extraction complete")


