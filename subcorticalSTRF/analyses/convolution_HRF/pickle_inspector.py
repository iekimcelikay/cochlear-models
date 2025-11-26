#!/usr/bin/env python3
"""
Utility functions for inspecting and visualizing pickle files in Jupyter notebooks.

Functions:
- explore_pickle_structure: Recursively explore nested data structures
- create_summary_dataframe: Create pandas DataFrame from extracted BEZ data
- plot_sample_runs: Visualize sample runs from extracted data
- inspect_pickle_file: Quick inspector for any pickle file
- compare_pickle_files: Compare multiple pickle files
"""

import pickle
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


def explore_pickle_structure(
    obj,
    max_depth=4,
    current_depth=0,
    max_items=5,
    prefix=""
):
    """
    Recursively explore and display the structure of a pickle object.
    
    Parameters:
    - obj: The object to explore
    - max_depth: Maximum depth to traverse
    - current_depth: Current recursion depth
    - max_items: Maximum items to show at each level
    - prefix: Indentation prefix for display
    """
    if current_depth >= max_depth:
        print(f"{prefix}... (max depth reached)")
        return
    
    indent = "  " * current_depth
    
    if isinstance(obj, dict):
        print(f"{indent}Dictionary with {len(obj)} keys:")
        for i, (key, value) in enumerate(list(obj.items())[:max_items]):
            print(f"{indent}  [{key}] ({type(value).__name__}):")
            explore_pickle_structure(
                value,
                max_depth,
                current_depth + 1,
                max_items,
                prefix
            )
        if len(obj) > max_items:
            print(f"{indent}  ... and {len(obj) - max_items} more keys")
    
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}{type(obj).__name__} with {len(obj)} items:")
        if len(obj) > 0:
            print(f"{indent}  First item ({type(obj[0]).__name__}):")
            explore_pickle_structure(
                obj[0],
                max_depth,
                current_depth + 1,
                max_items,
                prefix
            )
        if len(obj) > 1:
            print(f"{indent}  ... and {len(obj) - 1} more items")
    
    elif isinstance(obj, np.ndarray):
        print(f"{indent}NumPy array: shape={obj.shape}, "
              f"dtype={obj.dtype}")
        if obj.size < 10:
            print(f"{indent}  Values: {obj}")
    
    else:
        print(f"{indent}{type(obj).__name__}: {str(obj)[:100]}")


def create_summary_dataframe(extracted_data):
    """
    Create a summary DataFrame from extracted BEZ data.
    
    Parameters:
    - extracted_data: Extracted data dictionary
    
    Returns:
    - DataFrame with summary information
    """
    summary_rows = []
    
    for fiber_type in extracted_data:
        for cf_val in extracted_data[fiber_type]:
            for freq_val in extracted_data[fiber_type][cf_val]:
                runs = extracted_data[fiber_type][cf_val][freq_val]
                num_runs = len(runs)
                
                # Get sample data shape if available
                if num_runs > 0:
                    first_run_idx = list(runs.keys())[0]
                    first_run_data = runs[first_run_idx]
                    if isinstance(first_run_data, np.ndarray):
                        data_shape = first_run_data.shape
                        data_size = first_run_data.size
                    else:
                        data_shape = 'N/A'
                        data_size = 'N/A'
                else:
                    data_shape = 'N/A'
                    data_size = 'N/A'
                
                summary_rows.append({
                    'Fiber Type': fiber_type.upper(),
                    'CF (Hz)': cf_val,
                    'Frequency (Hz)': freq_val,
                    'Num Runs': num_runs,
                    'Data Shape': str(data_shape),
                    'Data Size': data_size
                })
    
    return pd.DataFrame(summary_rows)


def plot_sample_runs(extracted_data, fiber_type='hsr', num_samples=3):
    """
    Plot sample runs from extracted data.
    
    Parameters:
    - extracted_data: Extracted data dictionary
    - fiber_type: Fiber type to visualize
    - num_samples: Number of sample runs to plot
    """
    if fiber_type not in extracted_data:
        print(f"Fiber type '{fiber_type}' not found")
        return
    
    # Get first few conditions
    sample_count = 0
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for cf_val in extracted_data[fiber_type]:
        for freq_val in extracted_data[fiber_type][cf_val]:
            if sample_count >= num_samples:
                break
            
            runs = extracted_data[fiber_type][cf_val][freq_val]
            first_run_idx = list(runs.keys())[0]
            run_data = runs[first_run_idx]
            
            if isinstance(run_data, np.ndarray):
                ax = axes[sample_count]
                
                # Plot based on dimensionality
                if run_data.ndim == 1:
                    ax.plot(run_data)
                    ax.set_ylabel('Amplitude')
                elif run_data.ndim == 2:
                    im = ax.imshow(
                        run_data,
                        aspect='auto',
                        cmap='viridis'
                    )
                    plt.colorbar(im, ax=ax)
                
                ax.set_title(
                    f'{fiber_type.upper()}: CF={cf_val}Hz, '
                    f'Freq={freq_val}Hz, Run {first_run_idx}'
                )
                ax.set_xlabel('Time samples' if run_data.ndim == 1 
                             else 'Time')
                
                sample_count += 1
        
        if sample_count >= num_samples:
            break
    
    plt.tight_layout()
    plt.show()


def inspect_pickle_file(filepath, detailed=True):
    """
    Quick inspector for pickle files.
    
    Parameters:
    - filepath: Path to pickle file
    - detailed: If True, shows detailed structure exploration
    
    Returns:
    - Loaded data object or None if error
    """
    filepath = Path(filepath)
    
    # File information
    print("=" * 70)
    print("PICKLE FILE INSPECTOR")
    print("=" * 70)
    print(f"File: {filepath.name}")
    print(f"Location: {filepath.parent}")
    print(f"Size: {os.path.getsize(filepath) / 1024:.2f} KB")
    print(f"Exists: {filepath.exists()}")
    print()
    
    if not filepath.exists():
        print("File not found!")
        return None
    
    # Load and inspect
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print("Successfully loaded!")
        print(f"Root type: {type(data).__name__}")
        print()
        
        # Type-specific inspection
        if isinstance(data, dict):
            print(f"Dictionary with {len(data)} keys:")
            for key in data.keys():
                value = data[key]
                print(f"  - '{key}': {type(value).__name__}", end="")
                
                if isinstance(value, dict):
                    print(f" ({len(value)} items)")
                elif isinstance(value, (list, tuple)):
                    print(f" ({len(value)} elements)")
                elif isinstance(value, np.ndarray):
                    print(f" shape={value.shape}")
                else:
                    print()
            
            if detailed:
                print("\n" + "=" * 70)
                print("DETAILED STRUCTURE")
                print("=" * 70)
                explore_pickle_structure(data, max_depth=4, max_items=3)
        
        elif isinstance(data, np.ndarray):
            print("NumPy Array:")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Size: {data.size} elements")
            print(f"  Memory: {data.nbytes / 1024:.2f} KB")
        
        return data
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def compare_pickle_files(filepaths):
    """
    Compare multiple pickle files.
    
    Parameters:
    - filepaths: List of file paths to compare
    
    Returns:
    - DataFrame with comparison results
    """
    print("=" * 70)
    print("PICKLE FILE COMPARISON")
    print("=" * 70)
    
    results = []
    
    for filepath in filepaths:
        filepath = Path(filepath)
        
        if not filepath.exists():
            results.append({
                'File': filepath.name,
                'Status': 'Not Found',
                'Size (KB)': 0,
                'Type': 'N/A'
            })
            continue
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Get summary info
            file_size = os.path.getsize(filepath) / 1024
            data_type = type(data).__name__
            
            # Count items if dict
            if isinstance(data, dict):
                num_items = len(data)
                type_info = f"Dict ({num_items} keys)"
            elif isinstance(data, (list, tuple)):
                num_items = len(data)
                type_info = f"{data_type} ({num_items} items)"
            elif isinstance(data, np.ndarray):
                type_info = f"Array {data.shape}"
            else:
                type_info = data_type
            
            results.append({
                'File': filepath.name,
                'Status': '✓ Loaded',
                'Size (KB)': f"{file_size:.2f}",
                'Type': type_info
            })
            
        except Exception as e:
            results.append({
                'File': filepath.name,
                'Status': f'Error: {str(e)[:30]}',
                'Size (KB)': f"{os.path.getsize(filepath) / 1024:.2f}",
                'Type': 'N/A'
            })
    
    # Display comparison table
    df_comparison = pd.DataFrame(results)
    display(df_comparison)
    
    return df_comparison
