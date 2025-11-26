#!/usr/bin/env python3
"""
Test script to verify data loading from MATLAB files
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
import h5py
from pathlib import Path

# Set style
plt.style.use('default')
sns.set_theme(style="whitegrid")

def test_data_loading():
    """Test loading of MATLAB data files"""
    
    # Test BEZ model data
    bez_dir = Path("//subcorticalSTRF/BEZ2018_meanrate/results/processed_data")
    bez_file = bez_dir / 'bez_acrossruns_psths.mat'
    
    print(f"Checking BEZ file: {bez_file}")
    print(f"BEZ file exists: {bez_file.exists()}")
    
    if bez_file.exists():
        print("Loading BEZ data...")
        try:
            bez_data = loadmat(str(bez_file), struct_as_record=False, squeeze_me=True)
            print(f"BEZ data keys: {list(bez_data.keys())}")
        except NotImplementedError:
            print("Using h5py for MATLAB v7.3 format...")
            with h5py.File(bez_file, 'r') as f:
                bez_data = {}
                for key in f.keys():
                    if not key.startswith('__'):
                        bez_data[key] = f[key]
                print(f"BEZ data keys: {list(bez_data.keys())}")
                # Show structure of data
                for key, value in bez_data.items():
                    print(f"  {key}: {type(value)} {getattr(value, 'shape', 'no shape')}")
        except Exception as e:
            print(f"Error loading BEZ data: {e}")
    
    # Test cochlea model data  
    cochlea_dir = Path("//subcorticalSTRF/cochlea_meanrate/out/condition_psths")
    cochlea_file = cochlea_dir / 'cochlea_psths.mat'
    
    print(f"\nChecking cochlea file: {cochlea_file}")
    print(f"Cochlea file exists: {cochlea_file.exists()}")
    
    if cochlea_file.exists():
        print("Loading cochlea data...")
        try:
            cochlea_data = loadmat(str(cochlea_file), struct_as_record=False, squeeze_me=True)
            print(f"Cochlea data keys: {list(cochlea_data.keys())}")
        except NotImplementedError:
            print("Using h5py for MATLAB v7.3 format...")
            with h5py.File(cochlea_file, 'r') as f:
                cochlea_data = {}
                for key in f.keys():
                    if not key.startswith('__'):
                        cochlea_data[key] = f[key]
                print(f"Cochlea data keys: {list(cochlea_data.keys())}")
                # Show structure of data
                for key, value in cochlea_data.items():
                    print(f"  {key}: {type(value)} {getattr(value, 'shape', 'no shape')}")
        except Exception as e:
            print(f"Error loading cochlea data: {e}")

if __name__ == "__main__":
    test_data_loading()