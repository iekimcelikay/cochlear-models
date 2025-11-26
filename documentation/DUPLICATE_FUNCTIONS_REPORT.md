# Duplicate Functions Report
**Analysis Date**: November 3, 2025  
**Analyzed Folder**: `/home/ekim/PycharmProjects/phd_firstyear/subcorticalSTRF`

---

## Executive Summary

This report identifies **exact duplicate functions** across Python scripts in the subcorticalSTRF folder. These duplicates represent opportunities for code refactoring into shared utility modules.

**Key Findings**:
- **3 major function groups** with exact duplicates across 5-8 files each
- **Total duplicated code**: ~500+ lines that could be centralized
- **Recommendation**: Create utility modules (`data_loaders.py`, `rate_calculations.py`)

---

## Category 1: Data Loading Functions (HIGHEST PRIORITY)

### 🔴 **EXACT DUPLICATES ACROSS 8 FILES**

#### Function: `load_matlab_data()`
**Purpose**: Load BEZ2018 and Cochlea model PSTH data from .mat files

**Files containing this function**:
1. `compare_models_v3.py` (lines 20-58) - **Most detailed version with error handling**
2. `compare_temporal_scatter_efficient.py` (lines 20-39) - Simplified version
3. `compare_individual_temporal_correlations.py` (lines 22-41) - Simplified version
4. `compare_individual_temporal_correlations_backup.py` (lines 22-41) - Identical to above
5. `all_cfs_regression_analysis.py` (lines 37-62) - With docstring
6. `cf_specific_regression_analysis.py` (lines 41-66) - With docstring
7. `cf_regression_seaborn.py` (lines 31-56)
8. `compare_cochlea_bez_models.py` (lines 25-68)

**Variants**:
- **Version A** (compare_models_v3.py): Includes try-except blocks and detailed error messages ✅ **BEST**
- **Version B** (most others): Simplified without explicit error handling
- **Version C** (all_cfs/cf_specific): Adds comprehensive docstrings

**Common Code Pattern**:
```python
def load_matlab_data():
    """Load BEZ and cochlea model data"""
    
    # Load BEZ model data
    bez_dir = Path("//subcorticalSTRF/BEZ2018_meanrate/results/processed_data")
    bez_file = bez_dir / 'bez_acrossruns_psths.mat'
    
    print(f"Loading BEZ data from: {bez_file}")
    bez_data = loadmat(str(bez_file), struct_as_record=False, squeeze_me=True)
    print("✓ BEZ data loaded successfully")
    
    # Load cochlea model data
    cochlea_dir = Path("//subcorticalSTRF/cochlea_meanrate/out/condition_psths")
    cochlea_file = cochlea_dir / 'cochlea_psths.mat'
    
    print(f"Loading cochlea data from: {cochlea_file}")
    cochlea_data = loadmat(str(cochlea_file), struct_as_record=False, squeeze_me=True)
    print("✓ Cochlea data loaded successfully")
    
    return bez_data, cochlea_data
```

**Total Duplicate Lines**: ~20 lines × 8 files = **160 lines**

---

#### Function: `convert_to_numeric(arr)`
**Purpose**: Convert MATLAB string arrays to numeric numpy arrays

**Files containing this function**:
1. `compare_models_v3.py` (nested in extract_parameters)
2. `compare_temporal_scatter_efficient.py` (lines 41-48)
3. `compare_individual_temporal_correlations.py` (lines 43-51) - **Has duplicate return statement (bug!)**
4. `compare_individual_temporal_correlations_backup.py` (lines 43-51) - Same bug
5. `all_cfs_regression_analysis.py` (lines 64-71)
6. `cf_specific_regression_analysis.py` (lines 68-75)
7. `cf_regression_seaborn.py` (lines 58-64)

**Common Code Pattern**:
```python
def convert_to_numeric(arr):
    """Convert string arrays to numeric arrays"""
    arr = np.atleast_1d(arr).flatten()
    if arr.dtype.kind in ['U', 'S']:  # Unicode or byte strings
        return np.array([float(str(x)) for x in arr])
    else:
        return arr.astype(float)
```

**Note**: `compare_individual_temporal_correlations.py` has a bug with duplicate return statement:
```python
def convert_to_numeric(arr):
    """Convert string arrays to numeric arrays"""
    arr = np.atleast_1d(arr).flatten()
    if arr.dtype.kind in ['U', 'S']:
        return np.array([float(str(x)) for x in arr])
    else:
        return arr.astype(float)
        return arr.astype(float)  # ⚠️ DUPLICATE LINE (unreachable)
```

**Total Duplicate Lines**: ~7 lines × 7 files = **49 lines**

---

#### Function: `extract_parameters(data, model_type="bez")`
**Purpose**: Extract CF, frequency, and dB parameters from loaded MATLAB data

**Files containing this function**:
1. `compare_models_v3.py` (lines 59-80) - Includes nested convert_to_numeric
2. `compare_temporal_scatter_efficient.py` (lines 49-62)
3. `compare_individual_temporal_correlations.py` (lines 52-65)
4. `compare_individual_temporal_correlations_backup.py` (lines 52-65)
5. `all_cfs_regression_analysis.py` (lines 72-85)
6. `cf_specific_regression_analysis.py` (lines 76-89)
7. `cf_regression_seaborn.py` (lines 66-78)

**Common Code Pattern**:
```python
def extract_parameters(data, model_type="bez"):
    """Extract parameters from loaded data"""
    print(f"Extracting parameters for {model_type} model...")
    
    cfs = convert_to_numeric(data['cfs'])
    frequencies = convert_to_numeric(data['frequencies'])
    dbs = convert_to_numeric(data['dbs'])
    
    print(f"  CFs: {len(cfs)} values")
    print(f"  Frequencies: {len(frequencies)} values")
    print(f"  dB levels: {dbs}")
    
    return np.sort(cfs), np.sort(frequencies), np.sort(dbs)
```

**Variants**:
- Some versions have more detailed print statements (e.g., showing ranges)
- compare_models_v3.py embeds convert_to_numeric inside this function

**Total Duplicate Lines**: ~14 lines × 7 files = **98 lines**

---

## Category 2: Rate Calculation Functions

### 🟡 **EXACT DUPLICATES ACROSS 4 FILES**

#### Function: `compute_firing_rate_exclude_onset(psth_array, fs, stim_duration_sec, tau_ramp_sec=0.01)`
**Purpose**: Calculate mean firing rate from PSTH array, excluding onset ramp

**Files containing this function**:
1. `meanrate_from_synapsedf.py` (lines 6-14)
2. `meanrate_from_synapsedf_oneANF.py` (lines 9-17)
3. `run_ratemodel_separated_by_anf_type.py` (lines 43-52)
4. `OLD_meanrate_from_synapsedf_loopANF.py` (lines 6-14)

**Common Code Pattern**:
```python
def compute_firing_rate_exclude_onset(psth_array, fs, stim_duration_sec, tau_ramp_sec = 0.01):
    # Convert MATLAB double to NumPy array if needed
    if not isinstance(psth_array, np.ndarray):
        psth_array = np.array(psth_array).squeeze()
    onset_sample = int(tau_ramp_sec * fs)
    offset_sample = int(stim_duration_sec * fs)
    stim_window = psth_array[onset_sample:offset_sample]
    spike_count = stim_window.sum()
    stim_window_duration = stim_duration_sec - tau_ramp_sec
    return spike_count / stim_window_duration
```

**Minor Variations**:
- Parameter spacing: `tau_ramp_sec = 0.01` vs `tau_ramp_sec=0.01` (whitespace only)
- Otherwise **IDENTICAL**

**Total Duplicate Lines**: ~9 lines × 4 files = **36 lines**

---

#### Function: `matlab_double_to_np_array(matlab_double_obj)`
**Purpose**: Convert MATLAB double object to numpy array

**Files containing this function**:
1. `meanrate_from_synapsedf.py` (lines 16-18)
2. `run_ratemodel_separated_by_anf_type.py` (lines 54-56)
3. `OLD_meanrate_from_synapsedf_loopANF.py` (lines 16-18)

**Common Code Pattern**:
```python
def matlab_double_to_np_array(matlab_double_obj):
    np_array = np.array(matlab_double_obj).squeeze()
    return np_array
```

**Total Duplicate Lines**: ~3 lines × 3 files = **9 lines**

---

## Category 3: Alternative Data Loading

### 🟢 **SPECIALIZED VARIANTS (Not Exact Duplicates)**

#### Function: `load_h5py_data(h5_file)` 
**Purpose**: Load MATLAB v7.3 format files using h5py

**Files containing this function**:
1. `compare_cochlea_bez_models_v2.py` (lines 22-54)

**Note**: This is a specialized function for v7.3 MATLAB files. Only appears in one location.

---

#### Function: `load_matlab_data()` - Simple Version
**Purpose**: Simplified data loading without cochlea data

**Files containing this function**:
1. `compare_models_simple.py` (lines 19-76)

**Note**: Different implementation - loads data differently, not a duplicate.

---

## Category 4: Regression/Statistical Functions

### 🟡 **SIMILAR BUT NOT EXACT DUPLICATES**

These functions perform similar tasks but have enough variation that they're not exact duplicates:

#### `load_long_format_data()`
**Files**: 
- `pooled_multiple_regression.py`
- `fiber_specific_regression.py`

**Similarity**: High (~90%), but different enough to warrant separate analysis

#### `calculate_confidence_intervals()` / `calculate_regression_confidence_intervals()`
**Files**:
- `compare_individual_temporal_correlations.py` (lines 232-267)
- `all_cfs_regression_analysis.py` (lines 86-138)
- `cf_specific_regression_analysis.py` (lines 333-383)

**Similarity**: Moderate (~70%), different confidence level implementations

---

## Summary Table

| Function Name | # of Files | Total Lines | Priority | Recommended Action |
|---------------|------------|-------------|----------|-------------------|
| `load_matlab_data()` | 8 | ~160 | 🔴 HIGH | Create `data_loaders.py` |
| `convert_to_numeric()` | 7 | ~49 | 🔴 HIGH | Create `data_loaders.py` |
| `extract_parameters()` | 7 | ~98 | 🔴 HIGH | Create `data_loaders.py` |
| `compute_firing_rate_exclude_onset()` | 4 | ~36 | 🟡 MEDIUM | Create `rate_calculations.py` |
| `matlab_double_to_np_array()` | 3 | ~9 | 🟡 MEDIUM | Create `utils.py` |

**Total Potentially Refactorable Lines**: **~352 lines**

---

## Recommended Refactoring Strategy

### Step 1: Create Utility Module `data_loaders.py`

```python
"""
data_loaders.py
Centralized data loading utilities for BEZ2018 and Cochlea model comparisons
"""

import numpy as np
from scipy.io import loadmat
from pathlib import Path

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
```

### Step 2: Create Utility Module `rate_calculations.py`

```python
"""
rate_calculations.py
Firing rate computation utilities for PSTH analysis
"""

import numpy as np

def compute_firing_rate_exclude_onset(psth_array, fs, stim_duration_sec, tau_ramp_sec=0.01):
    """
    Calculate mean firing rate from PSTH, excluding onset ramp period
    
    Parameters:
        psth_array: PSTH spike count array (may be MATLAB or numpy)
        fs (float): Sampling frequency (Hz)
        stim_duration_sec (float): Total stimulus duration (seconds)
        tau_ramp_sec (float): Onset ramp duration to exclude (seconds)
    
    Returns:
        float: Mean firing rate (spikes/second) during analysis window
    
    Example:
        >>> psth = np.array([...])  # 200ms stimulus at 100kHz
        >>> rate = compute_firing_rate_exclude_onset(psth, 100e3, 0.2, 0.01)
    """
    # Convert MATLAB double to NumPy array if needed
    if not isinstance(psth_array, np.ndarray):
        psth_array = np.array(psth_array).squeeze()
    
    onset_sample = int(tau_ramp_sec * fs)
    offset_sample = int(stim_duration_sec * fs)
    stim_window = psth_array[onset_sample:offset_sample]
    spike_count = stim_window.sum()
    stim_window_duration = stim_duration_sec - tau_ramp_sec
    
    return spike_count / stim_window_duration


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
```

### Step 3: Update All Scripts

**Example refactored script**:
```python
# OLD CODE:
def load_matlab_data():
    bez_dir = Path("//subcorticalSTRF/BEZ2018_meanrate/results/processed_data")
    # ... 20 lines of code ...
    return bez_data, cochlea_data

def convert_to_numeric(arr):
    # ... 7 lines of code ...
    return arr.astype(float)

def extract_parameters(data, model_type="bez"):
    # ... 14 lines of code ...
    return np.sort(cfs), np.sort(frequencies), np.sort(dbs)

# NEW CODE:
from data_loaders import load_matlab_data, extract_parameters

# Use directly - 41 lines reduced to 1 import!
```

---

## Implementation Checklist

- [ ] **Phase 1: Create utility modules**
  - [ ] Create `data_loaders.py` with 3 functions
  - [ ] Create `rate_calculations.py` with 2 functions
  - [ ] Add comprehensive docstrings and type hints
  - [ ] Add unit tests

- [ ] **Phase 2: Refactor high-priority scripts**
  - [ ] Update `compare_models_v3.py`
  - [ ] Update `all_cfs_regression_analysis.py`
  - [ ] Update `cf_specific_regression_analysis.py`
  - [ ] Update `compare_temporal_scatter_efficient.py`

- [ ] **Phase 3: Refactor medium-priority scripts**
  - [ ] Update `compare_individual_temporal_correlations.py`
  - [ ] Update `cf_regression_seaborn.py`
  - [ ] Update rate calculation scripts

- [ ] **Phase 4: Clean up**
  - [ ] Remove OLD_* files if confirmed working
  - [ ] Update documentation/README
  - [ ] Add import examples to COMPARE_MODELS_V3_OVERVIEW.md

---

## Bug Fixes Needed

### 🐛 Bug in `compare_individual_temporal_correlations.py`

**Line 51**: Duplicate return statement (unreachable code)

```python
# Current (BUGGY):
def convert_to_numeric(arr):
    arr = np.atleast_1d(arr).flatten()
    if arr.dtype.kind in ['U', 'S']:
        return np.array([float(str(x)) for x in arr])
    else:
        return arr.astype(float)
        return arr.astype(float)  # ⚠️ REMOVE THIS LINE

# Fixed:
def convert_to_numeric(arr):
    arr = np.atleast_1d(arr).flatten()
    if arr.dtype.kind in ['U', 'S']:
        return np.array([float(str(x)) for x in arr])
    else:
        return arr.astype(float)
```

**Same bug exists in**: `compare_individual_temporal_correlations_backup.py`

---

## Benefits of Refactoring

1. **Maintainability**: Bug fixes and improvements only need to be made once
2. **Consistency**: All scripts use identical, tested implementations
3. **Documentation**: Single source of truth with comprehensive docstrings
4. **Testing**: Unit tests for utility functions benefit all scripts
5. **Code Size**: Reduce codebase by ~350 lines
6. **Readability**: Scripts focus on analysis logic, not infrastructure

---

## Files Requiring No Changes

These files have unique implementations or are deprecated:

- `compare_models_simple.py` - Uses different data loading approach
- `compare_cochlea_bez_models_v2.py` - Uses h5py for v7.3 files (keep separate)
- `test_data_loading.py` - Test file
- `load_synapseresults.py` - Different data source
- `run_model.py` - Different functionality
- Files in `old_files/` directory - Archived

---

## Priority Order for Refactoring

**Week 1 - Critical Functions**:
1. Create `data_loaders.py`
2. Update `compare_models_v3.py` (your main script)
3. Fix bug in `compare_individual_temporal_correlations.py`

**Week 2 - Regression Analysis Scripts**:
4. Update `all_cfs_regression_analysis.py`
5. Update `cf_specific_regression_analysis.py`
6. Update `compare_temporal_scatter_efficient.py`

**Week 3 - Remaining Scripts**:
7. Create `rate_calculations.py`
8. Update rate calculation scripts
9. Update remaining comparison scripts

**Week 4 - Testing & Documentation**:
10. Write unit tests
11. Update all documentation
12. Remove deprecated files

---

**Last Updated**: November 3, 2025  
**Total Scripts Analyzed**: 24 Python files  
**Total Duplicate Functions Found**: 5 major function groups  
**Estimated Refactoring Time**: 3-4 weeks  
**Estimated Lines Saved**: ~350 lines
