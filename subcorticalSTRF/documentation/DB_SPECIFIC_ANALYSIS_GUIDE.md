# dB-Specific Analysis Guide

## Overview

**IMPORTANT:** All regression analyses should be conducted **WITHIN a single dB level** to ensure meaningful comparisons. Mixing data across different sound pressure levels can confound results because neural responses change dramatically with intensity.

## Why Analyze Within dB Levels?

1. **Physiological Validity**: Neurons respond differently at different intensities
2. **Rate-Level Functions**: Firing rates are non-linear with dB level
3. **Saturation Effects**: High dB levels may saturate responses
4. **Threshold Effects**: Low dB levels may be near threshold

## Updated Architecture

### New Function: `extract_data_at_db()`

```python
from subcorticalSTRF.data_loader import extract_data_at_db

# Filter data to a single dB level
cochlea_70db = extract_data_at_db(cochlea_means, db_level=70.0)
bez_70db = extract_data_at_db(bez_means, db_level=70.0)

# Now both datasets only contain data at 70 dB
# Structure: [fiber_type][cf][freq][run] (dB dimension removed)
```

### Updated Visualization Functions

All plotting functions now support `db_level` parameter:

```python
# Scatter plot at specific dB
plot_regression_scatter(cochlea, bez, run_idx=0, db_level=70.0)

# Fiber-type comparison at specific dB
plot_regression_by_fiber_type(cochlea, bez, run_idx=0, db_level=70.0)

# Slope distribution at specific dB
plot_slope_distribution(cochlea, bez, db_level=70.0)

# All fiber types, specific dB
plot_slope_distribution_by_fiber_type(cochlea, bez, db_level=70.0)
```

## Jupyter Notebook Workflow

The updated `test_regression_workflow.ipynb` follows this structure:

### Step 1-3: Load Data and Compute Means
```python
bez_data, params = load_bez_psth_data()
cochlea_data = ...  # Load your cochlea data
cochlea_means = mean_across_time_psth_cochlea(cochlea_data, params)
bez_means = mean_across_time_psth_bez(bez_data, params)
```

### Step 4: Select dB Level
```python
selected_db = 70.0  # IMPORTANT: Choose your dB level
selected_run = 0
```

### Step 5: Filter to Selected dB
```python
# Extract ONLY data at selected dB level
cochlea_db = extract_data_at_db(cochlea_means, selected_db)
bez_db = extract_data_at_db(bez_means, selected_db)
```

### Step 6: Extract Data Pairs
```python
def extract_pairs_at_db(cochlea_db_data, bez_db_data, fiber_type, run_idx):
    """Extract (Cochlea, BEZ) pairs at a single dB level"""
    cochlea_vals = []
    bez_vals = []
    
    fiber_cochlea = cochlea_db_data[fiber_type]
    fiber_bez = bez_db_data[fiber_type]
    
    for cf in fiber_cochlea.keys():
        if cf not in fiber_bez:
            continue
        for freq in fiber_cochlea[cf].keys():
            if freq not in fiber_bez[cf]:
                continue
            
            cochlea_val = fiber_cochlea[cf][freq]  # No dB dimension
            bez_val = fiber_bez[cf][freq][run_idx]  # Still has run dimension
            
            cochlea_vals.append(cochlea_val)
            bez_vals.append(bez_val)
    
    return np.array(cochlea_vals), np.array(bez_vals)

# Extract for each fiber type
data_pairs = {}
for fiber in ['hsr', 'msr', 'lsr']:
    cochlea_vals, bez_vals = extract_pairs_at_db(cochlea_db, bez_db, fiber, selected_run)
    data_pairs[fiber] = (cochlea_vals, bez_vals)
```

### Step 7-10: Regression and Visualization
```python
# Compute regression for each fiber type
for fiber in ['hsr', 'msr', 'lsr']:
    cochlea_vals, bez_vals = data_pairs[fiber]
    slope, intercept, r_value, p_value, stderr = linregress(cochlea_vals, bez_vals)
    # ... plot results
```

### Optional: Analyze Across All dB Levels
```python
all_db_results = {}

for db_level in params['dbs']:
    cochlea_db_temp = extract_data_at_db(cochlea_means, db_level)
    bez_db_temp = extract_data_at_db(bez_means, db_level)
    
    all_db_results[db_level] = {}
    
    for fiber in ['hsr', 'msr', 'lsr']:
        c_vals, b_vals = extract_pairs_at_db(cochlea_db_temp, bez_db_temp, fiber, selected_run)
        slope, intercept, r_value, p_value, stderr = linregress(c_vals, b_vals)
        
        all_db_results[db_level][fiber] = {
            'slope': slope,
            'r_squared': r_value**2,
            'n_points': len(c_vals)
        }

# Compare slopes across dB levels
print("SUMMARY: Regression Slopes Across dB Levels")
for db_level in sorted(all_db_results.keys()):
    hsr_slope = all_db_results[db_level]['hsr']['slope']
    msr_slope = all_db_results[db_level]['msr']['slope']
    lsr_slope = all_db_results[db_level]['lsr']['slope']
    print(f"{db_level} dB: HSR={hsr_slope:.3f}, MSR={msr_slope:.3f}, LSR={lsr_slope:.3f}")
```

## Key Points

✅ **DO**: Always specify `db_level` parameter for meaningful comparisons
✅ **DO**: Analyze each dB level separately
✅ **DO**: Compare results across dB levels to understand intensity effects

⚠️ **DON'T**: Mix data from different dB levels in a single regression
⚠️ **DON'T**: Pool across dB levels unless you have a specific scientific reason

## Example Output

```
Analyzing 50 dB
============================================================
  HSR: slope=0.856, R²=0.892, N=400
  MSR: slope=0.923, R²=0.847, N=400
  LSR: slope=0.734, R²=0.801, N=400

Analyzing 60 dB
============================================================
  HSR: slope=0.912, R²=0.908, N=400
  MSR: slope=0.945, R²=0.873, N=400
  LSR: slope=0.789, R²=0.824, N=400

Analyzing 70 dB
============================================================
  HSR: slope=0.967, R²=0.925, N=400
  MSR: slope=0.981, R²=0.891, N=400
  LSR: slope=0.845, R²=0.856, N=400

Analyzing 80 dB
============================================================
  HSR: slope=1.023, R²=0.887, N=400
  MSR: slope=1.012, R²=0.834, N=400
  LSR: slope=0.901, R²=0.782, N=400

SUMMARY: Regression Slopes Across dB Levels
============================================================
dB Level     HSR Slope       MSR Slope       LSR Slope       
------------------------------------------------------------
50.0         0.8560          0.9230          0.7340          
60.0         0.9120          0.9450          0.7890          
70.0         0.9670          0.9810          0.8450          
80.0         1.0230          1.0120          0.9010          
============================================================
```

## Scientific Interpretation

- **Slope < 1**: BEZ underestimates firing rates relative to Cochlea
- **Slope ≈ 1**: BEZ and Cochlea have similar firing rates
- **Slope > 1**: BEZ overestimates firing rates relative to Cochlea

Changes in slope across dB levels reveal:
- How models differ in their intensity encoding
- Whether models have different saturation characteristics
- Fiber-type specific differences in rate-level functions
