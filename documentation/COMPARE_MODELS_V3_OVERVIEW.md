# Overview: compare_models_v3.py

**Script Purpose**: Statistical comparison of BEZ2018 and Cochlea auditory nerve fiber models based on mean PSTH firing rates.

**Last Updated**: October 28, 2024  
**Lines of Code**: 513  
**Language**: Python 3

---

## Table of Contents
- [Planned Changes](#planned-changes)
- [Quick Summary](#quick-summary)
- [Prerequisites](#prerequisites)
- [Workflow Diagram](#workflow-diagram)
- [Detailed Function Reference](#detailed-function-reference)
- [Data Structures](#data-structures)
- [Visualization Features](#visualization-features)
- [Output Files](#output-files)
- [Interpretation Guide](#interpretation-guide)
- [Troubleshooting](#troubleshooting)

---

## Planned Changes

### TODO List

**Priority: High**
- [ ] 

**Priority: Medium**
- [ ] 

**Priority: Low**
- [ ] 

### Completed Changes
- [ ] *Date*: Description of change

### Notes & Ideas
- I need to change the loaded files of BEZ. I want to compare the means of each run individually, not the aggregated response of 10 runs. 
- For this, I can use the loading functions of the convolution script. (I actually moved this function to data_loaders.py )
``` python
from data_loaders.py import load_bez_psth_data

```


---

## Quick Summary

This script performs a comprehensive statistical comparison between two auditory nerve (AN) fiber models:

- **BEZ2018 Model**: Mechanistic model by Bruce et al. (2018)
- **Cochlea Model**: Phenomenological model (Python `cochlea` package)

**What it does:**
1. Loads PSTH (Peristimulus Time Histogram) data from both models
2. Extracts mean firing rates across all experimental conditions
3. Performs linear regression to quantify model agreement
4. Creates publication-quality scatter plots with advanced visual encoding
5. Generates comprehensive statistical summaries

**Key Output**: Scatter plots showing BEZ vs. Cochlea firing rates, colored by characteristic frequency (CF) and brightness-coded by frequency distance from CF.

---

## Prerequisites

### Required Input Files

```
//subcorticalSTRF/BEZ2018_meanrate/results/processed_data/
└── bez_acrossruns_psths.mat

//subcorticalSTRF/cochlea_meanrate/out/condition_psths/
└── cochlea_psths.mat
```

### MATLAB Preprocessing Scripts
Must be run BEFORE this script:
- `bez_compute_mean_across_runs.m` - Aggregates BEZ model PSTH data
- `aggregate_cochlea_psths.m` - Aggregates Cochlea model PSTH data

### Python Dependencies
```python
numpy
pandas
matplotlib
seaborn
scipy
pathlib
```

### Expected Data Format

**BEZ Model Structure:**
```matlab
bez_acrossruns_psths.mat:
├── cfs          [1 × N_cf]        % Characteristic frequencies
├── frequencies  [1 × N_freq]      % Stimulus frequencies
├── dbs          [1 × N_db]        % dB SPL levels
└── bez_rates    (struct)
    ├── lsr      [N_cf × N_freq × N_db × N_time]
    ├── msr      [N_cf × N_freq × N_db × N_time]
    └── hsr      [N_cf × N_freq × N_db × N_time]
```

**Cochlea Model Structure:**
```matlab
cochlea_psths.mat:
├── cfs          [1 × N_cf]
├── frequencies  [1 × N_freq]
├── dbs          [1 × N_db]
├── lsr_all      [N_cf × N_freq × N_db × N_time]
├── msr_all      [N_cf × N_freq × N_db × N_time]
└── hsr_all      [N_cf × N_freq × N_db × N_time]
```

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATA LOADING                    │
├─────────────────────────────────────────────────────────────┤
│  load_matlab_data()                                         │
│  ├─ Load bez_acrossruns_psths.mat (scipy.io.loadmat)      │
│  ├─ Load cochlea_psths.mat (scipy.io.loadmat)             │
│  └─ Return: bez_data, cochlea_data dictionaries           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                PHASE 2: PARAMETER EXTRACTION                │
├─────────────────────────────────────────────────────────────┤
│  extract_parameters()                                       │
│  ├─ Convert MATLAB arrays to numeric (handle strings)      │
│  ├─ Extract: CFs, Frequencies, dB levels                   │
│  └─ Return: Sorted numpy arrays                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 PHASE 3: PSTH DATA EXTRACTION               │
├─────────────────────────────────────────────────────────────┤
│  extract_psth_data()                                        │
│  ├─ Extract fiber type arrays: LSR, MSR, HSR               │
│  ├─ Handle different MATLAB structure formats              │
│  └─ Return: Dictionary of 4D arrays per fiber type         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              PHASE 4: DATAFRAME CONSTRUCTION                │
├─────────────────────────────────────────────────────────────┤
│  create_comparison_dataframe()                              │
│                                                             │
│  For each (CF, Freq, dB, FiberType):                       │
│    1. Extract BEZ PSTH[time]                               │
│    2. Extract Cochlea PSTH[time]                           │
│    3. Calculate mean(BEZ PSTH) → bez_rate                  │
│    4. Calculate mean(Cochlea PSTH) → cochlea_rate          │
│    5. Calculate freq_diff = Freq - CF                      │
│    6. Append to comparison list                            │
│                                                             │
│  Return: Pandas DataFrame with columns:                    │
│          [cf, frequency, db_level, fiber_type,             │
│           bez_rate, cochlea_rate, freq_diff, cf_group]     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│           PHASE 5: VISUALIZATION & REGRESSION               │
├─────────────────────────────────────────────────────────────┤
│  create_comparison_plots()                                  │
│                                                             │
│  For each fiber type (LSR, MSR, HSR):                      │
│    ┌─────────────────────────────────────────┐            │
│    │ 1. Data Cleaning                        │            │
│    │    - Remove NaN/Inf values              │            │
│    │    - Filter valid points                │            │
│    └─────────────────────────────────────────┘            │
│    ┌─────────────────────────────────────────┐            │
│    │ 2. Visual Encoding                      │            │
│    │    - Color = CF (Spectral palette)      │            │
│    │    - Alpha = Distance from CF           │            │
│    │      α = 0.1 + 0.9*(1-norm_diff)²       │            │
│    └─────────────────────────────────────────┘            │
│    ┌─────────────────────────────────────────┐            │
│    │ 3. Statistical Analysis                 │            │
│    │    - scipy.linregress()                 │            │
│    │    - numpy.polyfit() [MATLAB verify]    │            │
│    │    - Calculate: slope, R², p-value      │            │
│    └─────────────────────────────────────────┘            │
│    ┌─────────────────────────────────────────┐            │
│    │ 4. Plot Generation                      │            │
│    │    - Scatter: BEZ vs Cochlea rates      │            │
│    │    - Regression line                    │            │
│    │    - Perfect match diagonal (y=x)       │            │
│    │    - Statistics text box                │            │
│    │    - Custom legend (all CFs + alpha)    │            │
│    └─────────────────────────────────────────┘            │
│    ┌─────────────────────────────────────────┐            │
│    │ 5. Save Outputs                         │            │
│    │    - PNG (300 DPI)                      │            │
│    │    - PDF (vector graphics)              │            │
│    └─────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              PHASE 6: STATISTICAL SUMMARY                   │
├─────────────────────────────────────────────────────────────┤
│  main() - Summary Generation                                │
│  ├─ Console table (Slope, R², p-value, n)                  │
│  ├─ CSV: model_comparison_statistics_db{X}.csv             │
│  ├─ TXT: model_comparison_summary_db{X}.txt                │
│  │   └─ Interpreted results (excellent/good/moderate)      │
│  └─ CSV: model_comparison_raw_data_db{X}.csv               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                PHASE 7: INTERACTIVE MODE                    │
├─────────────────────────────────────────────────────────────┤
│  plt.ion() + infinite loop                                  │
│  └─ Keeps all plots open until Ctrl+C                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Function Reference

### `load_matlab_data()`
**Purpose**: Load MATLAB .mat files containing PSTH data from both models.

**Returns**: 
- `bez_data` (dict): BEZ model data structure
- `cochlea_data` (dict): Cochlea model data structure

**Key Features**:
- Uses `scipy.io.loadmat()` with `struct_as_record=False, squeeze_me=True`
- Prints loaded keys for debugging
- Raises `FileNotFoundError` if files missing

**Error Handling**: Try-except with detailed error messages

---

### `extract_parameters(data, data_type="bez")`
**Purpose**: Extract experimental parameters from loaded MATLAB data.

**Parameters**:
- `data` (dict): Loaded MATLAB data
- `data_type` (str): "bez" or "cochlea" (for diagnostic messages)

**Returns**: Tuple of (cfs, frequencies, dbs) as sorted numpy arrays

**Nested Function**: `convert_to_numeric(arr)`
- Handles MATLAB string arrays (Unicode/byte strings)
- Converts to float arrays
- Flattens multi-dimensional arrays

**Diagnostic Output**:
```
Extracting parameters for bez model...
  CFs: 8 values, range: 125.0 - 8000.0 Hz
  Frequencies: 10 values, range: 100.0 - 1000.0 Hz
  dB levels: [50. 60. 70. 80.]
```

---

### `extract_psth_data(data, data_type="bez")`
**Purpose**: Extract PSTH firing rate arrays for all fiber types.

**Parameters**:
- `data` (dict): Loaded MATLAB data
- `data_type` (str): "bez" or "cochlea"

**Returns**: Dictionary with keys ['lsr', 'msr', 'hsr'], values are 4D numpy arrays

**Handles Multiple Structures**:
```python
# BEZ Model - tries both formats:
bez_rates.lsr  # or
bez_rates['lsr']

# Cochlea Model - tries:
data['lsr_all']  # or
data['psth_struct'].lsr
```

**Diagnostic Output**: Prints shape of each fiber type array

---

### `create_comparison_dataframe(bez_data, cochlea_data)`
**Purpose**: Core data processing - creates point-by-point comparison DataFrame.

**Algorithm**:
```python
for cf in CFs:
    for freq in Frequencies:
        for db in dB_levels:
            for fiber_type in ['lsr', 'msr', 'hsr']:
                # Extract PSTH time series
                bez_psth = bez_rates[fiber_type][cf_idx, freq_idx, db_idx, :]
                cochlea_psth = cochlea_rates[fiber_type][cf_idx, freq_idx, db_idx, :]
                
                # Calculate mean rates
                bez_mean = np.mean(bez_psth)
                cochlea_mean = np.mean(cochlea_psth)
                
                # Store comparison point
                append({cf, freq, db, fiber_type, 
                        bez_rate, cochlea_rate, freq_diff})
```

**Returns**: Pandas DataFrame
- Rows = individual conditions
- Columns = cf, frequency, db_level, fiber_type, bez_rate, cochlea_rate, freq_diff, cf_group

**Example DataFrame**:
| cf | frequency | db_level | fiber_type | bez_rate | cochlea_rate | freq_diff | cf_group |
|----|-----------|----------|------------|----------|--------------|-----------|----------|
| 125 | 100 | 60.0 | lsr | 45.2 | 43.8 | -25.0 | CF_125Hz |
| 125 | 125 | 60.0 | lsr | 89.7 | 91.2 | 0.0 | CF_125Hz |
| 125 | 150 | 60.0 | lsr | 52.1 | 50.3 | 25.0 | CF_125Hz |

---

### `create_comparison_plots(df, db_level, output_dir)`
**Purpose**: Generate publication-quality scatter plots with statistical analysis.

**Visual Encoding Strategy**:

1. **X-axis**: BEZ model firing rate (spikes/s)
2. **Y-axis**: Cochlea model firing rate (spikes/s)
3. **Color (Hue)**: Characteristic Frequency
   - Palette: "Spectral" (perceptually uniform)
   - Each CF gets unique color
4. **Brightness (Alpha)**: Distance from CF
   - Formula: `α = 0.1 + 0.9 * (1 - normalized_distance)²`
   - At CF (freq = cf): α = 1.0 (bright)
   - Far from CF: α = 0.1 (dim)
   - Exponential scaling for dramatic contrast

**Statistical Analysis**:
```python
# 1. Data cleaning
clean_data = remove_nan_and_inf(data)

# 2. Linear regression (two methods for validation)
slope, intercept, r_value, p_value, std_err = linregress(bez, cochlea)
poly_slope, poly_intercept = np.polyfit(bez, cochlea, 1)

# 3. R² calculation (MATLAB-style)
correlation_matrix = np.corrcoef(bez, cochlea)
r_squared = correlation_matrix[0, 1]**2
```

**Plot Elements**:
- Scatter points (size=80, black edges)
- Linear regression line (black, thick)
- Perfect match diagonal (y=x, dashed)
- Statistics text box (R², slope, intercept, n)
- Custom legend:
  - All CF colors
  - Alpha interpretation (bright/dim)

**Diagnostic Output**:
```
Creating comparison plot for LSR fibers (240 points)...
  Frequency difference range: -100.0 to 200.0 Hz
  Alpha range: 0.10 to 1.00
  Data quality check:
    BEZ rates - Min: 5.234, Max: 234.567
    Cochlea rates - Min: 4.892, Max: 228.901
    NaN values - BEZ: 0, Cochlea: 0
    Infinite values - BEZ: 0, Cochlea: 0
  Linear regression comparison:
    scipy.linregress - Slope: 0.9847, R²: 0.9234
    numpy.polyfit   - Slope: 0.9847, R²: 0.9234
    Difference      - Slope: 0.000000, R²: 0.000000
  Final stats - Slope: 0.985, R²: 0.923, n: 240
  Plot saved to //subcorticalSTRF/model_comparisons_python/compare_lsr_cf_brightness_seaborn_db60.png
```

---

### `main()`
**Purpose**: Orchestrate entire analysis workflow and generate summaries.

**Execution Flow**:
```python
1. Load data files
2. Extract parameters
3. Extract PSTH data
4. Select dB level (db_idx = 1, typically 60 dB)
5. Create comparison DataFrame
6. Generate plots for all fiber types
7. Calculate summary statistics
8. Save outputs:
   - Statistics CSV
   - Summary text file
   - Raw data CSV
9. Enter interactive mode (keep plots open)
```

**Console Output Example**:
```
=== BEZ vs Cochlea Model Comparison ===

Loading MATLAB data files...
Loading BEZ data from: //subcorticalSTRF/BEZ2018_meanrate/results/processed_data/bez_acrossruns_psths.mat
✓ BEZ data loaded successfully with scipy.io.loadmat
Loading cochlea data from: //subcorticalSTRF/cochlea_meanrate/out/condition_psths/cochlea_psths.mat
✓ Cochlea data loaded successfully with scipy.io.loadmat

=== Creating Comparison at 60.0 dB ===

Summary of slopes and correlations:
Fiber Type | Slope | R²   | p-value | n_points
-----------+-------+------+---------+---------
       LSR | 0.985 | 0.923 | 1.23e-45|      240
       MSR | 1.023 | 0.881 | 3.45e-38|      240
       HSR | 0.967 | 0.942 | 5.67e-52|      240

Statistics saved to: //subcorticalSTRF/model_comparisons_python/model_comparison_statistics_db60.csv
Detailed summary saved to: //subcorticalSTRF/model_comparisons_python/model_comparison_summary_db60.txt
Raw data saved to: //subcorticalSTRF/model_comparisons_python/model_comparison_raw_data_db60.csv

Script is running to keep plots open. Press Ctrl+C to exit.
```

---

## Data Structures

### Input Data Dimensions
```
Typical experiment:
├─ N_cf = 8           (e.g., 125, 250, 500, 1000, 2000, 4000, 8000 Hz)
├─ N_freq = 10        (e.g., 100-1000 Hz in steps)
├─ N_db = 4           (e.g., 50, 60, 70, 80 dB SPL)
├─ N_time = 200       (PSTH bins, e.g., 1ms bins for 200ms stimulus)
└─ N_fibers = 3       (LSR, MSR, HSR)

Total conditions per model: 8 × 10 × 4 × 3 = 960
Total comparison points: 960
```

### DataFrame Structure
After `create_comparison_dataframe()`:
```python
df.shape = (960, 8)  # 960 rows, 8 columns

Columns:
├─ cf              float64  # Characteristic frequency (Hz)
├─ frequency       float64  # Stimulus frequency (Hz)
├─ db_level        float64  # Sound level (dB SPL)
├─ fiber_type      object   # 'lsr', 'msr', or 'hsr'
├─ bez_rate        float64  # Mean BEZ PSTH (spikes/s)
├─ cochlea_rate    float64  # Mean Cochlea PSTH (spikes/s)
├─ freq_diff       float64  # frequency - cf (Hz)
└─ cf_group        object   # 'CF_125Hz', 'CF_250Hz', etc.
```

---

## Visualization Features

### Color Palette Selection
```python
# Perceptually uniform palette
cf_palette = sns.mpl_palette("Spectral", n_colors=n_cfs)
```

**Why "Spectral"?**
- Perceptually uniform: Equal visual differences = equal data differences
- High contrast: Distinguishable even with 8+ CFs
- Colorblind-friendly options available

### Alpha Calculation Detail
```python
# Calculate distance from CF
abs_freq_diff = abs(frequency - cf)
max_abs_diff = max(abs_freq_diff)

# Normalize to [0, 1]
normalized_diff = abs_freq_diff / max_abs_diff

# Exponential scaling for dramatic contrast
alpha = 0.1 + 0.9 * (1 - normalized_diff)²

# Examples:
# At CF (diff=0):     α = 0.1 + 0.9*(1-0)² = 1.00 (brightest)
# Halfway (diff=0.5): α = 0.1 + 0.9*(1-0.5)² = 0.325
# Far (diff=1.0):     α = 0.1 + 0.9*(1-1)² = 0.10 (dimmest)
```

**Why Exponential?**
- Linear scaling: insufficient visual contrast
- Exponential: dramatic difference between near-CF and off-CF responses
- Mimics actual tuning curve shape

### Legend Design
```python
Legend entries:
├─ All CFs (colored circles)
│  ├─ CF 125Hz  ●
│  ├─ CF 250Hz  ●
│  ├─ CF 500Hz  ●
│  └─ ...
├─ Alpha interpretation
│  ├─ Far from CF (dim)   ◐
│  └─ At CF (bright)      ●
└─ Regression elements
   ├─ Linear fit ─────
   └─ Perfect match ┄┄┄
```

---

## Output Files

### Per Fiber Type (3 sets):

1. **High-Resolution Raster Images**
   ```
   compare_lsr_cf_brightness_seaborn_db60.png
   compare_msr_cf_brightness_seaborn_db60.png
   compare_hsr_cf_brightness_seaborn_db60.png
   ```
   - Format: PNG
   - Resolution: 300 DPI
   - Size: ~2-3 MB each
   - Use: Presentations, reports

2. **Vector Graphics**
   ```
   compare_lsr_cf_brightness_seaborn_db60.pdf
   compare_msr_cf_brightness_seaborn_db60.pdf
   compare_hsr_cf_brightness_seaborn_db60.pdf
   ```
   - Format: PDF
   - Resolution: Scalable
   - Size: ~500 KB each
   - Use: Publications, high-quality prints

### Summary Files:

3. **Statistics CSV**
   ```
   model_comparison_statistics_db60.csv
   ```
   - Machine-readable statistics
   - Import into R, MATLAB, Excel
   - Columns: fiber_type, slope, intercept, r_squared, correlation, p_value, std_error, n_points, db_level, min/max/mean/std rates

4. **Human-Readable Summary**
   ```
   model_comparison_summary_db60.txt
   ```
   - Formatted text report
   - Includes interpretations:
     - "Excellent correlation (R² = 0.94)"
     - "Near-perfect model agreement (slope = 0.985)"
     - "BEZ model shows higher rates"

5. **Raw Comparison Data**
   ```
   model_comparison_raw_data_db60.csv
   ```
   - Complete DataFrame (960 rows)
   - All conditions × fiber types
   - Use: Custom analyses, model refinement

---

## Interpretation Guide

### Understanding the Scatter Plots

**Perfect Agreement Scenario**:
```
All points lie on diagonal (y = x)
└─ Slope = 1.0, R² = 1.0
   → Models are identical
```

**Common Patterns**:

1. **Slope ≈ 1.0, High R²** (e.g., slope=0.98, R²=0.94)
   - Models agree well
   - Proportional relationship maintained
   - Slight systematic difference (scaling)

2. **Slope > 1.0** (e.g., slope=1.15)
   - Cochlea model produces higher rates
   - BEZ model underestimates

3. **Slope < 1.0** (e.g., slope=0.85)
   - BEZ model produces higher rates
   - Cochlea model underestimates

4. **High R², Slope ≠ 1.0**
   - Strong correlation (models capture same trends)
   - Systematic scaling difference
   - May need calibration

5. **Low R²** (e.g., R²=0.5)
   - Weak correlation
   - Models capture different aspects
   - Check for:
     - Different CF tuning
     - Different temporal dynamics
     - Implementation errors

### R² Thresholds
```
R² > 0.9  →  Excellent agreement
R² > 0.7  →  Good agreement
R² > 0.5  →  Moderate agreement
R² < 0.5  →  Poor agreement (investigate!)
```

### Fiber Type Differences

**Typical Observations**:
- **HSR**: Often best agreement (highest R²)
  - Simple saturating response
  - Less adaptation complexity
  
- **LSR**: Variable agreement
  - Complex adaptation dynamics
  - Harder to model

- **MSR**: Intermediate behavior

### Color/Brightness Patterns

**What to Look For**:

1. **Bright points (at CF)**: Should show highest rates
2. **Dim points (off-CF)**: Lower rates
3. **Color clustering**: Same CF should cluster
4. **Spread per color**: Indicates tuning curve width

**Example Interpretation**:
```
Blue points (CF=500Hz) clustered tightly on diagonal
└─ Good agreement for this CF

Red points (CF=4000Hz) scattered widely
└─ Poor agreement at high CFs
   → Check model implementations
```

---

## Troubleshooting

### Common Errors

#### 1. FileNotFoundError
```
Error: BEZ model PSTH data not found.
```

**Cause**: Input .mat files missing

**Solution**:
```bash
# Check files exist:
ls //subcorticalSTRF/BEZ2018_meanrate/results/processed_data/bez_acrossruns_psths.mat
ls //subcorticalSTRF/cochlea_meanrate/out/condition_psths/cochlea_psths.mat

# Run preprocessing scripts:
matlab -r "bez_compute_mean_across_runs"
matlab -r "aggregate_cochlea_psths"
```

#### 2. KeyError: 'Could not find PSTH data'
```
KeyError: Could not find PSTH data in cochlea file
```

**Cause**: MATLAB structure format mismatch

**Solutions**:
1. Check MATLAB file contents:
   ```matlab
   load('cochlea_psths.mat')
   whos  % List variables
   ```

2. Modify script to match your structure:
   ```python
   # Add your structure name:
   elif 'your_psth_name' in data:
       rates['lsr'] = data['your_psth_name'].lsr
   ```

#### 3. Shape Mismatch
```
IndexError: index 5 is out of bounds for axis 0 with size 4
```

**Cause**: Different number of CFs/frequencies/dB levels between models

**Debug**:
```python
# Add after loading:
print("BEZ CFs:", len(bez_cfs))
print("Cochlea CFs:", len(cochlea_cfs))
print("BEZ shape:", bez_rates['lsr'].shape)
print("Cochlea shape:", cochlea_rates['lsr'].shape)
```

**Solution**: Ensure both models use same experimental parameters

#### 4. Empty DataFrame
```
Created dataframe with 0 comparison points
```

**Cause**: All try-except blocks failing (dimension mismatch)

**Debug**:
```python
# Replace continue with print:
except (IndexError, ValueError) as e:
    print(f"Error at CF={cf_val}, freq={freq_val}: {e}")
    continue
```

#### 5. Plots Don't Appear
```
Script exits immediately, no plots shown
```

**Cause**: Interactive mode issues

**Solutions**:
1. Run with specific backend:
   ```python
   import matplotlib
   matplotlib.use('TkAgg')  # or 'Qt5Agg'
   ```

2. Force display:
   ```python
   plt.show(block=True)  # Blocks until closed
   ```

#### 6. NaN in Regression
```
Linear regression comparison:
  scipy.linregress - Slope: nan, R²: nan
```

**Cause**: All data points NaN or constant values

**Debug**:
```python
print("Unique BEZ values:", len(np.unique(clean_data['bez_rate'])))
print("Unique Cochlea values:", len(np.unique(clean_data['cochlea_rate'])))
```

**Solution**: Check PSTH calculations in preprocessing scripts

---

## Configuration Options

### Current Settings
```python
# In main():
db_idx = 1  # Second dB level (typically 60 dB)

# In create_comparison_plots():
point_size = 80
edge_width = 0.5
edge_color = 'black'
dpi = 300
palette = "Spectral"

# Alpha scaling:
alpha_min = 0.1
alpha_max = 1.0
alpha_exponent = 2  # (1 - norm_diff)^2
```

### Customization Examples

**Change dB Level**:
```python
# In main():
db_idx = 0  # Use first dB level
db_idx = 2  # Use third dB level
```

**Different Color Palette**:
```python
# In create_comparison_plots():
cf_palette = sns.color_palette("husl", n_colors=n_cfs)  # Circular
cf_palette = sns.color_palette("viridis", n_colors=n_cfs)  # Sequential
cf_palette = sns.color_palette("Set1", n_colors=n_cfs)  # Qualitative
```

**Adjust Alpha Contrast**:
```python
# Linear (less dramatic):
alpha = 0.3 + 0.7 * (1 - normalized_diff)

# More dramatic:
alpha = 0.05 + 0.95 * (1 - normalized_diff)^3
```

**Analyze All dB Levels**:
```python
# In main(), replace single comparison with loop:
for db_idx in range(len(cochlea_dbs)):
    db_level = cochlea_dbs[db_idx]
    df = create_comparison_dataframe(bez_data, cochlea_data)
    create_comparison_plots(df, db_level, output_dir)
```

---

## Performance Notes

**Expected Runtime**:
- Small dataset (8 CFs × 10 freqs × 4 dBs): ~30 seconds
- Large dataset (16 CFs × 20 freqs × 6 dBs): ~2-3 minutes

**Memory Usage**:
- Typical: ~500 MB
- Large datasets: ~2 GB

**Bottlenecks**:
1. MATLAB file loading (~5 seconds)
2. DataFrame creation (~10 seconds)
3. Plot generation (~5 seconds per fiber type)

**Optimization Tips**:
```python
# Process single fiber type:
for ft in ['lsr']:  # Instead of ['lsr', 'msr', 'hsr']

# Reduce data quality checks:
# Comment out diagnostic prints in loops

# Use lower DPI for testing:
plt.savefig(output_file, dpi=150)  # Instead of 300
```

---

## Related Scripts

**Preprocessing (MATLAB)**:
- `bez_compute_mean_across_runs.m` - Aggregate BEZ PSTHs
- `aggregate_cochlea_psths.m` - Aggregate Cochlea PSTHs

**Alternative Comparisons (Python)**:
- `compare_models_simple.py` - Simplified version
- `compare_temporal_scatter.py` - Time bin analysis
- `compare_cf_responses.py` - CF-specific plots
- `compare_individual_temporal_correlations.py` - Detailed CF regression

**Analysis Scripts**:
- `all_cfs_regression_analysis.py` - Pooled regression
- `fiber_specific_regression.py` - Fiber-specific analysis

---

## Citations & References

**BEZ2018 Model**:
Bruce, I. C., Erfani, Y., & Zilany, M. S. (2018). A phenomenological model of the synapse between the inner hair cell and auditory nerve: Implications of limited neurotransmitter release sites. Hearing Research, 360, 40-54.

**Cochlea Package**:
- Rudnicki, M., & Hemmert, W. (2017). High entrainment constrains synaptic depression levels of an in vivo globular bushy cell model. Frontiers in Computational Neuroscience, 11, 16.
- Documentation: https://github.com/mrkrd/cochlea

**Statistical Methods**:
- Linear regression: `scipy.stats.linregress`
- Polynomial fitting: `numpy.polyfit`
- Pearson correlation: `numpy.corrcoef`

---

## Version History

**v3 (Current - October 28, 2024)**:
- Uses scipy.io.loadmat for regular MATLAB format
- Enhanced visualization with CF-color and frequency-brightness encoding
- Comprehensive statistical summaries
- MATLAB regression validation
- Interactive plot mode

**v2**:
- Improved h5py handling for MATLAB v7.3
- Recursive structure loading

**v1**:
- Basic comparison with dual loading strategy
- Simple scatter plots

---

## Contact & Support

**Author**: [Your Name]  
**Lab**: [Your Lab]  
**Email**: [Your Email]

**Issues**: Report to GitHub repository or lab Slack channel

**Related Documentation**:
- Main README: `subcorticalSTRF/README.md`
- Function index: `subcorticalSTRF/function_explanations_index.md`

---

## Quick Start Example

```bash
# 1. Ensure preprocessing complete
cd //subcorticalSTRF
matlab -r "bez_compute_mean_across_runs; aggregate_cochlea_psths; exit"

# 2. Run comparison
python3 compare_models_v3.py

# 3. View outputs
cd model_comparisons_python
ls -lh compare_*.png
cat model_comparison_summary_db60.txt

# 4. Open plots
eog compare_lsr_cf_brightness_seaborn_db60.png &
eog compare_msr_cf_brightness_seaborn_db60.png &
eog compare_hsr_cf_brightness_seaborn_db60.png &
```

Expected output in terminal:
```
=== BEZ vs Cochlea Model Comparison ===
Loading MATLAB data files...
✓ BEZ data loaded successfully
✓ Cochlea data loaded successfully
...
Summary of slopes and correlations:
Fiber Type | Slope | R²   | p-value | n_points
-----------+-------+------+---------+---------
       LSR | 0.985 | 0.923 | 1.23e-45|      240
       MSR | 1.023 | 0.881 | 3.45e-38|      240
       HSR | 0.967 | 0.942 | 5.67e-52|      240

Script is running to keep plots open. Press Ctrl+C to exit.
```

---

**Last Updated**: November 3, 2025  
**Script Version**: 3.0  
**Documentation Version**: 1.0
