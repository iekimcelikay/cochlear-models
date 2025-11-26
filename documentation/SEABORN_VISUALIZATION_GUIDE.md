# Advanced Seaborn Visualization Guide

## Overview

The `plot_regression_seaborn_cf_brightness()` function creates publication-quality scatter plots that encode **two dimensions of information** in the visual appearance of data points:

1. **Color (Hue)**: Characteristic Frequency (CF)
2. **Brightness (Alpha)**: Distance from CF (|frequency - CF|)

## Function Signature

```python
from subcorticalSTRF.data_loader import plot_regression_seaborn_cf_brightness

fig, axes, stats = plot_regression_seaborn_cf_brightness(
    cochlea_means,      # Cochlea model mean rates
    bez_means,          # BEZ model mean rates
    run_idx=0,          # Which BEZ run to compare
    db_level=70.0,      # dB level (REQUIRED)
    rtol=1e-5,          # Fuzzy matching tolerance
    atol=1e-8,          # Absolute tolerance
    figsize=(14, 10),   # Figure size
    save_path=None      # Path to save (PNG + PDF)
)
```

## Visual Encoding

### Color (Hue) - Characteristic Frequency
- Uses **Spectral** colormap from seaborn
- **Perceptually uniform**: Each CF gets a distinct, easily distinguishable color
- **Consistent across panels**: Same CF = same color in HSR, MSR, and LSR plots
- **Legend**: Shows sampled CFs (every ~6th CF to avoid crowding)

### Brightness (Alpha) - Distance from CF
- **Formula**: `alpha = 0.1 + 0.9 * (1 - |freq - CF| / max_diff)²`
- **Exponential scaling**: Creates pronounced visual differences
- **Bright (α = 1.0)**: Stimulus frequency equals CF (f = CF)
  - These are the "best frequency" responses
  - Most physiologically relevant
- **Medium (α ~ 0.5)**: Stimulus moderately far from CF
  - Off-frequency responses
- **Dim (α = 0.1)**: Stimulus far from CF
  - Tail responses
  - Less physiologically significant

## What the Plot Shows

### Three Subplots
1. **Left**: HSR (High Spontaneous Rate) fibers
2. **Middle**: MSR (Medium Spontaneous Rate) fibers  
3. **Right**: LSR (Low Spontaneous Rate) fibers

### On Each Subplot
- **X-axis**: BEZ model mean firing rate (Hz)
- **Y-axis**: Cochlea model mean firing rate (Hz)
- **Black solid line**: Linear regression fit
- **Black dashed line**: Identity line (y = x, perfect agreement)
- **Colored points**: Data from each CF-frequency combination
- **Text box**: Regression statistics (R², slope, n)

### Legend (Right Side)
- **Top section**: Sample of CFs with their colors
- **Bottom section**: 
  - "Far from CF (dim)" - shows minimum alpha
  - "At CF (bright)" - shows maximum alpha

## Scientific Interpretation

### Pattern Analysis

**1. Bright Points Near Regression Line**
- Good model agreement at best frequencies
- Models capture tuning curve peaks similarly

**2. Scatter in Bright vs. Dim Points**
- **Tight bright cluster**: Models agree well at CF
- **Scattered dim points**: Models differ in off-CF responses
- Indicates different frequency selectivity

**3. Color Clustering**
- **Vertical clustering**: BEZ model varies more at certain CFs
- **Horizontal clustering**: Cochlea model varies more at certain CFs
- **Diagonal clustering**: Both models scale similarly

**4. Slope Differences Across Fiber Types**
- **Slope > 1**: BEZ rates systematically higher than Cochlea
- **Slope < 1**: Cochlea rates systematically higher than BEZ
- **Different slopes by fiber type**: Reveals fiber-specific encoding

### Regression Statistics

**R² (Coefficient of Determination)**
- R² > 0.9: Excellent model agreement
- 0.7 < R² < 0.9: Good agreement  
- R² < 0.7: Models differ substantially

**Slope**
- Slope ≈ 1.0: Linear scaling matches
- Slope > 1.0: BEZ overestimates relative to Cochlea
- Slope < 1.0: BEZ underestimates relative to Cochlea

## Usage Examples

### Example 1: Single dB Level Analysis
```python
# Load and process data
bez_data, params = load_bez_psth_data()
cochlea_data = load_matlab_data()  # Your cochlea data

cochlea_means = mean_across_time_psth_cochlea(cochlea_data, params)
bez_means = mean_across_time_psth_bez(bez_data, params)

# Create seaborn plot at 70 dB
fig, axes, stats = plot_regression_seaborn_cf_brightness(
    cochlea_means, bez_means, 
    run_idx=0, 
    db_level=70.0,
    save_path='regression_70dB.png'
)

plt.show()

# Access statistics
print(f"HSR slope: {stats['hsr']['slope']:.3f}")
print(f"MSR R²: {stats['msr']['r_squared']:.3f}")
```

### Example 2: Compare Across dB Levels
```python
all_stats = {}

for db_level in [50, 60, 70, 80]:
    fig, axes, stats = plot_regression_seaborn_cf_brightness(
        cochlea_means, bez_means,
        run_idx=0,
        db_level=db_level,
        save_path=f'regression_{int(db_level)}dB.png'
    )
    all_stats[db_level] = stats
    plt.show()

# Compare slopes across dB levels
for db in sorted(all_stats.keys()):
    print(f"{db} dB: HSR slope = {all_stats[db]['hsr']['slope']:.3f}")
```

### Example 3: Analyze Different Runs
```python
for run_idx in range(10):  # 10 BEZ runs
    fig, axes, stats = plot_regression_seaborn_cf_brightness(
        cochlea_means, bez_means,
        run_idx=run_idx,
        db_level=70.0,
        save_path=f'regression_70dB_run{run_idx}.png'
    )
    plt.close()  # Don't display, just save

print("All runs plotted and saved!")
```

## Output Files

When `save_path` is specified, the function saves:
1. **PNG file**: High-resolution (300 DPI) for presentations/manuscripts
2. **PDF file**: Vector format for publication-quality figures

Example:
```python
save_path='regression_70dB.png'
```
Creates:
- `regression_70dB.png` (300 DPI raster)
- `regression_70dB.pdf` (vector graphics)

## Advantages Over Simple Scatter Plots

### Information Density
- Simple plot: 2 dimensions (x, y)
- Seaborn plot: 4 dimensions (x, y, CF, frequency difference)

### Physiological Relevance
- **Brightness encoding** highlights on-CF responses (most important)
- **Color encoding** shows CF population effects
- Reveals frequency tuning properties visually

### Publication Quality
- Perceptually uniform colors
- Clean seaborn styling
- Professional legends and annotations
- Dual-format output (PNG + PDF)

### Pattern Recognition
- Easy to spot:
  - CF-specific deviations
  - On-CF vs. off-CF agreement differences
  - Fiber-type specific patterns
  - Systematic biases by frequency

## Comparison with compare_models_v3.py

The function in `data_loader.py` is based on the logic from `compare_models_v3.py` but:

✅ **Integrated**: Part of main data_loader module
✅ **Simplified**: Uses existing data structures (no DataFrame conversion)
✅ **Flexible**: Works with filtered data from `extract_data_at_db()`
✅ **Consistent**: Uses same fuzzy matching as other functions
✅ **Efficient**: Direct access to nested dictionaries

## Tips for Best Results

1. **Always specify db_level**: Required for meaningful CF-frequency analysis
2. **Use at physiologically relevant dB**: 60-80 dB SPL typical
3. **Save outputs**: Creates permanent record of analysis
4. **Compare runs**: Check consistency across BEZ runs
5. **Examine bright points**: Most physiologically relevant data
6. **Look for color patterns**: May reveal CF-dependent effects

## Troubleshooting

**Import Error: seaborn not found**
```bash
pip install seaborn
```

**ValueError: db_level must be specified**
```python
# Wrong:
fig, axes, stats = plot_regression_seaborn_cf_brightness(cochlea, bez)

# Correct:
fig, axes, stats = plot_regression_seaborn_cf_brightness(cochlea, bez, db_level=70.0)
```

**No data points plotted**
- Check that data exists at the specified dB level
- Verify fuzzy matching tolerances (rtol, atol)
- Ensure CFs and frequencies match between models

**Legend too crowded**
- Function automatically samples CFs (shows ~6)
- Increase figure size: `figsize=(18, 10)`
- Adjust legend position if needed

## Related Functions

- `extract_data_at_db()`: Filter data to specific dB level
- `plot_regression_scatter()`: Simple scatter plot (no CF encoding)
- `plot_regression_by_fiber_type()`: Standard 3-panel regression
- `match_data_points()`: Get matched data pairs for custom analysis
