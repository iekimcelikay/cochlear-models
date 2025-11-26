# Visualization Functions - Quick Guide

## 📊 Two Simple Plotting Functions

Added to `data_loader.py`:

### **1. `plot_regression_scatter()`** - Scatter plot with regression line

Shows the relationship between Cochlea and BEZ firing rates.

```python
from subcorticalSTRF.data_loader import plot_regression_scatter

fig, ax = plot_regression_scatter(
    cochlea_means,
    bez_means,
    run_idx=0,              # BEZ run to compare
    title=None,             # Auto-generated if None
    figsize=(8, 8),         # Figure size
    save_path=None          # Save path (None = display only)
)
```

**Features:**
- ✅ Scatter plot of all matched points
- ✅ Regression line (red)
- ✅ Identity line y=x (black dashed)
- ✅ Statistics in title: r, p-value, n
- ✅ Equal aspect ratio
- ✅ Auto-save option

---

### **2. `plot_slope_distribution()`** - Histogram of slopes

Shows distribution of regression slopes across all BEZ runs.

```python
from subcorticalSTRF.data_loader import plot_slope_distribution

fig, ax, slopes = plot_slope_distribution(
    cochlea_means,
    bez_means,
    bins=20,                # Number of histogram bins
    title=None,             # Auto-generated if None
    figsize=(10, 6),        # Figure size
    save_path=None          # Save path (None = display only)
)

# Access slopes array
print(f"Mean slope: {np.mean(slopes):.3f}")
print(f"Std slope: {np.std(slopes):.3f}")
```

**Features:**
- ✅ Histogram of slopes across runs
- ✅ Mean line (red dashed)
- ✅ ±1 SD shading (red)
- ✅ Reference line at slope=1 (green)
- ✅ Statistics printed to console
- ✅ Returns slopes array for further analysis

---

## 🚀 Quick Start

### **Minimal Example:**

```python
from subcorticalSTRF.data_loader import (
    load_bez_psth_data,
    load_matlab_data,
    mean_across_time_psth_cochlea,
    mean_across_time_psth_bez,
    plot_regression_scatter,
    plot_slope_distribution
)
import matplotlib.pyplot as plt

# 1. Load data
bez_data, params = load_bez_psth_data()
_, cochlea_data = load_matlab_data()

# 2. Compute means
cochlea_means = mean_across_time_psth_cochlea(cochlea_data, params)
bez_means = mean_across_time_psth_bez(bez_data, params)

# 3. Plot
plot_regression_scatter(cochlea_means, bez_means, run_idx=0, 
                         save_path='scatter.png')
plot_slope_distribution(cochlea_means, bez_means, 
                        save_path='slopes.png')

plt.show()
```

---

## 🎨 Customization Examples

### **Custom Title:**

```python
plot_regression_scatter(
    cochlea_means, bez_means,
    run_idx=0,
    title='My Custom Title\nCochlea vs BEZ Model',
    save_path='custom_scatter.png'
)
```

### **Adjust Figure Size:**

```python
plot_regression_scatter(
    cochlea_means, bez_means,
    figsize=(12, 12)  # Larger figure
)
```

### **More Histogram Bins:**

```python
fig, ax, slopes = plot_slope_distribution(
    cochlea_means, bez_means,
    bins=30  # More detailed histogram
)
```

### **Display Without Saving:**

```python
import matplotlib.pyplot as plt

plot_regression_scatter(cochlea_means, bez_means)  # No save_path
plot_slope_distribution(cochlea_means, bez_means)  # No save_path

plt.show()  # Display both
```

---

## 🔧 Extending the Functions

The functions are designed to be simple but extensible. Here's how to add features:

### **Example 1: Color by Fiber Type**

```python
def plot_scatter_colored_by_fiber(cochlea_means, bez_means, run_idx=0):
    """Extended: Color points by fiber type"""
    import matplotlib.pyplot as plt
    from subcorticalSTRF.data_loader import match_data_points
    
    fiber_types = ['hsr', 'msr', 'lsr']
    colors = {'hsr': 'red', 'msr': 'green', 'lsr': 'blue'}
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for fiber_type in fiber_types:
        # Get data for this fiber type
        matched_rates, _ = match_data_points(
            cochlea_means, bez_means,
            fiber_type=fiber_type,
            run2_idx=run_idx
        )
        
        x = [pt['data1_rate'] for pt in matched_rates]
        y = [pt['data2_rate'] for pt in matched_rates]
        
        ax.scatter(x, y, alpha=0.6, s=40, 
                   color=colors[fiber_type], 
                   label=fiber_type.upper())
    
    # Add identity line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    
    ax.set_xlabel('Cochlea (Hz)')
    ax.set_ylabel('BEZ (Hz)')
    ax.legend()
    ax.set_aspect('equal')
    
    return fig, ax
```

### **Example 2: Add Confidence Interval**

```python
def plot_scatter_with_ci(cochlea_means, bez_means, run_idx=0):
    """Extended: Add confidence interval shading"""
    import matplotlib.pyplot as plt
    from subcorticalSTRF.data_loader import match_data_points
    from scipy import stats
    
    matched_rates, _ = match_data_points(
        cochlea_means, bez_means, run2_idx=run_idx
    )
    
    x = np.array([pt['data1_rate'] for pt in matched_rates])
    y = np.array([pt['data2_rate'] for pt in matched_rates])
    
    # Regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter
    ax.scatter(x, y, alpha=0.5, s=30)
    
    # Regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2)
    
    # Confidence interval (approximate)
    # Calculate prediction interval
    predict_error = np.sqrt(np.sum((y - (slope * x + intercept))**2) / (len(x) - 2))
    margin = 1.96 * predict_error  # 95% CI
    
    ax.fill_between(x_line, y_line - margin, y_line + margin,
                     alpha=0.2, color='red', label='95% CI')
    
    ax.legend()
    ax.set_aspect('equal')
    
    return fig, ax
```

### **Example 3: Subplots for Multiple Runs**

```python
def plot_multiple_runs(cochlea_means, bez_means, run_indices=[0, 1, 2, 3]):
    """Extended: Show multiple runs in subplots"""
    import matplotlib.pyplot as plt
    from subcorticalSTRF.data_loader import plot_regression_scatter
    
    n_runs = len(run_indices)
    cols = 2
    rows = (n_runs + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6*rows))
    axes = axes.flatten()
    
    for idx, run_idx in enumerate(run_indices):
        # Create scatter for this run (without saving)
        fig_temp, ax_temp = plot_regression_scatter(
            cochlea_means, bez_means, 
            run_idx=run_idx,
            figsize=(6, 6)
        )
        
        # Copy to subplot (simplified - you'd need to extract data)
        # This is a placeholder - actual implementation would 
        # re-create the plot in the subplot
        
    plt.tight_layout()
    return fig, axes
```

---

## 📊 Output Interpretation

### **Scatter Plot:**

- **Points**: Each point = one experimental condition (CF × freq × dB × fiber)
- **Red line**: Linear regression fit
- **Black dashed**: Identity line (y=x, perfect agreement)
- **Distance from identity**: How much models differ
- **Slope < 1**: BEZ fires lower than Cochlea
- **Slope > 1**: BEZ fires higher than Cochlea
- **Slope = 1**: Perfect scaling (parallel responses)

### **Slope Distribution:**

- **Histogram**: Frequency of slope values across runs
- **Red dashed line**: Mean slope across all runs
- **Red shading**: ±1 standard deviation
- **Green dotted line**: Perfect scaling reference (slope=1)
- **Narrow distribution**: Consistent across runs
- **Wide distribution**: Variable run-to-run behavior

---

## 📁 Generated Files

Both functions save high-resolution PNG files (300 DPI) when `save_path` is provided:

```python
# Saves to current directory
plot_regression_scatter(cochlea, bez, save_path='scatter.png')

# Saves to specific directory
plot_regression_scatter(cochlea, bez, 
                         save_path='/path/to/figures/scatter.png')

# No save (display only)
plot_regression_scatter(cochlea, bez)
plt.show()
```

---

## 🎯 Advanced Examples

See `example_advanced_visualization.py` for:
- Regression by fiber type (3 subplots)
- Slope comparison bar chart
- Slope evolution across runs
- Custom color schemes
- Multiple panels

---

## 🔬 Statistical Information Provided

### **Scatter Plot Prints:**
```
✓ Figure saved to: scatter.png
```

### **Slope Distribution Prints:**
```
Computing regressions for all runs...
  Run 0... ✓ r = 0.943, n = 432
  Run 1... ✓ r = 0.938, n = 432
  ...

Slope Statistics:
  Mean: 0.9521
  Std:  0.0234
  Min:  0.9102 (Run 3)
  Max:  0.9876 (Run 7)
  Median: 0.9543

✓ Figure saved to: slopes.png
```

---

## 💡 Tips

1. **Use `verbose=False`** in regression functions if you don't need progress output
2. **Adjust `rtol/atol`** if you get too few matched points
3. **Set `figsize`** based on your screen/publication needs
4. **Use `save_path`** with descriptive names for batch processing
5. **Call `plt.show()`** at the end to display all figures
6. **Close figures** with `plt.close('all')` to free memory

---

## 📚 See Also

- `REGRESSION_USAGE_EXAMPLES.md` - Regression function usage
- `example_visualization.py` - Basic usage script
- `example_advanced_visualization.py` - Advanced customization examples
