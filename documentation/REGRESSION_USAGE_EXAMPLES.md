# Regression Functions - Usage Guide

The hybrid regression implementation in `compare_bez_cochlea_mainscript.py` provides simple, clear functions backed by flexible engines.

## ⚠️ IMPORTANT: BEZ Data Loading

There are **two different functions** for loading BEZ data:

### **1. `load_bez_psth_data()` - Individual Runs** ✅ Recommended for regression
```python
bez_data, params = load_bez_psth_data()
# Loads: psth_data_128fibers.mat
# Structure: [cf, freq, db, run] - HAS individual runs
# Use for: Run-by-run comparisons, all regression functions
```

### **2. `load_matlab_data()` - Aggregated Runs** ⚠️ Limited functionality
```python
bez_data, cochlea_data = load_matlab_data()
# Loads: bez_acrossruns_psths.mat  
# Structure: [cf, freq, db] - NO run dimension (averaged across runs)
# Use for: Simple comparisons, but CANNOT use:
#   - regress_cochlea_vs_bez_all_runs() ✗
#   - regress_all_bez_run_pairs() ✗
#   - regress_bez_run_vs_run() ✗
```

**Bottom line:** Use `load_bez_psth_data()` for full regression functionality!

---

## 📚 Quick Reference

### **What's Available:**

| Function | What It Does | Output |
|----------|-------------|--------|
| `regress_cochlea_vs_bez_single_run()` | Cochlea vs 1 BEZ run | 1 DataFrame |
| `regress_cochlea_vs_bez_all_runs()` | Cochlea vs all BEZ runs | dict of DataFrames |
| `regress_bez_run_vs_run()` | BEZ run i vs run j | 1 DataFrame |
| `regress_all_bez_run_pairs()` | All BEZ run pairs | dict of DataFrames |
| `regress_by_fiber_type()` | Separate per fiber type | dict by fiber |
| `regress_by_cf()` | Separate per CF | dict by CF |
| `regress_by_db()` | Separate per dB level | dict by dB |
| `correlate_by_fiber_type()` | Correlation (no regression) | dict by fiber |

---

## 🚀 Complete Workflow Example

```python
from pathlib import Path
from subcorticalSTRF.compare_bez_cochlea_mainscript import (
    load_bez_psth_data,  # ← IMPORTANT: Use this for individual runs!
    load_matlab_data,     # ← This loads averaged BEZ data (no runs)
    extract_parameters,
    mean_across_time_psth_cochlea,
    mean_across_time_psth_bez,
    regress_cochlea_vs_bez_single_run,
    regress_cochlea_vs_bez_all_runs,
    regress_by_fiber_type,
    regress_by_cf,
    correlate_by_fiber_type
)

# ===================================================================
# STEP 1: Load Data
# ===================================================================

# OPTION A: Load individual BEZ runs (for run-by-run analysis)
bez_data, params = load_bez_psth_data()  # Returns individual runs
# This loads psth_data_128fibers.mat with structure: [cf, freq, db, run]

# OPTION B: Load aggregated BEZ data (no individual runs)
# bez_data, cochlea_data = load_matlab_data()  
# This loads bez_acrossruns_psths.mat (averaged across runs)
# ⚠️ Cannot use with regress_cochlea_vs_bez_all_runs() or run comparisons!

# Load Cochlea data separately (if using Option A)
_, cochlea_data = load_matlab_data()  # Just get cochlea_data

# Extract parameters (if not already done)
# params = extract_parameters(cochlea_data, model_type="cochlea")

# ===================================================================
# STEP 2: Compute Mean PSTH Rates
# ===================================================================

# Compute mean rates across time (gives nested dicts)
cochlea_means = mean_across_time_psth_cochlea(
    cochlea_data, 
    params, 
    fiber_types=['hsr', 'msr', 'lsr']
)

# For BEZ with individual runs (using load_bez_psth_data())
bez_means = mean_across_time_psth_bez(
    bez_data,
    params,
    fiber_types=['hsr', 'msr', 'lsr']
)
# Result structure: bez_means[fiber_type][cf][freq][db][run_idx] = mean_rate

# ⚠️ NOTE: If you loaded aggregated BEZ data (load_matlab_data()), 
# the BEZ data won't have run dimension and you can only compare to 
# the aggregated mean, not individual runs!

print("✓ Data loaded and processed")

# ===================================================================
# STEP 3: Single Comparison - Cochlea vs BEZ Run 0
# ===================================================================

result = regress_cochlea_vs_bez_single_run(
    cochlea_means, 
    bez_means, 
    run_idx=0,
    rtol=1e-5,  # 0.001% tolerance for fuzzy matching
    verbose=True
)

print("\n=== Cochlea vs BEZ Run 0 ===")
print(f"Correlation (r): {result['r_value'].iloc[0]:.3f}")
print(f"Pearson r: {result['r_pearson'].iloc[0]:.3f}")  # Should match r_value
print(f"Slope: {result['slope'].iloc[0]:.3f}")
print(f"Intercept: {result['intercept'].iloc[0]:.3f}")
print(f"P-value: {result['p_value'].iloc[0]:.3e}")
print(f"Std Error: {result['stderr'].iloc[0]:.3f}")
print(f"N points: {int(result['n_points'].iloc[0])}")
print(f"Equation: Cochlea = {result['slope'].iloc[0]:.3f} × BEZ + {result['intercept'].iloc[0]:.3f}")

# ===================================================================
# STEP 4: Batch Processing - All BEZ Runs
# ===================================================================

all_runs = regress_cochlea_vs_bez_all_runs(
    cochlea_means,
    bez_means,
    rtol=1e-5,
    verbose=True  # Shows progress
)

print("\n=== Cochlea vs All BEZ Runs ===")
for run_idx, df in all_runs.items():
    if df is not None:
        r = df['r_value'].iloc[0]
        n = int(df['n_points'].iloc[0])
        print(f"Run {run_idx}: r = {r:.3f}, n = {n}")

# Find best and worst runs
correlations = {run: df['r_value'].iloc[0] for run, df in all_runs.items() if df is not None}
best_run = max(correlations, key=correlations.get)
worst_run = min(correlations, key=correlations.get)
print(f"\nBest agreement: Run {best_run} (r = {correlations[best_run]:.3f})")
print(f"Worst agreement: Run {worst_run} (r = {correlations[worst_run]:.3f})")

# ===================================================================
# STEP 5: Grouped Analysis - By Fiber Type
# ===================================================================

by_fiber = regress_by_fiber_type(
    cochlea_means,
    bez_means,
    run2_idx=0,  # Compare to BEZ Run 0
    rtol=1e-5,
    verbose=True
)

print("\n=== Agreement by Fiber Type (vs BEZ Run 0) ===")
for fiber_type, df in by_fiber.items():
    if df is not None:
        r = df['r_value'].iloc[0]
        slope = df['slope'].iloc[0]
        n = int(df['n_points'].iloc[0])
        print(f"{fiber_type.upper()}: r = {r:.3f}, slope = {slope:.3f}, n = {n}")

# ===================================================================
# STEP 6: Grouped Analysis - By Characteristic Frequency
# ===================================================================

by_cf = regress_by_cf(
    cochlea_means,
    bez_means,
    run2_idx=0,
    rtol=1e-5,
    verbose=True
)

print("\n=== Agreement by CF (vs BEZ Run 0) ===")
for cf_val, df in sorted(by_cf.items()):
    if df is not None:
        r = df['r_value'].iloc[0]
        n = int(df['n_points'].iloc[0])
        print(f"CF = {cf_val:6.0f} Hz: r = {r:.3f}, n = {n}")

# ===================================================================
# STEP 7: Correlation Only (No Regression)
# ===================================================================

corr_by_fiber = correlate_by_fiber_type(
    cochlea_means,
    bez_means,
    run2_idx=0,
    rtol=1e-5,
    verbose=True
)

print("\n=== Correlation by Fiber Type (no regression) ===")
for fiber_type, df in corr_by_fiber.items():
    if df is not None:
        r = df['r_value'].iloc[0]
        p = df['p_value'].iloc[0]
        n = int(df['n_points'].iloc[0])
        print(f"{fiber_type.upper()}: r = {r:.3f}, p = {p:.3e}, n = {n}")
        # Note: No slope/intercept in correlation-only analysis

# ===================================================================
# STEP 8: BEZ Run-to-Run Consistency
# ===================================================================

from subcorticalSTRF.data_loader import regress_all_bez_run_pairs

run_pairs = regress_all_bez_run_pairs(
    bez_means,
    rtol=1e-5,
    verbose=True
)

print("\n=== BEZ Run-to-Run Consistency ===")
# Show first 5 pairs
for (run_i, run_j), df in list(run_pairs.items())[:5]:
    if df is not None:
        r = df['r_value'].iloc[0]
        print(f"Run {run_i} vs Run {run_j}: r = {r:.3f}")

# Calculate mean consistency
all_r = [df['r_value'].iloc[0] for df in run_pairs.values() if df is not None]
print(f"\nMean run-to-run correlation: {np.mean(all_r):.3f} ± {np.std(all_r):.3f}")

print("\n✓ All analyses complete!")
```

---

## 📊 Output Interpretation

### **Regression Results Include:**

- **`r_value`**: Correlation coefficient from linear regression (-1 to 1)
  - `r > 0.9`: Excellent agreement
  - `r = 0.7-0.9`: Good agreement
  - `r < 0.7`: Poor agreement

- **`r_pearson`**: Pearson correlation (should match `r_value`)

- **`slope`**: Regression slope
  - `slope = 1`: Models have same scaling
  - `slope > 1`: Data2 fires higher than Data1
  - `slope < 1`: Data2 fires lower than Data1

- **`intercept`**: Regression intercept
  - `intercept ≈ 0`: No systematic offset
  - `intercept > 0`: Data2 has positive baseline offset
  - `intercept < 0`: Data2 has negative baseline offset

- **`p_value`**: Statistical significance (p < 0.05 = significant)

- **`stderr`**: Standard error of the slope

- **`n_points`**: Number of matched data points

### **Correlation-Only Results Include:**

- **`r_value`**: Pearson correlation coefficient
- **`p_value`**: Statistical significance
- **`n_points`**: Number of matched points
- *(No slope/intercept since no regression performed)*

---

## 🎯 Common Use Cases

### **Use Case 1: "Do the models agree overall?"**
```python
result = regress_cochlea_vs_bez_single_run(cochlea, bez, run_idx=0)
print(f"Overall agreement: r = {result['r_value'].iloc[0]:.3f}")
```

### **Use Case 2: "Which BEZ run agrees best with Cochlea?"**
```python
all_runs = regress_cochlea_vs_bez_all_runs(cochlea, bez)
best_run = max(all_runs.items(), key=lambda x: x[1]['r_value'].iloc[0])
print(f"Best run: {best_run[0]} with r = {best_run[1]['r_value'].iloc[0]:.3f}")
```

### **Use Case 3: "Does agreement differ by fiber type?"**
```python
by_fiber = regress_by_fiber_type(cochlea, bez, run2_idx=0)
for fiber, df in by_fiber.items():
    print(f"{fiber}: r = {df['r_value'].iloc[0]:.3f}")
```

### **Use Case 4: "Are BEZ runs consistent with each other?"**
```python
pairs = regress_all_bez_run_pairs(bez)
mean_r = np.mean([df['r_value'].iloc[0] for df in pairs.values()])
print(f"Mean run-to-run consistency: r = {mean_r:.3f}")
```

### **Use Case 5: "Does agreement vary across frequency?"**
```python
by_cf = regress_by_cf(cochlea, bez, run2_idx=0)
# Plot r_value vs CF to see frequency dependence
```

---

## ⚙️ Advanced Options

### **Fuzzy Matching Tolerance**

If you get few matched points, adjust tolerances:

```python
# More lenient (allows 0.01% difference)
result = regress_cochlea_vs_bez_single_run(
    cochlea, bez, 
    run_idx=0,
    rtol=1e-4,  # Default: 1e-5
    atol=1e-7   # Default: 1e-8
)

# Stricter (requires exact match)
result = regress_cochlea_vs_bez_single_run(
    cochlea, bez,
    run_idx=0, 
    rtol=1e-8,  # Very strict
    atol=1e-12
)
```

### **Verbose Output**

```python
# See detailed progress
result = regress_cochlea_vs_bez_all_runs(
    cochlea, bez,
    verbose=True  # Shows: "Processing 10 runs... Run 0... ✓ r = 0.943"
)

# Silent mode
result = regress_cochlea_vs_bez_all_runs(
    cochlea, bez,
    verbose=False  # No output
)
```

---

## 🔧 Troubleshooting

### **Problem: "No points matched!"**

**Cause:** CF/frequency/dB values don't match due to floating-point precision

**Solution:** Increase tolerance
```python
result = regress_cochlea_vs_bez_single_run(cochlea, bez, rtol=1e-4)
```

### **Problem: "n_points is very low"**

**Cause:** Filtering too restrictive or data mismatch

**Solution:** Check with verbose mode
```python
from subcorticalSTRF.data_loader import match_data_points
matched, meta = match_data_points(cochlea, bez, verbose=True)
# Shows: "Data1: 432 points, Data2: 432 points, Matched: 12 points"
```

### **Problem: "r_value is NaN"**

**Cause:** No matched points or constant data

**Solution:** Check that datasets have variation
```python
if result['n_points'].iloc[0] == 0:
    print("No data matched - check tolerance or data structure")
```

---

## 📈 Visualization Example

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Get data for scatter plot
from subcorticalSTRF.data_loader import match_data_points

matched_rates, matched_meta = match_data_points(
    cochlea_means, bez_means, 
    run2_idx=0
)

x = [pt['data1_rate'] for pt in matched_rates]  # Cochlea
y = [pt['data2_rate'] for pt in matched_rates]  # BEZ

# Get regression stats
result = regress_cochlea_vs_bez_single_run(cochlea_means, bez_means, run_idx=0)
slope = result['slope'].iloc[0]
intercept = result['intercept'].iloc[0]
r = result['r_value'].iloc[0]

# Create scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(x, y, alpha=0.5, s=20)
plt.plot(x, np.array(x) * slope + intercept, 'r-', label=f'Regression (r={r:.3f})')
plt.plot([min(x), max(x)], [min(x), max(x)], 'k--', label='Identity (y=x)')
plt.xlabel('Cochlea Mean Rate (Hz)')
plt.ylabel('BEZ Mean Rate (Hz)')
plt.title(f'Cochlea vs BEZ Run 0\ny = {slope:.3f}x + {intercept:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cochlea_vs_bez_regression.png', dpi=300)
plt.show()
```

---

## 🎓 Understanding the Hybrid Design

**You use simple functions:**
```python
regress_by_fiber_type(cochlea, bez)  # Clear, descriptive name
```

**Under the hood, it uses flexible engines:**
```python
def regress_by_fiber_type(...):
    return _analyze_by_groups(...)  # Generic engine
```

**Benefits:**
- ✅ Simple for users (you!)
- ✅ Easy to maintain (less code duplication)
- ✅ Flexible (can add new analyses easily)
- ✅ Testable (can test engines separately)

---

## 📚 Next Steps

1. **Try the basic example** above with your data
2. **Experiment with different groupings** (by CF, dB, fiber type)
3. **Visualize results** using the plotting example
4. **Compare to your existing analysis** in `compare_models_v3.py`

For questions or issues, check the function docstrings:
```python
help(regress_cochlea_vs_bez_single_run)
```
