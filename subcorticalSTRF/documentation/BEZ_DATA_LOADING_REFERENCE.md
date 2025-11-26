# BEZ Data Loading - Quick Reference Card

## ⚠️ Critical: Two Different BEZ Loaders

### **OPTION 1: Individual Runs** ✅ RECOMMENDED

```python
from subcorticalSTRF.data_loader import load_bez_psth_data

bez_data, params = load_bez_psth_data()
```

**What it loads:**
- File: `psth_data_128fibers.mat`
- Location: `BEZ2018_meanrate/results/processed_data/`
- Structure: `bez_data['hsr_all'][cf_idx, freq_idx, db_idx, run_idx]`
- Dimensions: [16 CFs × 9 freqs × 3 dBs × 10 runs]

**What you can do:**
- ✅ All regression functions
- ✅ Run-by-run analysis
- ✅ `regress_cochlea_vs_bez_all_runs()`
- ✅ `regress_all_bez_run_pairs()`
- ✅ `regress_bez_run_vs_run()`
- ✅ Full flexibility

**Use when:** You need individual run data (most analyses)

---

### **OPTION 2: Aggregated Mean** ⚠️ LIMITED

```python
from subcorticalSTRF.data_loader import load_matlab_data

bez_data, cochlea_data = load_matlab_data()
```

**What it loads:**
- File: `bez_acrossruns_psths.mat`
- Location: `BEZ2018_meanrate/results/processed_data/`
- Structure: `bez_data['bez_rates'].hsr[cf_idx, freq_idx, db_idx]`
- Dimensions: [16 CFs × 9 freqs × 3 dBs] - **NO run dimension**

**What you can do:**
- ✅ `regress_cochlea_vs_bez_single_run()` (treats as single "aggregated run")
- ✅ `regress_by_fiber_type()`, `regress_by_cf()`, `regress_by_db()`
- ❌ `regress_cochlea_vs_bez_all_runs()` - NO RUNS!
- ❌ `regress_all_bez_run_pairs()` - NO RUNS!
- ❌ `regress_bez_run_vs_run()` - NO RUNS!

**Use when:** You only need the averaged BEZ response (rare)

---

## 🎯 Recommended Workflow

```python
from subcorticalSTRF.data_loader import (
    load_bez_psth_data,      # ← Individual runs
    load_matlab_data,        # ← Cochlea only
    mean_across_time_psth_cochlea,
    mean_across_time_psth_bez,
    regress_cochlea_vs_bez_all_runs
)

# Step 1: Load data
bez_data, params = load_bez_psth_data()      # BEZ with runs
_, cochlea_data = load_matlab_data()         # Cochlea (no BEZ)

# Step 2: Process
cochlea_means = mean_across_time_psth_cochlea(cochlea_data, params)
bez_means = mean_across_time_psth_bez(bez_data, params)

# Step 3: Analyze
all_runs = regress_cochlea_vs_bez_all_runs(cochlea_means, bez_means)
print(f"Run 0: r = {all_runs[0]['r_value'].iloc[0]:.3f}")
```

---

## 🔍 How to Check What You Loaded

```python
# After loading, check the structure:

# Individual runs (load_bez_psth_data):
print(bez_data['hsr_all'].shape)  
# Output: (16, 9, 3, 10)  ← 4D with run dimension
#         (cf, freq, db, run)

# Aggregated (load_matlab_data):
print(bez_data['bez_rates'].hsr.shape)
# Output: (16, 9, 3)  ← 3D, NO run dimension
#         (cf, freq, db)
```

---

## ❓ FAQ

### **Q: Which should I use?**
**A:** Use `load_bez_psth_data()` unless you specifically only need the averaged BEZ response.

### **Q: Can I load both?**
**A:** Yes, but they're different files:
```python
# Individual runs
bez_runs, params = load_bez_psth_data()

# Aggregated mean
bez_agg, cochlea = load_matlab_data()
```

### **Q: What if I already loaded with load_matlab_data()?**
**A:** Reload with `load_bez_psth_data()` to get individual runs:
```python
# Replace this:
bez_data, cochlea_data = load_matlab_data()

# With this:
bez_data, params = load_bez_psth_data()
_, cochlea_data = load_matlab_data()  # Still need cochlea
```

### **Q: Why two different files?**
**A:** 
- `psth_data_128fibers.mat` - Raw output from all simulation runs
- `bez_acrossruns_psths.mat` - Processed average for quick comparisons

---

## 📊 Data Structure Comparison

### **Individual Runs (load_bez_psth_data):**
```
bez_data
├── hsr_all: [16, 9, 3, 10]  ← 10 runs
├── msr_all: [16, 9, 3, 10]  ← 10 runs
├── lsr_all: [16, 9, 3, 10]  ← 10 runs
├── cfs: [16]
├── frequencies: [9]
└── dbs: [3]
```

After `mean_across_time_psth_bez()`:
```
bez_means
├── hsr
│   ├── 125.0 (CF)
│   │   ├── 125.0 (freq)
│   │   │   ├── 30.0 (dB)
│   │   │   │   ├── 0: 10.3  ← Run 0
│   │   │   │   ├── 1: 10.5  ← Run 1
│   │   │   │   └── ...
```

### **Aggregated (load_matlab_data):**
```
bez_data
├── bez_rates
│   ├── hsr: [16, 9, 3]  ← NO run dimension
│   ├── msr: [16, 9, 3]  ← NO run dimension
│   └── lsr: [16, 9, 3]  ← NO run dimension
├── cfs: [16]
├── frequencies: [9]
└── dbs: [3]
```

After `mean_across_time_psth_bez()`:
```
bez_means
├── hsr
│   ├── 125.0 (CF)
│   │   ├── 125.0 (freq)
│   │   │   └── 30.0 (dB): 10.4  ← Single value (averaged)
```

---

## 🎯 Bottom Line

**For regression analysis:** Use `load_bez_psth_data()` to access individual runs!

**For quick averaged comparison:** Use `load_matlab_data()` but limited functionality.

**When in doubt:** Use `load_bez_psth_data()` - it gives you the most flexibility!
