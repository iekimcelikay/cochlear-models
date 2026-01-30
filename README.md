File created on: 26/01/2026

- You need to run the script in project root for the imports to work.


Last edit to the repo: 30/01/2026
Changed the repository name from 'cochlear-models' to 'auditory-pRF-subcortical'

Commits:
https://github.com/iekimcelikay/auditory-pRF-subcortical/commits/main/

---
## IMPORTANT:
- Use the cochlea package in this repo, its' Zilany2014 model codes are updated for Python 3.
- To make thorns work in Python3, these changes might be needed:
-> replace `import imp` with `import importlib as imp`
-> replace `from collections` with `collections.abc

## Changelog

### 2026-01-28: Modularization and Power-Law Analysis

**New Files Created:**
- `utils/calculate_population_rate.py` - Weighted population rate calculation across AN fiber types (HSR/MSR/LSR)
- `utils/cochlea_loader_functions.py` - Reusable data loading and organization functions
- `visualization/plot_cochlea_output.py` - Visualization utilities for cochlea output

**Modified Files:**
- `powerlaw_simulation_260126_1651.py` - Added power-law transformation and visualization functions

**New Functionality:**

*Data Processing:*
- `calculate_population_rate()` - Compute weighted average across fiber types with configurable weights (default: 63% HSR, 25% MSR, 12% LSR). Moved to utils as its own script.
- `load_cochlea_results()` - Load .npz simulation files and compute population responses
- `organize_for_eachtone_allCFs()` - Create 2D response matrices (num_cf × num_tones) for analysis

*Power-Law Transformations:*
- `apply_power_function()` - Apply power-law nonlinearity: r^n
- `apply_power_with_percf_normalization()` - Power transformation with per-CF normalization
- `apply_multiple_power_with_percf_normalization()` - Batch processing for multiple exponents

*Visualization:*
- `plot_spectrogram_forCFS()` - Heatmap spectrogram of CF responses
- `plot_tuning_curves()` - Grid of tuning curves for all CFs
- `plot_single_cf_single_exponent()` - Single CF with one power exponent
- `plot_single_cf_multiple_exponents()` - Compare multiple exponents for one CF
- `plot_all_cfs_multiple_exponents()` - Grid showing all CFs with all exponents

*Analysis (In Progress):*
- `calculate_Q10()` - Q10 tuning sharpness factor (TODO)

### TO-DOS:
- Move the remaining visualization functions to their scripts (from powerlawsimulation script)
- Update the jupyternotebook for powerlawsimulation
- (Maybe) try implementing lateral inhibition
