#!/usr/bin/env python3
"""
Minimal example: Just show how to call the fiber-type plotting functions

This is a template - you'll need to have your data loaded first.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from subcorticalSTRF.compare_bez_cochlea_mainscript import (
    plot_regression_by_fiber_type,
    plot_slope_distribution_by_fiber_type
)

# ==============================================================================
# USAGE EXAMPLES (uncomment when you have data loaded)
# ==============================================================================

# Assuming you have already loaded and processed your data:
# 
# from subcorticalSTRF.data_loader import (
#     load_bez_psth_data,
#     load_matlab_data,  # or load cochlea separately
#     mean_across_time_psth_cochlea,
#     mean_across_time_psth_bez
# )
#
# bez_data, params = load_bez_psth_data()
# cochlea_data = ...  # Load your cochlea data
# cochlea_means = mean_across_time_psth_cochlea(cochlea_data, params)
# bez_means = mean_across_time_psth_bez(bez_data, params)


# ==============================================================================
# EXAMPLE 1: Regression scatter plots by fiber type
# ==============================================================================

# fig1, axes1, results = plot_regression_by_fiber_type(
#     cochlea_means,
#     bez_means,
#     run_idx=0,              # Compare to BEZ Run 0
#     figsize=(18, 6),        # Wide figure for 3 subplots
#     save_path='regression_by_fiber_type.png'
# )
#
# # Access results
# print(f"HSR slope: {results['hsr']['slope']:.3f}")
# print(f"MSR slope: {results['msr']['slope']:.3f}")
# print(f"LSR slope: {results['lsr']['slope']:.3f}")


# ==============================================================================
# EXAMPLE 2: Slope distribution by fiber type
# ==============================================================================

# fig2, axes2, slopes_dict = plot_slope_distribution_by_fiber_type(
#     cochlea_means,
#     bez_means,
#     bins=15,
#     figsize=(18, 6),
#     save_path='slope_distribution_by_fiber_type.png'
# )
#
# # Access slope arrays
# import numpy as np
# for fiber_type in ['hsr', 'msr', 'lsr']:
#     slopes = slopes_dict[fiber_type]
#     print(f"{fiber_type.upper()}: mean={np.mean(slopes):.3f}, std={np.std(slopes):.3f}")


# ==============================================================================
# SHOW PLOTS
# ==============================================================================

# plt.show()


print("""
================================================================================
FIBER-TYPE PLOTTING FUNCTIONS - USAGE TEMPLATE
================================================================================

This script shows how to use the new plotting functions.

STEP 1: Load your data
----------------------
from subcorticalSTRF.data_loader import (
    load_bez_psth_data,
    mean_across_time_psth_cochlea,
    mean_across_time_psth_bez
)

bez_data, params = load_bez_psth_data()
cochlea_data = ...  # Load your cochlea data
cochlea_means = mean_across_time_psth_cochlea(cochlea_data, params)
bez_means = mean_across_time_psth_bez(bez_data, params)


STEP 2: Plot regression by fiber type
-------------------------------------
fig, axes, results = plot_regression_by_fiber_type(
    cochlea_means,
    bez_means,
    run_idx=0,
    save_path='regression_by_fiber.png'
)

This creates 3 scatter plots side-by-side (HSR, MSR, LSR)


STEP 3: Plot slope distribution by fiber type
---------------------------------------------
fig, axes, slopes = plot_slope_distribution_by_fiber_type(
    cochlea_means,
    bez_means,
    save_path='slopes_by_fiber.png'
)

This creates 3 histograms side-by-side showing slope distributions


STEP 4: Show or analyze results
-------------------------------
plt.show()  # Display plots

# Or access results programmatically:
print(f"HSR slope: {results['hsr']['slope']:.3f}")
print(f"MSR mean slope: {np.mean(slopes['msr']):.3f}")


OUTPUT FILES:
------------
- regression_by_fiber.png  (3 scatter plots)
- slopes_by_fiber.png      (3 histograms)


CUSTOMIZATION:
-------------
- Change run_idx to compare different BEZ runs
- Adjust bins for histogram granularity
- Modify figsize for different aspect ratios
- Set save_path=None to display without saving

================================================================================
Uncomment the code above to run with your data!
================================================================================
""")
