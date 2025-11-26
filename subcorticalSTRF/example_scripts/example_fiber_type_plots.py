#!/usr/bin/env python3
"""
Example: Visualize regressions separately for each fiber type

Shows regression scatter plots and slope distributions for HSR, MSR, LSR
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from subcorticalSTRF.compare_bez_cochlea_mainscript import (
    load_bez_psth_data,
    load_matlab_data,
    mean_across_time_psth_cochlea,
    mean_across_time_psth_bez,
    plot_regression_by_fiber_type,
    plot_slope_distribution_by_fiber_type
)

def main():
    print("="*70)
    print("COCHLEA vs BEZ REGRESSION - BY FIBER TYPE")
    print("="*70)
    
    # ===================================================================
    # STEP 1: Load Data
    # ===================================================================
    print("\n1. Loading data...")
    
    # Load BEZ with individual runs
    bez_data, params = load_bez_psth_data()
    
    # Load Cochlea - try to get it from load_matlab_data, or use alternative
    try:
        _, cochlea_data = load_matlab_data()
    except FileNotFoundError:
        # If load_matlab_data fails, load cochlea directly
        from scipy.io import loadmat
        cochlea_dir = Path("subcorticalSTRF/cochlea_meanrate/out/condition_psths")
        cochlea_file = cochlea_dir / 'cochlea_psths.mat'
        
        if not cochlea_file.exists():
            print(f"\nError: Could not find cochlea data at {cochlea_file}")
            print("Please update the path to your cochlea data file.")
            return
        
        print(f"Loading cochlea data from: {cochlea_file}")
        cochlea_data = loadmat(str(cochlea_file), struct_as_record=False, squeeze_me=True)
    
    print("✓ Data loaded")
    
    # ===================================================================
    # STEP 2: Compute Mean Rates
    # ===================================================================
    print("\n2. Computing mean PSTH rates...")
    
    cochlea_means = mean_across_time_psth_cochlea(
        cochlea_data,
        params,
        fiber_types=['hsr', 'msr', 'lsr']
    )
    
    bez_means = mean_across_time_psth_bez(
        bez_data,
        params,
        fiber_types=['hsr', 'msr', 'lsr']
    )
    
    print("✓ Mean rates computed")
    
    # ===================================================================
    # STEP 3: Plot Regression Scatter by Fiber Type
    # ===================================================================
    print("\n3. Creating regression scatter plots by fiber type...")
    
    fig1, axes1, results = plot_regression_by_fiber_type(
        cochlea_means,
        bez_means,
        run_idx=0,  # Compare to BEZ Run 0
        figsize=(18, 6),
        save_path='regression_by_fiber_type.png'
    )
    
    print("\n✓ Scatter plots created")
    
    # ===================================================================
    # STEP 4: Plot Slope Distribution by Fiber Type
    # ===================================================================
    print("\n4. Creating slope distribution by fiber type...")
    
    fig2, axes2, slopes_dict = plot_slope_distribution_by_fiber_type(
        cochlea_means,
        bez_means,
        bins=15,
        figsize=(18, 6),
        save_path='slope_distribution_by_fiber_type.png'
    )
    
    print("\n✓ Slope distributions created")
    
    # ===================================================================
    # STEP 5: Print Summary
    # ===================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nRegression Results (Run 0):")
    print("-" * 70)
    for fiber_type in ['hsr', 'msr', 'lsr']:
        if fiber_type in results:
            r = results[fiber_type]
            print(f"{fiber_type.upper():3s}: slope={r['slope']:6.3f}, "
                  f"intercept={r['intercept']:7.2f}, "
                  f"r={r['r_value']:.3f}, "
                  f"n={r['n_points']:3d}")
    
    print("\nSlope Distribution Across All Runs:")
    print("-" * 70)
    for fiber_type in ['hsr', 'msr', 'lsr']:
        if fiber_type in slopes_dict:
            slopes = slopes_dict[fiber_type]
            print(f"{fiber_type.upper():3s}: mean={np.mean(slopes):6.3f} ± "
                  f"{np.std(slopes):.3f}, "
                  f"range=[{slopes.min():.3f}, {slopes.max():.3f}]")
    
    print("\n" + "="*70)
    print("FILES SAVED:")
    print("  - regression_by_fiber_type.png")
    print("  - slope_distribution_by_fiber_type.png")
    print("="*70)
    
    # ===================================================================
    # STEP 6: Display Plots
    # ===================================================================
    print("\nDisplaying plots... Close windows to exit.")
    plt.show()


if __name__ == '__main__':
    main()
