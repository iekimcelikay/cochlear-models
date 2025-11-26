#!/usr/bin/env python3
"""
Simple example script for visualizing Cochlea vs BEZ regressions

This script demonstrates the two new plotting functions:
1. plot_regression_scatter() - Scatter plot with regression line
2. plot_slope_distribution() - Distribution of slopes across runs
"""

import matplotlib.pyplot as plt
from subcorticalSTRF.compare_bez_cochlea_mainscript import (
    load_bez_psth_data,
    load_matlab_data,
    mean_across_time_psth_cochlea,
    mean_across_time_psth_bez,
    plot_regression_scatter,
    plot_slope_distribution
)

def main():
    print("="*60)
    print("VISUALIZING COCHLEA vs BEZ REGRESSIONS")
    print("="*60)
    
    # ===================================================================
    # STEP 1: Load Data
    # ===================================================================
    print("\n1. Loading data...")
    
    # Load BEZ with individual runs
    bez_data, params = load_bez_psth_data()
    
    # Load Cochlea
    _, cochlea_data = load_matlab_data()
    
    print("✓ Data loaded successfully")
    
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
    # STEP 3: Plot Regression Scatter (Run 0)
    # ===================================================================
    print("\n3. Creating regression scatter plot...")
    
    fig1, ax1 = plot_regression_scatter(
        cochlea_means,
        bez_means,
        run_idx=0,  # Compare to BEZ Run 0
        title=None,  # Auto-generate title
        figsize=(8, 8),
        save_path='cochlea_vs_bez_run0_scatter.png'
    )
    
    print("✓ Scatter plot created")
    
    # ===================================================================
    # STEP 4: Plot Slope Distribution
    # ===================================================================
    print("\n4. Creating slope distribution plot...")
    
    fig2, ax2, slopes = plot_slope_distribution(
        cochlea_means,
        bez_means,
        bins=15,  # Number of histogram bins
        title=None,  # Auto-generate title
        figsize=(10, 6),
        save_path='slope_distribution.png'
    )
    
    print("✓ Slope distribution created")
    
    # ===================================================================
    # STEP 5: Show Plots
    # ===================================================================
    print("\n5. Displaying plots...")
    print("\nClose the plot windows to exit.")
    
    plt.show()
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - cochlea_vs_bez_run0_scatter.png")
    print("  - slope_distribution.png")


if __name__ == '__main__':
    main()
