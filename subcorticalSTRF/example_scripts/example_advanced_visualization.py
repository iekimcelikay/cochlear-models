#!/usr/bin/env python3
"""
Advanced visualization examples showing how to extend the basic functions

This demonstrates how to customize and add features to the plotting functions.
"""

import matplotlib.pyplot as plt
import numpy as np
from subcorticalSTRF.compare_bez_cochlea_mainscript import (
    load_bez_psth_data,
    load_matlab_data,
    mean_across_time_psth_cochlea,
    mean_across_time_psth_bez,
    plot_regression_scatter,
    plot_slope_distribution,
    match_data_points,
    regress_cochlea_vs_bez_all_runs
)


def plot_regression_by_fiber_type(cochlea_means, bez_means, run_idx=0, 
                                   save_path=None):
    """
    Extended: Plot separate regression for each fiber type in subplots
    """
    fiber_types = ['hsr', 'msr', 'lsr']
    colors = {'hsr': 'darkred', 'msr': 'darkgreen', 'lsr': 'darkblue'}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, fiber_type in zip(axes, fiber_types):
        # Get matched data for this fiber type
        matched_rates, _ = match_data_points(
            cochlea_means, bez_means,
            fiber_type=fiber_type,
            run2_idx=run_idx,
            verbose=False
        )
        
        x = np.array([pt['data1_rate'] for pt in matched_rates])
        y = np.array([pt['data2_rate'] for pt in matched_rates])
        
        # Compute regression
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, _ = linregress(x, y)
        
        # Scatter
        ax.scatter(x, y, alpha=0.6, s=40, color=colors[fiber_type], 
                   edgecolors='black', linewidth=0.5)
        
        # Regression line
        x_line = np.array([x.min(), x.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=2.5,
                label=f'y = {slope:.3f}x + {intercept:.2f}')
        
        # Identity line
        identity = np.array([min(x.min(), y.min()), max(x.max(), y.max())])
        ax.plot(identity, identity, 'k--', linewidth=1.5, alpha=0.6)
        
        # Formatting
        ax.set_xlabel('Cochlea (Hz)', fontsize=11)
        ax.set_ylabel('BEZ (Hz)', fontsize=11)
        ax.set_title(f'{fiber_type.upper()}\nr = {r_value:.3f}, n = {len(x)}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle(f'Cochlea vs BEZ Run {run_idx} by Fiber Type', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, axes


def plot_slope_comparison(cochlea_means, bez_means, groupby='fiber_type',
                          save_path=None):
    """
    Extended: Compare slope distributions grouped by fiber type or CF
    """
    from subcorticalSTRF.compare_bez_cochlea_mainscript import regress_by_fiber_type
    
    if groupby == 'fiber_type':
        # Get regression for each fiber type
        by_fiber = regress_by_fiber_type(cochlea_means, bez_means, 
                                          run2_idx=0, verbose=False)
        
        groups = ['hsr', 'msr', 'lsr']
        slopes = [by_fiber[ft]['slope'].iloc[0] for ft in groups]
        colors_list = ['darkred', 'darkgreen', 'darkblue']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(groups, slopes, color=colors_list, alpha=0.7, 
                      edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar, slope in zip(bars, slopes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{slope:.3f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Reference line at slope = 1
        ax.axhline(1.0, color='green', linestyle='--', linewidth=2,
                   label='Perfect scaling (slope = 1)', zorder=0)
        
        ax.set_xlabel('Fiber Type', fontsize=13)
        ax.set_ylabel('Regression Slope', fontsize=13)
        ax.set_title('Regression Slopes by Fiber Type\nCochlea vs BEZ Run 0',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, ax


def plot_slope_evolution_across_runs(cochlea_means, bez_means, save_path=None):
    """
    Extended: Show how slopes change across BEZ runs
    """
    # Get all runs
    all_runs = regress_cochlea_vs_bez_all_runs(
        cochlea_means, bez_means, verbose=False
    )
    
    # Extract slopes and r-values
    run_indices = sorted(all_runs.keys())
    slopes = [all_runs[i]['slope'].iloc[0] for i in run_indices]
    r_values = [all_runs[i]['r_value'].iloc[0] for i in run_indices]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Slope evolution
    ax1.plot(run_indices, slopes, 'o-', color='steelblue', linewidth=2.5,
             markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax1.axhline(np.mean(slopes), color='red', linestyle='--', linewidth=2,
                label=f'Mean = {np.mean(slopes):.3f}')
    ax1.axhline(1.0, color='green', linestyle=':', linewidth=2,
                label='Perfect scaling = 1.0')
    ax1.fill_between(run_indices, 
                      np.mean(slopes) - np.std(slopes),
                      np.mean(slopes) + np.std(slopes),
                      alpha=0.2, color='red', label=f'±1 SD = {np.std(slopes):.3f}')
    ax1.set_ylabel('Regression Slope', fontsize=12)
    ax1.set_title('Slope Evolution Across BEZ Runs', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Correlation evolution
    ax2.plot(run_indices, r_values, 's-', color='darkgreen', linewidth=2.5,
             markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax2.axhline(np.mean(r_values), color='red', linestyle='--', linewidth=2,
                label=f'Mean r = {np.mean(r_values):.3f}')
    ax2.set_xlabel('BEZ Run Index', fontsize=12)
    ax2.set_ylabel('Correlation (r)', fontsize=12)
    ax2.set_title('Correlation Evolution Across BEZ Runs', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, (ax1, ax2)


def main():
    print("="*60)
    print("ADVANCED VISUALIZATION EXAMPLES")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    bez_data, params = load_bez_psth_data()
    _, cochlea_data = load_matlab_data()
    
    print("Computing mean rates...")
    cochlea_means = mean_across_time_psth_cochlea(cochlea_data, params)
    bez_means = mean_across_time_psth_bez(bez_data, params)
    
    # Example 1: Regression by fiber type
    print("\n1. Creating regression by fiber type...")
    fig1, _ = plot_regression_by_fiber_type(
        cochlea_means, bez_means, run_idx=0,
        save_path='regression_by_fiber_type.png'
    )
    
    # Example 2: Slope comparison
    print("\n2. Creating slope comparison...")
    fig2, _ = plot_slope_comparison(
        cochlea_means, bez_means,
        save_path='slope_comparison_by_fiber.png'
    )
    
    # Example 3: Slope evolution
    print("\n3. Creating slope evolution plot...")
    fig3, _ = plot_slope_evolution_across_runs(
        cochlea_means, bez_means,
        save_path='slope_evolution_across_runs.png'
    )
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - regression_by_fiber_type.png")
    print("  - slope_comparison_by_fiber.png")
    print("  - slope_evolution_across_runs.png")
    
    plt.show()


if __name__ == '__main__':
    main()
