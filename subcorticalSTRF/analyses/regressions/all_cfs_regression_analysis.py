#!/usr/bin/env python3
# STILL IN USE
# Will be moved to model comparison script later

"""
All-CFs Linear Regression Analysis: BEZ vs Cochlea Model Comparison

This script performs comprehensive linear regression analysis comparing BEZ and Cochlea 
auditory nerve models using ALL characteristic frequencies (CFs) combined at a specific
dB level. Unlike CF-specific analysis, this pools all data points from all CFs and 
tone frequencies to create a single regression with ~80,000 data points per fiber type.

Key Features:
- Single regression using all CFs and tone frequencies combined
- Individual data points (not means) - ~80,000 points per analysis
- User-specified dB level analysis
- Comprehensive statistical reporting with confidence intervals
- Scatter plots with regression lines and confidence bands
- Analysis across LSR, MSR, and HSR fiber types
- Detailed JSON and text output files

Data Structure: 20 CFs × 20 tone frequencies × 200 time points = 80,000 data points
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import linregress
import os
from pathlib import Path
from datetime import datetime
import argparse
from subcorticalSTRF.compare_bez_cochlea_mainscript import (convert_to_numeric, load_matlab_data, extract_parameters)

# Set seaborn style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100




def calculate_regression_confidence_intervals(x_data, y_data, slope, intercept, confidence=0.95):
    """
    Calculate confidence intervals for regression parameters and predictions
    
    Args:
        x_data: Input x values used for regression
        y_data: Input y values used for regression  
        slope: Regression slope
        intercept: Regression intercept
        confidence: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        dict: Contains parameter confidence intervals, standard errors, and prediction intervals
    """
    from scipy import stats
    
    n = len(x_data)
    if n < 3:
        return None
    
    # Convert to numpy arrays
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    # Calculate residuals and statistics
    y_pred = slope * x_data + intercept
    residuals = y_data - y_pred
    mse = np.sum(residuals**2) / (n - 2)  # Mean squared error
    x_mean = np.mean(x_data)
    sxx = np.sum((x_data - x_mean)**2)
    
    # Standard errors for parameters
    se_slope = np.sqrt(mse / sxx)
    se_intercept = np.sqrt(mse * (1/n + x_mean**2/sxx))
    
    # T-critical value for confidence interval
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, n - 2)
    
    # Parameter confidence intervals
    slope_ci = (slope - t_crit * se_slope, slope + t_crit * se_slope)
    intercept_ci = (intercept - t_crit * se_intercept, intercept + t_crit * se_intercept)
    
    return {
        'slope_ci': slope_ci,
        'intercept_ci': intercept_ci,
        'se_slope': se_slope,
        'se_intercept': se_intercept,
        'mse': mse,
        't_crit': t_crit,
        'degrees_freedom': n - 2
    }

def create_68_confidence_bands(x_data, y_data, slope, intercept, x_range):
    """
    Create 68% confidence bands (±1σ) for regression visualization
    
    Args:
        x_data: Original x data used for regression
        y_data: Original y data used for regression
        slope: Regression slope
        intercept: Regression intercept
        x_range: X values for which to calculate bands
        
    Returns:
        dict: Contains fitted line and 68% confidence bands
    """
    n = len(x_data)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_range = np.array(x_range)
    
    # Calculate regression statistics
    y_pred_orig = slope * x_data + intercept
    residuals = y_data - y_pred_orig
    mse = np.sum(residuals**2) / (n - 2)
    x_mean = np.mean(x_data)
    sxx = np.sum((x_data - x_mean)**2)
    
    # Standard error of slope (for 68% CI, use ±1σ instead of t-critical)
    se_slope = np.sqrt(mse / sxx)
    
    # Fitted line
    y_fit = slope * x_range + intercept
    
    # 68% confidence bands for slope (±1σ)
    slope_upper = slope + se_slope
    slope_lower = slope - se_slope
    
    y_upper = slope_upper * x_range + intercept
    y_lower = slope_lower * x_range + intercept
    
    return {
        'x_range': x_range,
        'y_fit': y_fit,
        'y_upper': y_upper,
        'y_lower': y_lower,
        'se_slope': se_slope
    }

def create_all_cfs_regression(bez_data, cochlea_data, fiber_type='lsr', db_idx=1, 
                             color_palette='viridis'):
    """
    Create linear regression analysis using ALL CFs combined at specified dB level
    
    Args:
        bez_data: MATLAB data structure containing BEZ model results
        cochlea_data: MATLAB data structure containing Cochlea model results
        fiber_type: Type of auditory nerve fiber ('lsr', 'msr', 'hsr')
        db_idx: Index for dB level (0=50dB, 1=60dB, 2=70dB, 3=80dB)
        color_palette: Seaborn color palette name for visualization
        
    Returns:
        dict: Comprehensive regression results and statistics
    """
    
    # Extract parameters
    bez_cfs, bez_frequencies, bez_dbs = extract_parameters(bez_data, "bez")
    
    # Extract PSTH data
    bez_rates = {
        'lsr': bez_data['bez_rates'].lsr,
        'msr': bez_data['bez_rates'].msr,
        'hsr': bez_data['bez_rates'].hsr
    }
    
    cochlea_rates = {
        'lsr': cochlea_data['lsr_all'],
        'msr': cochlea_data['msr_all'],
        'hsr': cochlea_data['hsr_all']
    }
    
    db_level = bez_dbs[db_idx]
    print(f"\nCreating ALL-CFs regression for {fiber_type.upper()} fibers at {db_level} dB")
    print("Pooling data from all CFs and tone frequencies...")
    
    # Collect ALL data points from ALL CFs and frequencies
    all_bez_points = []
    all_cochlea_points = []
    all_cf_labels = []
    all_freq_labels = []
    
    total_points = 0
    processed_combinations = 0
    
    for cf_idx, cf_val in enumerate(bez_cfs):
        for freq_idx, freq_val in enumerate(bez_frequencies):
            try:
                bez_psth = bez_rates[fiber_type][cf_idx, freq_idx, db_idx]
                cochlea_psth = cochlea_rates[fiber_type][cf_idx, freq_idx, db_idx]
                
                # Ensure both are arrays and finite
                bez_psth = np.atleast_1d(bez_psth).flatten()
                cochlea_psth = np.atleast_1d(cochlea_psth).flatten()
                
                # Remove non-finite values
                clean_mask = np.isfinite(bez_psth) & np.isfinite(cochlea_psth)
                
                if np.sum(clean_mask) > 0:
                    bez_clean = bez_psth[clean_mask]
                    cochlea_clean = cochlea_psth[clean_mask]
                    
                    all_bez_points.extend(bez_clean)
                    all_cochlea_points.extend(cochlea_clean)
                    all_cf_labels.extend([cf_val] * len(bez_clean))
                    all_freq_labels.extend([freq_val] * len(bez_clean))
                    
                    total_points += len(bez_clean)
                    processed_combinations += 1
                    
            except (IndexError, ValueError) as e:
                print(f"Warning: Skipping CF={cf_val}Hz, Freq={freq_val}Hz due to: {e}")
                continue
    
    print(f"Successfully collected {total_points:,} data points from {processed_combinations} CF-frequency combinations")
    print(f"Expected: {len(bez_cfs)} CFs × {len(bez_frequencies)} frequencies = {len(bez_cfs) * len(bez_frequencies)} combinations")
    
    if total_points < 100:
        print("Warning: Very few data points collected. Check data integrity.")
        return None
    
    # Convert to numpy arrays
    all_bez_points = np.array(all_bez_points)
    all_cochlea_points = np.array(all_cochlea_points)
    all_cf_labels = np.array(all_cf_labels)
    all_freq_labels = np.array(all_freq_labels)
    
    # Perform linear regression
    print("Performing linear regression on all data points...")
    slope, intercept, r_value, p_value, std_err = linregress(all_bez_points, all_cochlea_points)
    r_squared = r_value**2
    
    # Calculate confidence intervals
    ci_results = calculate_regression_confidence_intervals(
        all_bez_points, all_cochlea_points, slope, intercept, confidence=0.95)
    
    print(f"Regression Results:")
    print(f"  R² = {r_squared:.6f}")
    print(f"  Slope = {slope:.6f}")
    print(f"  Intercept = {intercept:.6f}")
    print(f"  P-value = {p_value:.2e}")
    print(f"  Standard Error = {std_err:.6f}")
    
    if ci_results:
        print(f"  95% CI for slope: ({ci_results['slope_ci'][0]:.6f}, {ci_results['slope_ci'][1]:.6f})")
        print(f"  95% CI for intercept: ({ci_results['intercept_ci'][0]:.6f}, {ci_results['intercept_ci'][1]:.6f})")
        
        # Calculate 68% confidence interval (±1σ) for slope
        se_slope_68 = ci_results['se_slope']
        slope_68_lower = slope - se_slope_68
        slope_68_upper = slope + se_slope_68
        print(f"  68% CI for slope: ({slope_68_lower:.6f}, {slope_68_upper:.6f})")
        print(f"  SEM for slope: {se_slope_68:.6f}")
    
    # Create visualization with CF-frequency based coloring
    print("Creating visualization with all data points...")
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Get seaborn color palette
    n_cfs = len(bez_cfs)
    base_colors = sns.mpl_palette(color_palette, n_colors=n_cfs)
    
    # Create color mapping: hue by CF, brightness by CF-frequency difference
    print("Generating color scheme: hue by CF, brightness by CF-frequency difference...")
    
    # Prepare color arrays for all points
    point_colors = []
    
    for i, (bez_val, cochlea_val, cf_val, freq_val) in enumerate(zip(
            all_bez_points, all_cochlea_points, all_cf_labels, all_freq_labels)):
        
        # Find CF index for base color (hue)
        cf_idx = np.argmin(np.abs(bez_cfs - cf_val))
        base_color = base_colors[cf_idx]
        
        # Calculate brightness based on CF-frequency difference
        cf_freq_diff = abs(cf_val - freq_val)
        
        # Use a more appropriate normalization
        # Maximum possible difference would be from lowest freq to highest CF
        max_possible_diff = max(max(bez_cfs) - min(bez_frequencies), 
                               max(bez_frequencies) - min(bez_cfs))
        normalized_diff = cf_freq_diff / max_possible_diff if max_possible_diff > 0 else 0
        
        # For exact matches (CF == frequency), set normalized_diff to 0
        if abs(cf_freq_diff) < 1:  # Within 1 Hz tolerance
            normalized_diff = 0
        
        # Brightness varies from 0.2 (dark) to 1.0 (bright) for better contrast
        # Closer to CF = brighter, farther from CF = darker
        brightness = 1.0 - (normalized_diff * 0.8)  # Range: 0.2 to 1.0
        
        # Apply brightness to base color
        colored_point = [c * brightness for c in base_color]
        point_colors.append(colored_point)
    
    # Convert to numpy array for plotting
    point_colors = np.array(point_colors)
    
    # Create scatter plot with uniform point size
    print(f"Plotting all {total_points:,} data points...")
    scatter = ax.scatter(all_bez_points, all_cochlea_points, 
                        c=point_colors, alpha=0.7, s=15, rasterized=True)
    
    # Add regression line and 68% confidence bands
    x_min, x_max = np.min(all_bez_points), np.max(all_bez_points)
    x_range = np.linspace(x_min, x_max, 200)
    
    # Calculate 68% confidence bands
    bands = create_68_confidence_bands(all_bez_points, all_cochlea_points, 
                                      slope, intercept, x_range)
    
    # Use a representative color from the palette for regression line
    # Choose last color (extreme) from palette for better contrast
    regression_color = base_colors[-1] if len(base_colors) > 0 else 'darkblue'
    
    # Plot regression line and bands
    ax.plot(bands['x_range'], bands['y_fit'], color=regression_color, linewidth=2, 
           label='Regression line', zorder=10)
    ax.fill_between(bands['x_range'], bands['y_lower'], bands['y_upper'], 
                   alpha=0.3, color=regression_color, label='68% Confidence band (±1σ)', zorder=5)
    ax.plot([x_min, x_max], [x_min, x_max], 'k--', alpha=0.7, linewidth=1, 
           label='Perfect match', zorder=8)

    ax.set_xlabel('BEZ Model Rate (spikes/s)', fontsize=12)
    ax.set_ylabel('Cochlea Model Rate (spikes/s)', fontsize=12)
    ax.set_title(f'{fiber_type.upper()} Fibers: ALL CFs Combined at {db_level} dB\n'
                f'N = {total_points:,} points | R² = {r_squared:.4f} | Slope = {slope:.4f}\n'
                , fontsize=14, fontweight='bold')

    # Calculate 68% CI for slope display
    se_slope_68 = ci_results['se_slope'] if ci_results else 0
    slope_68_lower = slope - se_slope_68
    slope_68_upper = slope + se_slope_68

    # First legend: regression statistics and lines (upper left)
    from matplotlib.lines import Line2D
    
    regression_elements = [
        Line2D([0], [0], color=regression_color, linewidth=2, label='Regression line'),
        Line2D([0], [0], color=regression_color, alpha=0.3, linewidth=8, label='68% Confidence band (±1σ)'),
        Line2D([0], [0], color='black', linestyle='--', alpha=0.7, linewidth=1, label='Perfect match'),
        Line2D([0], [0], linestyle='None', label=''),  # Spacer
        Line2D([0], [0], linestyle='None', label=f'Slope: {slope:.4f}'),
        Line2D([0], [0], linestyle='None', label=f'SEM: {se_slope_68:.4f}'),
        Line2D([0], [0], linestyle='None', label=f'68% CI: ({slope_68_lower:.4f}, {slope_68_upper:.4f})'),
        Line2D([0], [0], linestyle='None', label=f'R²: {r_squared:.4f}'),
        Line2D([0], [0], linestyle='None', label=f'P: {p_value:.2e}')
    ]
    
    regression_legend = ax.legend(handles=regression_elements, fontsize=10, loc='upper left',
                                 title='Regression & Statistics', title_fontsize=11,
                                 frameon=True, fancybox=True, shadow=True)
    ax.add_artist(regression_legend)  # Keep this legend while adding another

    ax.grid(True, alpha=0.3)    # Create CF color legend similar to CF-specific regression script
    from matplotlib.lines import Line2D
    
    # CF Colors legend elements
    cf_elements = []
    for i, cf_val in enumerate(bez_cfs):
        base_color = base_colors[i]
        cf_elements.append(Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=base_color, markersize=8,
                                 markeredgecolor='black', markeredgewidth=0.3,
                                 label=f'CF {cf_val:.0f} Hz'))
    
    # Add brightness explanation
    cf_elements.extend([
        Line2D([0], [0], linestyle='None', label=''),  # Spacer
        Line2D([0], [0], linestyle='None', label='Brightness by CF-Freq Distance:'),
        Line2D([0], [0], marker='o', color='w', 
              markerfacecolor='gray', alpha=0.1, markersize=6,
              markeredgecolor='black', markeredgewidth=0.3,
              label='Far from CF (dim)'),
        Line2D([0], [0], marker='o', color='w', 
              markerfacecolor='gray', alpha=1.0, markersize=6,
              markeredgecolor='black', markeredgewidth=0.3,
              label='At CF (bright)')
    ])
    
    # Create CF colors legend (center right)
    cf_legend = ax.legend(handles=cf_elements, fontsize=8, loc='center left', 
                         bbox_to_anchor=(1.02, 0.5), title='CF Colors', 
                         title_fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.78)  # Make room for the legend
    
    # Compile comprehensive results
    results = {
        'fiber_type': fiber_type,
        'db_level': float(db_level),
        'db_idx': db_idx,
        'color_palette': color_palette,
        'total_points': total_points,
        'processed_combinations': processed_combinations,
        'n_cfs': len(bez_cfs),
        'n_frequencies': len(bez_frequencies),
        'regression_stats': {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_value': float(r_value),
            'r_squared': float(r_squared),
            'p_value': float(p_value),
            'std_err': float(std_err)
        },
        'confidence_intervals': ci_results,
        'data_arrays': {
            'bez_points': all_bez_points,
            'cochlea_points': all_cochlea_points,
            'cf_labels': all_cf_labels,
            'freq_labels': all_freq_labels
        },
        'confidence_bands_68': bands,
        'figure': fig
    }
    
    return results

def save_comprehensive_results(results, output_dir):
    """
    Save comprehensive regression results to files
    
    Args:
        results: Dictionary containing regression results
        output_dir: Directory to save output files
    """
    from scipy import stats
    import json
    
    fiber_type = results['fiber_type']
    db_level = results['db_level']
    total_points = results['total_points']
    
    # Create detailed statistics file
    stats_file = os.path.join(output_dir, f'all_cfs_regression_statistics_{fiber_type}_{db_level}dB.txt')
    json_file = os.path.join(output_dir, f'all_cfs_regression_statistics_{fiber_type}_{db_level}dB.json')
    
    regression_stats = results['regression_stats']
    ci_results = results['confidence_intervals']
    
    with open(stats_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write(f"ALL-CFs LINEAR REGRESSION ANALYSIS: BEZ vs COCHLEA MODEL COMPARISON\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Analysis Type: All-CFs Combined Linear Regression\n")
        f.write(f"Fiber Type: {fiber_type.upper()}\n")
        f.write(f"Sound Level: {db_level} dB SPL\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Data Points: {total_points:,}\n")
        f.write(f"CF-Frequency Combinations: {results['processed_combinations']}\n")
        f.write(f"Number of CFs: {results['n_cfs']}\n")
        f.write(f"Number of Tone Frequencies: {results['n_frequencies']}\n")
        f.write(f"Confidence Level: 95% (also reporting 68% CI)\n\n")
        
        # Regression statistics
        f.write("=" * 100 + "\n")
        f.write("REGRESSION STATISTICS\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Slope:                   {regression_stats['slope']:12.6f}\n")
        f.write(f"Intercept:               {regression_stats['intercept']:12.6f}\n")
        f.write(f"R-value (correlation):   {regression_stats['r_value']:12.6f}\n")
        f.write(f"R² (coefficient of det): {regression_stats['r_squared']:12.6f}\n")
        f.write(f"P-value:                 {regression_stats['p_value']:12.2e}\n")
        f.write(f"Standard Error:          {regression_stats['std_err']:12.6f}\n\n")
        
        if ci_results:
            f.write("=" * 100 + "\n")
            f.write("CONFIDENCE INTERVALS\n")
            f.write("=" * 100 + "\n\n")
            
            # Calculate 68% CI (±1σ)
            sem_slope = ci_results['se_slope']
            slope_68_ci_lower = regression_stats['slope'] - sem_slope
            slope_68_ci_upper = regression_stats['slope'] + sem_slope
            
            f.write(f"Slope 95% CI:       ({ci_results['slope_ci'][0]:10.6f}, {ci_results['slope_ci'][1]:10.6f})\n")
            f.write(f"Slope 68% CI:       ({slope_68_ci_lower:10.6f}, {slope_68_ci_upper:10.6f})\n")
            f.write(f"Intercept 95% CI:   ({ci_results['intercept_ci'][0]:10.6f}, {ci_results['intercept_ci'][1]:10.6f})\n")
            f.write(f"Standard Error of Slope (SEM): {ci_results['se_slope']:12.6f}\n")
            f.write(f"Standard Error of Intercept:   {ci_results['se_intercept']:12.6f}\n")
            f.write(f"Mean Squared Error:            {ci_results['mse']:12.6f}\n")
            f.write(f"Degrees of Freedom:            {ci_results['degrees_freedom']:12.0f}\n")
            f.write(f"T-critical value (95%):        {ci_results['t_crit']:12.6f}\n")
            f.write(f"68% CI Range (±1σ):            {2 * sem_slope:12.6f}\n\n")
        
        # Model comparison interpretation
        f.write("=" * 100 + "\n")
        f.write("MODEL COMPARISON INTERPRETATION\n")
        f.write("=" * 100 + "\n\n")
        f.write("Slope Interpretation:\n")
        slope = regression_stats['slope']
        if 0.9 <= slope <= 1.1:
            f.write(f"  Near Perfect Agreement: slope = {slope:.6f} (0.9-1.1 range)\n")
        elif slope < 0.9:
            f.write(f"  Cochlea Model Higher Response: slope = {slope:.6f} < 0.9\n")
        else:
            f.write(f"  BEZ Model Higher Response: slope = {slope:.6f} > 1.1\n")
        
        f.write(f"\nR² Interpretation:\n")
        r2 = regression_stats['r_squared']
        if r2 >= 0.8:
            f.write(f"  Excellent Agreement: R² = {r2:.6f} ≥ 0.8\n")
        elif r2 >= 0.6:
            f.write(f"  Good Agreement: R² = {r2:.6f} (0.6-0.8)\n")
        elif r2 >= 0.4:
            f.write(f"  Moderate Agreement: R² = {r2:.6f} (0.4-0.6)\n")
        else:
            f.write(f"  Poor Agreement: R² = {r2:.6f} < 0.4\n")
        
        # Statistical significance
        pval = regression_stats['p_value']
        f.write(f"\nStatistical Significance:\n")
        if pval < 0.001:
            f.write(f"  Highly Significant: p = {pval:.2e} < 0.001\n")
        elif pval < 0.01:
            f.write(f"  Very Significant: p = {pval:.2e} < 0.01\n")
        elif pval < 0.05:
            f.write(f"  Significant: p = {pval:.2e} < 0.05\n")
        else:
            f.write(f"  Not Significant: p = {pval:.2e} ≥ 0.05\n")
    
    # Save as JSON for programmatic access
    json_data = {
        'fiber_type': fiber_type,
        'db_level': float(db_level),
        'db_idx': results['db_idx'],
        'analysis_date': datetime.now().isoformat(),
        'data_summary': {
            'total_points': total_points,
            'processed_combinations': results['processed_combinations'],
            'n_cfs': results['n_cfs'],
            'n_frequencies': results['n_frequencies']
        },
        'regression_statistics': {
            'slope': float(regression_stats['slope']),
            'intercept': float(regression_stats['intercept']),
            'r_value': float(regression_stats['r_value']),
            'r_squared': float(regression_stats['r_squared']),
            'p_value': float(regression_stats['p_value']),
            'std_err': float(regression_stats['std_err'])
        },
        'confidence_intervals': {
            'slope_ci_95_lower': float(ci_results['slope_ci'][0]) if ci_results else None,
            'slope_ci_95_upper': float(ci_results['slope_ci'][1]) if ci_results else None,
            'slope_ci_68_lower': float(regression_stats['slope'] - ci_results['se_slope']) if ci_results else None,
            'slope_ci_68_upper': float(regression_stats['slope'] + ci_results['se_slope']) if ci_results else None,
            'slope_ci_68_range': float(2 * ci_results['se_slope']) if ci_results else None,
            'intercept_ci_lower': float(ci_results['intercept_ci'][0]) if ci_results else None,
            'intercept_ci_upper': float(ci_results['intercept_ci'][1]) if ci_results else None,
            'se_slope': float(ci_results['se_slope']) if ci_results else None,
            'se_intercept': float(ci_results['se_intercept']) if ci_results else None,
            'mse': float(ci_results['mse']) if ci_results else None,
            'degrees_freedom': int(ci_results['degrees_freedom']) if ci_results else None
        }
    }
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Comprehensive results saved:")
    print(f"  Text report: {stats_file}")
    print(f"  JSON data: {json_file}")

def main():
    """Main function with command line argument support"""
    
    parser = argparse.ArgumentParser(description='All-CFs Linear Regression Analysis')
    parser.add_argument('--db_idx', type=int, default=1, 
                       help='dB index: 0=50dB, 1=60dB, 2=70dB, 3=80dB (default: 1 for 60dB)')
    parser.add_argument('--fiber_types', nargs='+', default=['lsr', 'msr', 'hsr'],
                       choices=['lsr', 'msr', 'hsr'], 
                       help='Fiber types to analyze (default: all)')
    parser.add_argument('--color_palette', type=str, default='viridis',
                       help='Seaborn color palette for visualization (default: viridis)')
    
    args = parser.parse_args()
    
    try:
        print("=== All-CFs Linear Regression Analysis ===\n")
        print("Analyzing BEZ vs Cochlea model agreement using ALL characteristic frequencies combined...")
        print(f"Using dB index {args.db_idx} (corresponding dB level will be shown in results)")
        print(f"Color palette: {args.color_palette}\n")
        
        # Load data
        bez_data, cochlea_data = load_matlab_data()
        
        # Create output directory
        output_dir = "//subcorticalSTRF/all_cfs_regression_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze each requested fiber type
        db_levels = ['50dB', '60dB', '70dB', '80dB']
        print(f"Analyzing at {db_levels[args.db_idx]}...")
        
        all_results = {}
        
        for fiber_type in args.fiber_types:
            print(f"\n=== Processing {fiber_type.upper()} Fibers ===")
            
            # Create all-CFs regression
            results = create_all_cfs_regression(
                bez_data, cochlea_data, 
                fiber_type=fiber_type, 
                db_idx=args.db_idx,
                color_palette=args.color_palette
            )
            
            if results is None:
                print(f"Failed to analyze {fiber_type.upper()} fibers")
                continue
            
            all_results[fiber_type] = results
            
            # Save plots
            db_level = results['db_level']
            plot_file = os.path.join(output_dir, f'all_cfs_regression_{fiber_type}_{db_level}dB.png')
            results['figure'].savefig(plot_file, dpi=300, bbox_inches='tight')
            results['figure'].savefig(plot_file.replace('.png', '.pdf'), bbox_inches='tight')
            print(f"Regression plots saved: {plot_file}")
            
            # Save CSV data
            csv_file = os.path.join(output_dir, f'all_cfs_regression_data_{fiber_type}_{db_level}dB.csv')
            data_df = pd.DataFrame({
                'bez_rate': results['data_arrays']['bez_points'],
                'cochlea_rate': results['data_arrays']['cochlea_points'],
                'cf_hz': results['data_arrays']['cf_labels'],
                'tone_freq_hz': results['data_arrays']['freq_labels']
            })
            data_df.to_csv(csv_file, index=False)
            print(f"Raw data saved: {csv_file}")
            
            # Save comprehensive statistics
            save_comprehensive_results(results, output_dir)
            
            # Print summary
            reg_stats = results['regression_stats']
            ci_info = results['confidence_intervals']
            
            # Calculate 68% CI for summary
            se_slope = ci_info['se_slope'] if ci_info else 0
            slope_68_lower = reg_stats['slope'] - se_slope
            slope_68_upper = reg_stats['slope'] + se_slope
            
            print(f"\nSummary for {fiber_type.upper()}:")
            print(f"  Total data points: {results['total_points']:,}")
            print(f"  R² = {reg_stats['r_squared']:.6f}")
            print(f"  Slope = {reg_stats['slope']:.6f}")
            print(f"  SEM for slope = {se_slope:.6f}")
            print(f"  68% CI for slope = ({slope_68_lower:.6f}, {slope_68_upper:.6f})")
            print(f"  P-value = {reg_stats['p_value']:.2e}")
            
            plt.show(block=False)
        
        # Cross-fiber comparison
        print("\n=== Cross-Fiber Type Comparison ===")
        print("Fiber Type | Data Points |     R²     |   Slope   |    SEM    | 68% CI Slope Range  |  P-value")
        print("-" * 85)
        for fiber_type, results in all_results.items():
            reg_stats = results['regression_stats']
            ci_info = results['confidence_intervals']
            se_slope = ci_info['se_slope'] if ci_info else 0
            slope_68_lower = reg_stats['slope'] - se_slope
            slope_68_upper = reg_stats['slope'] + se_slope
            ci_range = slope_68_upper - slope_68_lower
            
            print(f"{fiber_type.upper():10} | {results['total_points']:11,} | "
                  f"{reg_stats['r_squared']:10.6f} | {reg_stats['slope']:9.6f} | "
                  f"{se_slope:9.6f} | {ci_range:19.6f} | {reg_stats['p_value']:8.2e}")
        
        print("\n=== All-CFs Regression Analysis Complete! ===")
        print("Linear regression analysis completed using all CFs combined for each fiber type.")
        
        # Keep plots alive
        plt.ion()
        try:
            print("Script running to keep plots open. Press Ctrl+C to exit.")
            while True:
                plt.pause(1)
        except KeyboardInterrupt:
            print("\nExiting script.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()