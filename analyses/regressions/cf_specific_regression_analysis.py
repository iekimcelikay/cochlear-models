#!/usr/bin/env python3
"""
CF-Specific Linear Regression Analysis: BEZ vs Cochlea Model Comparison

This script performs comprehensive linear regression analysis comparing BEZ and Cochlea 
auditory nerve models across different characteristic frequencies (CFs). For each CF,
it creates individual regression analyses using firing rate responses to different 
tone frequencies at a user-specified dB level, providing detailed statistical 
comparisons between the two models with confidence intervals and performance metrics.

Key Features:
- CF-specific regression plots with confidence bands
- User-specified dB level analysis via command line arguments
- Comprehensive statistical reporting (R², slopes, confidence intervals)
- Analysis across LSR, MSR, and HSR fiber types  
- Detailed JSON and text output files
- Customizable color palettes for visualization
- Command line argument support for flexible analysis

Usage Examples:
    python cf_specific_regression_analysis.py --db_idx 1 --fiber_types lsr msr
    python cf_specific_regression_analysis.py --db_idx 3 --color_palette viridis
    python cf_specific_regression_analysis.py --fiber_types hsr --color_palette managua
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import linregress
import os
from pathlib import Path
from matplotlib.lines import Line2D
from datetime import datetime

# Set seaborn style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100

def load_matlab_data():
    """
    Load BEZ and Cochlea auditory nerve model data from MATLAB files
    
    Returns:
        tuple: (bez_data, cochlea_data) containing model responses across
               different characteristic frequencies, tone frequencies, and dB levels
    """
    
    # Load BEZ model data
    bez_dir = Path("//subcorticalSTRF/BEZ2018_meanrate/results/processed_data")
    bez_file = bez_dir / 'bez_acrossruns_psths.mat'
    
    print(f"Loading BEZ data from: {bez_file}")
    bez_data = loadmat(str(bez_file), struct_as_record=False, squeeze_me=True)
    print("✓ BEZ data loaded successfully")
    
    # Load cochlea model data
    cochlea_dir = Path("//subcorticalSTRF/cochlea_meanrate/out/condition_psths")
    cochlea_file = cochlea_dir / 'cochlea_psths.mat'
    
    print(f"Loading cochlea data from: {cochlea_file}")
    cochlea_data = loadmat(str(cochlea_file), struct_as_record=False, squeeze_me=True)
    print("✓ Cochlea data loaded successfully")
    
    return bez_data, cochlea_data

def convert_to_numeric(arr):
    """Convert string arrays to numeric arrays"""
    arr = np.atleast_1d(arr).flatten()
    if arr.dtype.kind in ['U', 'S']:
        return np.array([float(str(x)) for x in arr])
    else:
        return arr.astype(float)

def extract_parameters(data, model_type="bez"):
    """Extract parameters from loaded data"""
    print(f"Extracting parameters for {model_type} model...")
    
    cfs = convert_to_numeric(data['cfs'])
    frequencies = convert_to_numeric(data['frequencies'])
    dbs = convert_to_numeric(data['dbs'])
    
    print(f"  CFs: {len(cfs)} values")
    print(f"  Frequencies: {len(frequencies)} values")
    print(f"  dB levels: {dbs}")
    
    return np.sort(cfs), np.sort(frequencies), np.sort(dbs)

def create_cf_specific_regressions(bez_data, cochlea_data, fiber_type='lsr', db_idx=1, color_palette='Spectral'):
    """
    Create CF-specific linear regression plots comparing BEZ vs Cochlea models
    
    For each characteristic frequency (CF), this function:
    1. Extracts firing rate responses across all tone frequencies at 60 dB
    2. Performs linear regression between BEZ and Cochlea responses
    3. Calculates confidence intervals and statistical metrics
    4. Creates subplot visualizations with regression lines and confidence bands
    
    Args:
        bez_data: MATLAB data structure containing BEZ model results
        cochlea_data: MATLAB data structure containing Cochlea model results
        fiber_type: Type of auditory nerve fiber ('lsr', 'msr', 'hsr')
        db_idx: Index for dB level (1 corresponds to 60 dB SPL)
        color_palette: Color palette for visualizations
        
    Returns:
        pandas.DataFrame: Regression statistics for each CF
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
    print(f"Creating CF-specific regressions for {fiber_type.upper()} fibers at {db_level} dB")
    
    # Calculate grid dimensions
    n_cfs = len(bez_cfs)
    n_cols = int(np.ceil(np.sqrt(n_cfs)))
    n_rows = int(np.ceil(n_cfs / n_cols))
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    fig.suptitle(f'{fiber_type.upper()} Fibers: Linear Regression by CF at {db_level} dB\n'
                f'Each subplot = one CF, points = all tone frequencies', fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier indexing
    if n_cfs == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Color palette for frequency differences
    colors = sns.color_palette(color_palette, n_colors=len(bez_frequencies))
    freq_color_map = dict(zip(bez_frequencies, colors))
    
    # Storage for regression statistics and legend data
    cf_regression_stats = []
    cf_legend_data = []  # For unified legend
    
    for cf_idx, cf_val in enumerate(bez_cfs):
        ax = axes[cf_idx]
        all_bez_points, all_cochlea_points, all_frequencies = [], [], []

        for freq_idx, freq_val in enumerate(bez_frequencies):
            try:
                bez_psth = bez_rates[fiber_type][cf_idx, freq_idx, db_idx]
                cochlea_psth = cochlea_rates[fiber_type][cf_idx, freq_idx, db_idx]
                clean_mask = np.isfinite(bez_psth) & np.isfinite(cochlea_psth)
                if np.sum(clean_mask) > 0:
                    bez_clean = bez_psth[clean_mask]
                    cochlea_clean = cochlea_psth[clean_mask]
                    all_bez_points.extend(bez_clean)
                    all_cochlea_points.extend(cochlea_clean)
                    all_frequencies.extend([freq_val] * len(bez_clean))
                    ax.scatter(bez_clean, cochlea_clean,
                               color=freq_color_map[freq_val], alpha=0.6, s=20)
            except (IndexError, ValueError):
                continue

        # Perform regression + CI plotting
        if len(all_bez_points) > 2:
            slope, intercept, r_value, p_value, std_err = linregress(all_bez_points, all_cochlea_points)
            r_squared = r_value**2
            
            # Calculate 68% confidence intervals and SEM
            ci_68_results = calculate_parameter_confidence_intervals(
                all_bez_points, all_cochlea_points, slope, intercept, confidence=0.68)
            
            # Store statistics with CI and SEM
            cf_stat = {
                'cf': cf_val,
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'r_squared': r_squared,
                'p_value': p_value,
                'std_err': std_err,
                'n_points': len(all_bez_points),
                'n_frequencies': len(set(all_frequencies))
            }
            
            if ci_68_results:
                cf_stat.update({
                    'se_slope': ci_68_results['se_slope'],
                    'se_intercept': ci_68_results['se_intercept'],
                    'slope_ci_68_lower': ci_68_results['slope_ci'][0],
                    'slope_ci_68_upper': ci_68_results['slope_ci'][1],
                    'intercept_ci_68_lower': ci_68_results['intercept_ci'][0],
                    'intercept_ci_68_upper': ci_68_results['intercept_ci'][1]
                })
            else:
                cf_stat.update({
                    'se_slope': np.nan,
                    'se_intercept': np.nan,
                    'slope_ci_68_lower': np.nan,
                    'slope_ci_68_upper': np.nan,
                    'intercept_ci_68_lower': np.nan,
                    'intercept_ci_68_upper': np.nan
                })
            
            cf_regression_stats.append(cf_stat)
            
            # Store color and CF info for legend
            cf_color = colors[cf_idx % len(colors)]
            cf_legend_data.append({
                'cf': cf_val,
                'color': cf_color,
                'slope': slope,
                'r_squared': r_squared,
                'se_slope': cf_stat.get('se_slope', np.nan),
                'slope_ci_68_lower': cf_stat.get('slope_ci_68_lower', np.nan),
                'slope_ci_68_upper': cf_stat.get('slope_ci_68_upper', np.nan)
            })

            # --- Plot regression lines (no legend on subplot) ---
            x_min, x_max = min(all_bez_points), max(all_bez_points)
            x_range = np.linspace(x_min, x_max, 200)

            conf_lines = slope_confidence_lines(all_bez_points, all_cochlea_points,
                                                slope, intercept, x_range)
            ax.plot(conf_lines['x_range'], conf_lines['y_fit'],
                    color=cf_color, linewidth=2)  # Use CF-specific color
            ax.plot(conf_lines['x_range'], conf_lines['y_plus'],
                    color=cf_color, linestyle='--', linewidth=1, alpha=0.7)
            ax.plot(conf_lines['x_range'], conf_lines['y_minus'],
                    color=cf_color, linestyle='--', linewidth=1, alpha=0.7)

            # Perfect match line
            ax.plot([x_min, x_max], [x_min, x_max], 'k--', alpha=0.3, linewidth=1)
            
            # Add detailed statistics text to subplot
            stats_text = f'R² = {r_squared:.3f}\nSlope = {slope:.3f}'
            
            # Add SEM and 68% CI if available
            if ci_68_results and not np.isnan(ci_68_results['se_slope']):
                se_slope = ci_68_results['se_slope']
                ci_low = ci_68_results['slope_ci'][0]
                ci_high = ci_68_results['slope_ci'][1]
                stats_text += f'\nSEM = {se_slope:.4f}'
                stats_text += f'\n68% CI: ({ci_low:.3f}, {ci_high:.3f})'
            
            stats_text += f'\np = {p_value:.2e}'
            stats_text += f'\nn = {len(all_bez_points)}'
            
            # Position text in upper left corner of subplot
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_title(f'CF = {cf_val:.0f} Hz')
        ax.set_xlabel('BEZ Model Rate (spikes/s)')
        ax.set_ylabel('Cochlea Model Rate (spikes/s)')
        ax.grid(True, alpha=0.3)
        # NO LEGEND ON INDIVIDUAL SUBPLOTS
    
    # Hide unused subplots
    for idx in range(n_cfs, len(axes)):
        axes[idx].set_visible(False)
    
    # Create unified legend on the right side with all CFs
    from matplotlib.lines import Line2D
    
    legend_elements = []
    
    # Add title for CF legend
    legend_elements.append(Line2D([0], [0], linestyle='None', 
                                 label='CF Statistics (68% CI):'))
    legend_elements.append(Line2D([0], [0], linestyle='None', label=''))  # Spacer
    
    # Add each CF with its color, slope, SEM, and 68% CI
    for cf_data in cf_legend_data:
        cf_val = cf_data['cf']
        cf_color = cf_data['color']
        slope = cf_data['slope']
        se_slope = cf_data['se_slope']
        ci_lower = cf_data['slope_ci_68_lower']
        ci_upper = cf_data['slope_ci_68_upper']
        r2 = cf_data['r_squared']
        
        # Format the label with SEM and 68% CI
        if not np.isnan(se_slope):
            label = f'CF {cf_val:.0f}Hz: Slope={slope:.3f}, SEM={se_slope:.3f}\n    68%CI=({ci_lower:.3f},{ci_upper:.3f}), R²={r2:.3f}'
        else:
            label = f'CF {cf_val:.0f}Hz: Slope={slope:.3f}, R²={r2:.3f}'
        
        legend_elements.append(Line2D([0], [0], color=cf_color, linewidth=3,
                                     label=label))
    
    # Add explanation elements
    legend_elements.append(Line2D([0], [0], linestyle='None', label=''))  # Spacer
    legend_elements.append(Line2D([0], [0], color='black', linestyle='-', linewidth=2,
                                 label='Solid: Regression line'))
    legend_elements.append(Line2D([0], [0], color='gray', linestyle='--', linewidth=1,
                                 label='Dashed: ±1σ slope bounds'))
    legend_elements.append(Line2D([0], [0], color='black', linestyle='--', alpha=0.3,
                                 label='Dotted: Perfect match'))
    
    # Create the legend and place it on the right with reduced spacing
    fig_legend = fig.legend(handles=legend_elements, fontsize=9, loc='center left', 
                           bbox_to_anchor=(0.85, 0.5), title='All CFs Summary', 
                           title_fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.82)  # Reduced space for the legend
    
    # Convert regression stats to DataFrame
    regression_df = pd.DataFrame(cf_regression_stats)
    
    print(f"Successfully processed {len(regression_df)} CFs")
    print(f"Mean R² across CFs: {regression_df['r_squared'].mean():.3f}")
    print(f"Mean slope across CFs: {regression_df['slope'].mean():.3f}")
    
    return fig, regression_df, db_level, color_palette


# This function below is the one thing that I want. 
def calculate_parameter_confidence_intervals(x_data, y_data, slope, intercept, confidence=0.68):
    """
    Calculate confidence intervals for regression parameters (for statistical reporting)
    
    Args:
        x_data: Input x values used for regression
        y_data: Input y values used for regression  
        slope: Regression slope
        intercept: Regression intercept
        confidence: Confidence level (e.g., 0.68 for %)
        
    Returns:
        dict: Contains parameter confidence intervals and standard errors
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
        'se_intercept': se_intercept
    }

def slope_confidence_lines(x_data, y_data, slope, intercept, x_range=None):
    """
    Compute slope ±1σ lines for regression visualization.
    
    This shows how much the regression line could vary if the slope changes
    by ± its standard error (1σ, ~68% CI for slope).
    
    Args:
        x_data: array-like, predictor values
        y_data: array-like, response values
        slope: regression slope
        intercept: regression intercept
        x_range: array-like of x values to evaluate the lines. 
                 If None, use np.linspace(min(x_data), max(x_data), 200)
    
    Returns:
        dict with:
            'y_fit': central regression line
            'y_plus': line for slope +1σ
            'y_minus': line for slope -1σ
            'se_slope': standard error of slope
            'slope_1sigma_range': (slope - se_slope, slope + se_slope)
            'x_range': x values for plotting
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    n = len(x_data)
    if n < 3:
        return None

    if x_range is None:
        x_range = np.linspace(x_data.min(), x_data.max(), 200)
    else:
        x_range = np.array(x_range)
    
    # Compute standard error of slope
    x_mean = np.mean(x_data)
    sxx = np.sum((x_data - x_mean)**2)
    residuals = y_data - (slope * x_data + intercept)
    mse = np.sum(residuals**2) / (n - 2)
    se_slope = np.sqrt(mse / sxx)
    
    # Slope ± 1σ
    slope_plus = slope + se_slope
    slope_minus = slope - se_slope
    
    y_fit = slope * x_range + intercept
    y_plus = slope_plus * x_range + intercept
    y_minus = slope_minus * x_range + intercept
    
    return {
        'x_range': x_range,
        'y_fit': y_fit,
        'y_plus': y_plus,
        'y_minus': y_minus,
        'se_slope': se_slope,
        'slope_1sigma_range': (slope_minus, slope_plus)
    }


def save_detailed_regression_statistics(regression_df, fiber_type, db_level, color_palette, output_dir, 
                                       bez_data, cochlea_data, db_idx):
    """
    Save comprehensive CF-specific linear regression statistics with confidence intervals
    
    Creates detailed statistical reports comparing BEZ vs Cochlea models for each CF,
    including regression parameters, confidence intervals, and model performance metrics.
    
    Args:
        regression_df: DataFrame containing regression results for each CF
        fiber_type: Type of auditory nerve fiber ('lsr', 'msr', 'hsr')
        db_level: Sound pressure level in dB SPL
        color_palette: Color palette name for file identification
        output_dir: Directory to save output files
        bez_data: BEZ model data structure
        cochlea_data: Cochlea model data structure  
        db_idx: Index for dB level selection
    """
    
    from scipy import stats
    import json
    
    # Create detailed statistics file
    stats_file = os.path.join(output_dir, f'detailed_regression_statistics_{fiber_type}_{db_level}dB_{color_palette}.txt')
    json_file = os.path.join(output_dir, f'detailed_regression_statistics_{fiber_type}_{db_level}dB_{color_palette}.json')
    
    # Extract parameters for recalculating detailed stats
    bez_cfs, bez_frequencies, bez_dbs = extract_parameters(bez_data, "bez")
    
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
    
    detailed_stats = {}
    
    print(f"Calculating detailed statistics for {fiber_type.upper()} fibers...")
    
    with open(stats_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write(f"CF-SPECIFIC LINEAR REGRESSION ANALYSIS: BEZ vs COCHLEA MODEL COMPARISON\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Analysis Type: CF-Specific Linear Regression\n")
        f.write(f"Fiber Type: {fiber_type.upper()}\n")
        f.write(f"Sound Level: {db_level} dB SPL\n")
        f.write(f"Color Palette: {color_palette}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total CFs Analyzed: {len(regression_df)}\n")
        f.write(f"Confidence Level: 68%\n")
        f.write(f"Note: Plots show 68% confidence bands (≈±1σ) for regression lines\n\n")
        
        # Overall summary statistics
        f.write("=" * 100 + "\n")
        f.write("OVERALL SUMMARY STATISTICS\n")
        f.write("=" * 100 + "\n\n")
        
        # R-squared statistics
        r2_stats = regression_df['r_squared'].describe()
        f.write("R² (Coefficient of Determination) Statistics:\n")
        f.write(f"  Count:      {r2_stats['count']:8.0f}\n")
        f.write(f"  Mean:       {r2_stats['mean']:8.4f}\n")
        f.write(f"  Std:        {r2_stats['std']:8.4f}\n")
        f.write(f"  Min:        {r2_stats['min']:8.4f}\n")
        f.write(f"  25%:        {r2_stats['25%']:8.4f}\n")
        f.write(f"  Median:     {r2_stats['50%']:8.4f}\n")
        f.write(f"  75%:        {r2_stats['75%']:8.4f}\n")
        f.write(f"  Max:        {r2_stats['max']:8.4f}\n\n")
        
        # Slope statistics
        slope_stats = regression_df['slope'].describe()
        f.write("Slope Statistics:\n")
        f.write(f"  Count:      {slope_stats['count']:8.0f}\n")
        f.write(f"  Mean:       {slope_stats['mean']:8.4f}\n")
        f.write(f"  Std:        {slope_stats['std']:8.4f}\n")
        f.write(f"  Min:        {slope_stats['min']:8.4f}\n")
        f.write(f"  25%:        {slope_stats['25%']:8.4f}\n")
        f.write(f"  Median:     {slope_stats['50%']:8.4f}\n")
        f.write(f"  75%:        {slope_stats['75%']:8.4f}\n")
        f.write(f"  Max:        {slope_stats['max']:8.4f}\n\n")
        
        # P-value statistics
        pval_stats = regression_df['p_value'].describe()
        f.write("P-value Statistics:\n")
        f.write(f"  Count:      {pval_stats['count']:8.0f}\n")
        f.write(f"  Mean:       {pval_stats['mean']:8.2e}\n")
        f.write(f"  Std:        {pval_stats['std']:8.2e}\n")
        f.write(f"  Min:        {pval_stats['min']:8.2e}\n")
        f.write(f"  25%:        {pval_stats['25%']:8.2e}\n")
        f.write(f"  Median:     {pval_stats['50%']:8.2e}\n")
        f.write(f"  75%:        {pval_stats['75%']:8.2e}\n")
        f.write(f"  Max:        {pval_stats['max']:8.2e}\n\n")
        
        # Quality categorization
        f.write("=" * 100 + "\n")
        f.write("REGRESSION QUALITY CATEGORIZATION\n")
        f.write("=" * 100 + "\n\n")
        
        excellent = len(regression_df[regression_df['r_squared'] >= 0.8])
        good = len(regression_df[(regression_df['r_squared'] >= 0.6) & (regression_df['r_squared'] < 0.8)])
        moderate = len(regression_df[(regression_df['r_squared'] >= 0.4) & (regression_df['r_squared'] < 0.6)])
        poor = len(regression_df[regression_df['r_squared'] < 0.4])
        total = len(regression_df)
        
def create_seaborn_regression_plots(df, fiber_type='LSR', confidence_level=68, color_palette='viridis'):
    """
    Create CF-specific regression plots using seaborn.relplot
    Shows individual time points rather than averaged data
    
    Args:
        df: DataFrame with BEZ vs Cochlea data (individual time points)
        fiber_type: Fiber type name for titles
        confidence_level: Confidence level for bands (68 or 95)
        color_palette: Color palette name
    
    Returns:
        tuple: (figure, regression_stats_df)
    """
    
    # Get unique CFs
    cfs = sorted(df['CF'].unique())
    n_cfs = len(cfs)
    
    # Create relplot with regression lines for each CF
    g = sns.relplot(
        data=df, 
        x='BEZ', 
        y='Cochlea',
        col='CF',
        col_wrap=5,  # 5 plots per row
        kind='scatter',
        facet_kws={'sharex': False, 'sharey': False},
        height=3,
        aspect=1.0,
        alpha=0.6,
        s=20  # Point size
    )
    
    # Add regression lines and confidence bands to each subplot
    regression_stats = []
    
    for (cf_val, ax) in zip(cfs, g.axes.flat):
        # Filter data for this CF
        cf_data = df[df['CF'] == cf_val].copy()
        
        if len(cf_data) == 0:
            continue
            
        # Calculate regression statistics
        slope, intercept, r_value, p_value, std_err = linregress(cf_data['BEZ'].values, cf_data['Cochlea'].values)
        r_squared = r_value ** 2
        
        # Add regression line and confidence band using seaborn
        sns.regplot(
            data=cf_data,
            x='BEZ',
            y='Cochlea',
            ax=ax,
            scatter=False,  # Don't plot points again
            line_kws={'color': 'red', 'linewidth': 2},
            ci=confidence_level,  # Confidence interval
            color='red'
        )
        
        # Store statistics
        regression_stats.append({
            'CF': cf_val,
            'Slope': slope,

        f.write(f"Excellent (R² ≥ 0.8):    {excellent:3d} CFs ({excellent/total*100:5.1f}%)\n")
        f.write(f"Good (0.6 ≤ R² < 0.8):   {good:3d} CFs ({good/total*100:5.1f}%)\n")
        f.write(f"Moderate (0.4 ≤ R² < 0.6): {moderate:3d} CFs ({moderate/total*100:5.1f}%)\n")
        f.write(f"Poor (R² < 0.4):         {poor:3d} CFs ({poor/total*100:5.1f}%)\n\n")
        
        # Statistical significance analysis
        highly_sig = len(regression_df[regression_df['p_value'] < 0.001])
        very_sig = len(regression_df[(regression_df['p_value'] >= 0.001) & (regression_df['p_value'] < 0.01)])
        sig = len(regression_df[(regression_df['p_value'] >= 0.01) & (regression_df['p_value'] < 0.05)])
        not_sig = len(regression_df[regression_df['p_value'] >= 0.05])
        
        f.write("Statistical Significance:\n")
        f.write(f"Highly Significant (p < 0.001):     {highly_sig:3d} CFs ({highly_sig/total*100:5.1f}%)\n")
        f.write(f"Very Significant (0.001 ≤ p < 0.01): {very_sig:3d} CFs ({very_sig/total*100:5.1f}%)\n")
        f.write(f"Significant (0.01 ≤ p < 0.05):      {sig:3d} CFs ({sig/total*100:5.1f}%)\n")
        f.write(f"Not Significant (p ≥ 0.05):         {not_sig:3d} CFs ({not_sig/total*100:5.1f}%)\n\n")
        
        # Detailed CF-by-CF analysis with confidence intervals
        f.write("=" * 100 + "\n")
        f.write("DETAILED CF-BY-CF ANALYSIS WITH 68% CONFIDENCE INTERVALS\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("CF (Hz) |   R²   | Slope  | SEM Slope | Slope 68% CI        | Intercept | SEM Int  | Intercept 68% CI    | P-value  | N Points\n")
        f.write("-" * 130 + "\n")
        
        # Sort by R² for better organization
        sorted_df = regression_df.sort_values('r_squared', ascending=False).reset_index(drop=True)
        
        for idx, (_, row) in enumerate(sorted_df.iterrows()):
            cf_val = row['cf']
            
            # Get SEM and CI values from the regression_df (now includes these)
            se_slope = row.get('se_slope', np.nan)
            se_intercept = row.get('se_intercept', np.nan) 
            slope_ci_lower = row.get('slope_ci_68_lower', np.nan)
            slope_ci_upper = row.get('slope_ci_68_upper', np.nan)
            intercept_ci_lower = row.get('intercept_ci_68_lower', np.nan)
            intercept_ci_upper = row.get('intercept_ci_68_upper', np.nan)
            
            if not np.isnan(se_slope):
                f.write(f"{cf_val:7.0f} | {row['r_squared']:6.4f} | {row['slope']:6.4f} | "
                       f"{se_slope:9.4f} | ({slope_ci_lower:6.4f}, {slope_ci_upper:6.4f}) | "
                       f"{row['intercept']:9.4f} | {se_intercept:8.4f} | "
                       f"({intercept_ci_lower:7.4f}, {intercept_ci_upper:7.4f}) | "
                       f"{row['p_value']:8.2e} | {row['n_points']:8.0f}\n")
                
                # Store for JSON output
                detailed_stats[str(cf_val)] = {
                    'cf_hz': float(cf_val),
                    'r_squared': float(row['r_squared']),
                    'slope': float(row['slope']),
                    'intercept': float(row['intercept']),
                    'p_value': float(row['p_value']),
                    'std_err': float(row['std_err']),
                    'n_points': int(row['n_points']),
                    'n_frequencies': int(row['n_frequencies']),
                    'se_slope': float(se_slope),
                    'se_intercept': float(se_intercept),
                    'slope_ci_68_lower': float(slope_ci_lower),
                    'slope_ci_68_upper': float(slope_ci_upper),
                    'slope_ci_68_range': float(slope_ci_upper - slope_ci_lower),
                    'intercept_ci_68_lower': float(intercept_ci_lower),
                    'intercept_ci_68_upper': float(intercept_ci_upper)
                }
            else:
                f.write(f"{cf_val:7.0f} | {row['r_squared']:6.4f} | {row['slope']:6.4f} | "
                       f"{'N/A':>9} | {'N/A':>18} | {row['intercept']:9.4f} | "
                       f"{'N/A':>8} | {'N/A':>19} | {row['p_value']:8.2e} | {row['n_points']:8.0f}\n")
                
                # Store for JSON output (without CI data)
                detailed_stats[str(cf_val)] = {
                    'cf_hz': float(cf_val),
                    'r_squared': float(row['r_squared']),
                    'slope': float(row['slope']),
                    'intercept': float(row['intercept']),
                    'p_value': float(row['p_value']),
                    'std_err': float(row['std_err']),
                    'n_points': int(row['n_points']),
                    'n_frequencies': int(row['n_frequencies']),
                    'se_slope': None,
                    'se_intercept': None,
                    'slope_ci_68_lower': None,
                    'slope_ci_68_upper': None,
                    'slope_ci_68_range': None,
                    'intercept_ci_68_lower': None,
                    'intercept_ci_68_upper': None
                }
        
        # Model comparison analysis
        f.write("\n" + "=" * 100 + "\n")
        f.write("MODEL COMPARISON INTERPRETATION\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("Slope Interpretation:\n")
        near_perfect = len(regression_df[(regression_df['slope'] >= 0.9) & (regression_df['slope'] <= 1.1)])
        cochlea_higher = len(regression_df[regression_df['slope'] < 0.9])
        bez_higher = len(regression_df[regression_df['slope'] > 1.1])
        
        f.write(f"  Near Perfect Agreement (0.9-1.1): {near_perfect:3d} CFs ({near_perfect/total*100:5.1f}%)\n")
        f.write(f"  Cochlea Model Higher Response:     {cochlea_higher:3d} CFs ({cochlea_higher/total*100:5.1f}%)\n")
        f.write(f"  BEZ Model Higher Response:         {bez_higher:3d} CFs ({bez_higher/total*100:5.1f}%)\n\n")
        
        f.write("Best and Worst Performing CFs:\n")
        best_cf = regression_df.loc[regression_df['r_squared'].idxmax()]
        worst_cf = regression_df.loc[regression_df['r_squared'].idxmin()]
        f.write(f"  Best CF:  {best_cf['cf']:.0f} Hz (R² = {best_cf['r_squared']:.4f}, slope = {best_cf['slope']:.4f})\n")
        f.write(f"  Worst CF: {worst_cf['cf']:.0f} Hz (R² = {worst_cf['r_squared']:.4f}, slope = {worst_cf['slope']:.4f})\n\n")
        
        # Frequency range analysis
        f.write("Performance by Frequency Range:\n")
        low_freq = regression_df[regression_df['cf'] <= 500]
        mid_freq = regression_df[(regression_df['cf'] > 500) & (regression_df['cf'] <= 1500)]
        high_freq = regression_df[regression_df['cf'] > 1500]
        
        if len(low_freq) > 0:
            f.write(f"  Low Freq (≤500 Hz):   Mean R² = {low_freq['r_squared'].mean():.4f}, Mean slope = {low_freq['slope'].mean():.4f}\n")
        if len(mid_freq) > 0:
            f.write(f"  Mid Freq (500-1500):  Mean R² = {mid_freq['r_squared'].mean():.4f}, Mean slope = {mid_freq['slope'].mean():.4f}\n")
        if len(high_freq) > 0:
            f.write(f"  High Freq (>1500 Hz): Mean R² = {high_freq['r_squared'].mean():.4f}, Mean slope = {high_freq['slope'].mean():.4f}\n")
    
    # Save detailed statistics as JSON for programmatic access
    with open(json_file, 'w') as f:
        summary_stats = {
            'fiber_type': fiber_type,
            'db_level': float(db_level),
            'color_palette': color_palette,
            'analysis_date': datetime.now().isoformat(),
            'overall_statistics': {
                     \item Once the pRF modelling is done, move on to the second stage of the framework: STRF modelling.            'r_squared': {
                    'mean': float(regression_df['r_squared'].mean()),
                    'std': float(regression_df['r_squared'].std()),
                    'min': float(regression_df['r_squared'].min()),
                    'max': float(regression_df['r_squared'].max()),
                    'median': float(regression_df['r_squared'].median())
                },
                'slope': {
                    'mean': float(regression_df['slope'].mean()),
                    'std': float(regression_df['slope'].std()),
                    'min': float(regression_df['slope'].min()),
                    'max': float(regression_df['slope'].max()),
                    'median': float(regression_df['slope'].median())
                },
                'quality_categories': {
                    'excellent': int(excellent),
                    'good': int(good),
                    'moderate': int(moderate),
                    'poor': int(poor)
                }
            },
            'cf_specific_results': detailed_stats
        }
        json.dump(summary_stats, f, indent=2)
    
    print(f"Detailed regression statistics saved:")
    print(f"  Text report: {stats_file}")
    print(f"  JSON data: {json_file}")


def main():
    """Main function with command line argument support"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='CF-Specific Linear Regression Analysis')
    parser.add_argument('--db_idx', type=int, default=1, 
                       help='dB index: 0=50dB, 1=60dB, 2=70dB, 3=80dB (default: 1 for 60dB)')
    parser.add_argument('--fiber_types', nargs='+', default=['lsr', 'msr', 'hsr'],
                       choices=['lsr', 'msr', 'hsr'], 
                       help='Fiber types to analyze (default: all)')
    parser.add_argument('--color_palette', type=str, default='Spectral',
                       help='Seaborn color palette for visualization (default: Spectral)')
    
    args = parser.parse_args()
    
    try:
        print("=== CF-Specific Linear Regression Analysis ===\n")
        print("Analyzing BEZ vs Cochlea model agreement across characteristic frequencies...")
        print(f"Using dB index {args.db_idx} (corresponding dB level will be shown in results)")
        print(f"Color palette: {args.color_palette}\n")
        
        # Load data
        bez_data, cochlea_data = load_matlab_data()
        
        # Create output directory
        output_dir = "//subcorticalSTRF/cf_regression_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze each requested fiber type
        db_levels = ['50dB', '60dB', '70dB', '80dB']
        print(f"Analyzing at {db_levels[args.db_idx]}...")
        print(f"Each CF analyzed separately using responses to all tone frequencies at specified dB level")
        
        all_results = {}
        
        for fiber_type in args.fiber_types:
            print(f"\n=== Processing {fiber_type.upper()} Fibers ===")
            
            # Create CF-specific regressions
            fig, regression_df, db_level, palette_name = create_cf_specific_regressions(
                bez_data, cochlea_data, 
                fiber_type=fiber_type, 
                db_idx=args.db_idx, 
                color_palette=args.color_palette
            )
            
            all_results[fiber_type] = {
                'figure': fig,
                'regression_df': regression_df,
                'db_level': db_level,
                'palette_name': palette_name
            }
            
            # Save plots
            plot_file = os.path.join(output_dir, f'cf_regressions_{fiber_type}_{db_level}dB_{args.color_palette}.png')
            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            fig.savefig(plot_file.replace('.png', '.pdf'), bbox_inches='tight')
            print(f"CF regression plots saved: {plot_file}")
            
            # Save regression data
            csv_file = os.path.join(output_dir, f'cf_regression_data_{fiber_type}_{db_level}dB_{args.color_palette}.csv')
            regression_df.to_csv(csv_file, index=False)
            print(f"Regression data saved: {csv_file}")
            
            # Save detailed regression statistics with confidence intervals
            save_detailed_regression_statistics(regression_df, fiber_type, db_level, args.color_palette, 
                                               output_dir, bez_data, cochlea_data, args.db_idx)
            
            # Print summary statistics
            print(f"\nSummary for {fiber_type.upper()}:")
            print(f"  Mean R²: {regression_df['r_squared'].mean():.3f}")
            print(f"  Median R²: {regression_df['r_squared'].median():.3f}")
            print(f"  Mean slope: {regression_df['slope'].mean():.3f}")
            print(f"  Best CF (R²): {regression_df.loc[regression_df['r_squared'].idxmax(), 'cf']:.0f} Hz (R²={regression_df['r_squared'].max():.3f})")
            print(f"  Worst CF (R²): {regression_df.loc[regression_df['r_squared'].idxmin(), 'cf']:.0f} Hz (R²={regression_df['r_squared'].min():.3f})")
            
            plt.show(block=False)
        
        # Cross-fiber comparison
        print("\n=== Cross-Fiber Type Comparison ===")
        print("Fiber Type | Mean R²  | Median R² | Mean Slope | Best CF (Hz) | Best R² | Worst CF (Hz) | Worst R²")
        print("-" * 95)
        for fiber_type, results in all_results.items():
            df = results['regression_df']
            best_idx = df['r_squared'].idxmax()
            worst_idx = df['r_squared'].idxmin()
            
            print(f"{fiber_type.upper():10} | {df['r_squared'].mean():8.3f} | "
                  f"{df['r_squared'].median():9.3f} | {df['slope'].mean():10.3f} | "
                  f"{df.loc[best_idx, 'cf']:11.0f} | {df['r_squared'].max():7.3f} | "
                  f"{df.loc[worst_idx, 'cf']:12.0f} | {df['r_squared'].min():8.3f}")
        
        print("\n=== CF-Specific Regression Analysis Complete! ===")
        print("Linear regression analysis completed for all fiber types and characteristic frequencies.")
        
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