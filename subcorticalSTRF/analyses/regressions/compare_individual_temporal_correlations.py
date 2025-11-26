#!/usr/bin/env python3
"""
CF-specific linear regression: Compare BEZ vs Cochlea for each CF separately
Each CF gets its own subplot with linear regression across all tone frequencies at 60 dB
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
from subcorticalSTRF.compare_bez_cochlea_mainscript import load_matlab_data, convert_to_numeric, extract_parameters

# Set seaborn style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100


def create_cf_specific_regressions(bez_data, cochlea_data, fiber_type='lsr', db_idx=1, color_palette='Spectral'):
    """Create linear regression plots for each CF separately, using all tone frequencies as data points"""
    
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
    
    # Storage for regression statistics
    cf_regression_stats = []
    
    for cf_idx, cf_val in enumerate(bez_cfs):
        ax = axes[cf_idx]
        
        # Collect all temporal points for this CF across all frequencies
        all_bez_points = []
        all_cochlea_points = []
        all_frequencies = []
        
        print(f"Processing CF {cf_val:.0f} Hz...")
        
        for freq_idx, freq_val in enumerate(bez_frequencies):
            try:
                # Get PSTH time series for this CF-frequency combination
                bez_psth = bez_rates[fiber_type][cf_idx, freq_idx, db_idx]
                cochlea_psth = cochlea_rates[fiber_type][cf_idx, freq_idx, db_idx]
                
                # Clean data
                clean_mask = np.isfinite(bez_psth) & np.isfinite(cochlea_psth)
                if np.sum(clean_mask) > 0:
                    bez_clean = bez_psth[clean_mask]
                    cochlea_clean = cochlea_psth[clean_mask]
                    
                    # Add all temporal points from this frequency to our collection
                    all_bez_points.extend(bez_clean)
                    all_cochlea_points.extend(cochlea_clean)
                    all_frequencies.extend([freq_val] * len(bez_clean))
                    
                    # Plot points colored by frequency
                    ax.scatter(bez_clean, cochlea_clean, 
                             color=freq_color_map[freq_val], alpha=0.6, s=20,
                             label=f'{freq_val:.0f} Hz')
                
            except (IndexError, ValueError) as e:
                print(f"  Warning: Could not process Freq={freq_val}Hz: {e}")
                continue
        
        # Calculate linear regression for this CF using all frequency points
        if len(all_bez_points) > 2:
            slope, intercept, r_value, p_value, std_err = linregress(all_bez_points, all_cochlea_points)
            r_squared = r_value**2
            
            # Store statistics
            cf_regression_stats.append({
                'cf': cf_val,
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'r_squared': r_squared,
                'p_value': p_value,
                'std_err': std_err,
                'n_points': len(all_bez_points),
                'n_frequencies': len(set(all_frequencies))
            })
            
            # Plot regression line with 66% confidence bands
            x_min, x_max = min(all_bez_points), max(all_bez_points)
            x_range = np.linspace(x_min, x_max, 100)
            y_pred = slope * x_range + intercept
            
            # Calculate 66% confidence bands (1 STD)
            upper_band, lower_band = calculate_prediction_confidence_bands(
                all_bez_points, all_cochlea_points, slope, intercept, x_range, confidence=0.66)
            
            # Plot regression line
            ax.plot(x_range, y_pred, 'k-', linewidth=2, alpha=0.8,
                   label=f'R²={r_squared:.3f}, slope={slope:.3f}')
            
            # Plot confidence bands if available
            if upper_band is not None and lower_band is not None:
                ax.fill_between(x_range, lower_band, upper_band, 
                               color='gray', alpha=0.3, 
                               label='66% CI (±1σ)')
            
            # Add perfect match line
            ax.plot([x_min, x_max], [x_min, x_max], 'r--', alpha=0.5, linewidth=1,
                   label='Perfect match')
            
        # Formatting
        ax.set_title(f'CF = {cf_val:.0f} Hz\n'
                    f'R² = {r_squared:.3f}, slope = {slope:.3f}\n'
                    f'{len(set(all_frequencies))} frequencies, {len(all_bez_points)} points', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('BEZ Model Rate (spikes/s)')
        ax.set_ylabel('Cochlea Model Rate (spikes/s)')
        ax.grid(True, alpha=0.3)
        
        # Add legend for frequencies (show only a few to avoid clutter)
        unique_freqs = sorted(set(all_frequencies))
        if len(unique_freqs) <= 10:
            ax.legend(fontsize=8, loc='upper left')
        else:
            # Show only every few frequencies in legend
            step = max(1, len(unique_freqs) // 5)
            legend_freqs = unique_freqs[::step]
            legend_elements = []
            for freq in legend_freqs:
                legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=freq_color_map[freq], markersize=6,
                                                label=f'{freq:.0f} Hz'))
            legend_elements.append(Line2D([0], [0], color='k', linewidth=2,
                                            label=f'Regression'))
            ax.legend(handles=legend_elements, fontsize=8, loc='upper left')
    
    # Hide unused subplots
    for idx in range(n_cfs, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Convert regression stats to DataFrame
    regression_df = pd.DataFrame(cf_regression_stats)
    
    print(f"Successfully processed {len(regression_df)} CFs")
    print(f"Mean R² across CFs: {regression_df['r_squared'].mean():.3f}")
    print(f"Mean slope across CFs: {regression_df['slope'].mean():.3f}")
    
    return fig, regression_df, db_level, color_palette


def calculate_confidence_intervals(x, y, slope, intercept, confidence=0.95):
    """Calculate confidence intervals for regression parameters"""
    from scipy import stats
    
    n = len(x)
    if n < 3:
        return None, None, None, None
    
    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Calculate residuals
    y_pred = slope * x + intercept
    residuals = y - y_pred
    
    # Calculate standard errors
    mse = np.sum(residuals**2) / (n - 2)  # Mean squared error
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean)**2)
    
    # Standard error of slope and intercept
    se_slope = np.sqrt(mse / sxx)
    se_intercept = np.sqrt(mse * (1/n + x_mean**2/sxx))
    
    # T-critical value for confidence interval
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, n - 2)
    
    # Confidence intervals
    slope_ci = (slope - t_crit * se_slope, slope + t_crit * se_slope)
    intercept_ci = (intercept - t_crit * se_intercept, intercept + t_crit * se_intercept)
    
    return slope_ci, intercept_ci, se_slope, se_intercept


def calculate_prediction_confidence_bands(x_data, y_data, slope, intercept, x_range, confidence=0.66):
    """Calculate confidence bands for regression prediction lines"""
    from scipy import stats
    
    n = len(x_data)
    if n < 3:
        return None, None
    
    # Convert to numpy arrays
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_range = np.array(x_range)
    
    # Calculate residuals and MSE
    y_pred_data = slope * x_data + intercept
    residuals = y_data - y_pred_data
    mse = np.sum(residuals**2) / (n - 2)
    
    # Calculate statistics needed for confidence bands
    x_mean = np.mean(x_data)
    sxx = np.sum((x_data - x_mean)**2)
    
    # T-critical value (for 66% confidence interval, use 1 STD equivalent)
    # For 66% confidence interval in t-distribution
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, n - 2)
    
    # Prediction line
    y_pred_line = slope * x_range + intercept
    
    # Standard error of prediction for each x value in range
    se_pred = np.sqrt(mse * (1 + 1/n + (x_range - x_mean)**2 / sxx))
    
    # Confidence bands
    margin_error = t_crit * se_pred
    upper_band = y_pred_line + margin_error
    lower_band = y_pred_line - margin_error
    
    return upper_band, lower_band


def save_detailed_regression_statistics(regression_df, fiber_type, db_level, color_palette, output_dir, 
                                       bez_data, cochlea_data, db_idx):
    """Save comprehensive linear regression statistics with confidence intervals to detailed report"""
    
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
    
    def calculate_parameter_confidence_intervals(x, y, slope, intercept, confidence=0.66):
        """Calculate 66% confidence intervals for regression parameters"""
        from scipy import stats
        
        n = len(x)
        if n < 3:
            return None, None, None, None
        
        # Convert to numpy arrays
        x = np.array(x)
        y = np.array(y)
        
        # Calculate residuals
        y_pred = slope * x + intercept
        residuals = y - y_pred
        
        # Calculate standard errors
        mse = np.sum(residuals**2) / (n - 2)  # Mean squared error
        x_mean = np.mean(x)
        sxx = np.sum((x - x_mean)**2)
        
        # Standard error of slope and intercept
        se_slope = np.sqrt(mse / sxx)
        se_intercept = np.sqrt(mse * (1/n + x_mean**2/sxx))
        
        # T-critical value for confidence interval
        alpha = 1 - confidence
        t_crit = stats.t.ppf(1 - alpha/2, n - 2)
        
        # Confidence intervals
        slope_ci = (slope - t_crit * se_slope, slope + t_crit * se_slope)
        intercept_ci = (intercept - t_crit * se_intercept, intercept + t_crit * se_intercept)
        
        return slope_ci, intercept_ci, se_slope, se_intercept
    
    with open(stats_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write(f"COMPREHENSIVE LINEAR REGRESSION ANALYSIS REPORT\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Fiber Type: {fiber_type.upper()}\n")
        f.write(f"Sound Level: {db_level} dB SPL\n")
        f.write(f"Color Palette: {color_palette}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total CFs Analyzed: {len(regression_df)}\n")
        f.write(f"Confidence Level: 95%\n\n")
        
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
        f.write("DETAILED CF-BY-CF ANALYSIS WITH 95% CONFIDENCE INTERVALS\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("CF (Hz) |   R²   | Slope  | Slope 95% CI        | Intercept | Intercept 95% CI    | P-value  | SE Slope | SE Int   | N Points\n")
        f.write("-" * 120 + "\n")
        
        # Sort by R² for better organization
        sorted_df = regression_df.sort_values('r_squared', ascending=False).reset_index(drop=True)
        
        for idx, (_, row) in enumerate(sorted_df.iterrows()):
            cf_val = row['cf']
            cf_idx = list(bez_cfs).index(cf_val)
            
            # Recalculate detailed statistics with confidence intervals
            all_bez_points = []
            all_cochlea_points = []
            
            for freq_idx, freq_val in enumerate(bez_frequencies):
                try:
                    bez_psth = bez_rates[fiber_type][cf_idx, freq_idx, db_idx]
                    cochlea_psth = cochlea_rates[fiber_type][cf_idx, freq_idx, db_idx]
                    
                    clean_mask = np.isfinite(bez_psth) & np.isfinite(cochlea_psth)
                    if np.sum(clean_mask) > 0:
                        all_bez_points.extend(bez_psth[clean_mask])
                        all_cochlea_points.extend(cochlea_psth[clean_mask])
                except:
                    continue
            
            # Calculate confidence intervals
            slope_ci, intercept_ci, se_slope, se_intercept = calculate_confidence_intervals(
                all_bez_points, all_cochlea_points, row['slope'], row['intercept'])
            
            if slope_ci is not None:
                f.write(f"{cf_val:7.0f} | {row['r_squared']:6.4f} | {row['slope']:6.4f} | "
                       f"({slope_ci[0]:6.4f}, {slope_ci[1]:6.4f}) | {row['intercept']:9.4f} | "
                       f"({intercept_ci[0]:7.4f}, {intercept_ci[1]:7.4f}) | {row['p_value']:8.2e} | "
                       f"{se_slope:8.4f} | {se_intercept:8.4f} | {row['n_points']:8.0f}\n")
                
                # Store for JSON output
                detailed_stats[str(cf_val)] = {
                    'cf': float(cf_val),
                    'r_squared': float(row['r_squared']),
                    'slope': float(row['slope']),
                    'slope_ci_lower': float(slope_ci[0]),
                    'slope_ci_upper': float(slope_ci[1]),
                    'intercept': float(row['intercept']),
                    'intercept_ci_lower': float(intercept_ci[0]),
                    'intercept_ci_upper': float(intercept_ci[1]),
                    'p_value': float(row['p_value']),
                    'se_slope': float(se_slope),
                    'se_intercept': float(se_intercept),
                    'n_points': int(row['n_points']),
                    'n_frequencies': int(row['n_frequencies'])
                }
            else:
                f.write(f"{cf_val:7.0f} | {row['r_squared']:6.4f} | {row['slope']:6.4f} | "
                       f"{'N/A':>18} | {row['intercept']:9.4f} | {'N/A':>18} | "
                       f"{row['p_value']:8.2e} | {'N/A':>8} | {'N/A':>8} | {row['n_points']:8.0f}\n")
        
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
                'r_squared': {
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
    """Main function"""
    
    try:
        print("=== Individual Condition Temporal Correlation Analysis ===\n")
        
        # Load data
        bez_data, cochlea_data = load_matlab_data()
        
        # Create output directory
        output_dir = "//subcorticalSTRF/individual_temporal_correlations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze each fiber type
        fiber_types = ['lsr', 'msr', 'hsr']
        db_idx =  1 # 60 dB
        
        # Color palette selection - change this to compare different palettes
        # Popular options: 'Spectral', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'tab10', 'Set3', 'Dark2'
        color_palette = 'managua'  
        print(f"Using color palette: {color_palette}")
        print("To compare palettes, change 'color_palette' and rerun - files will be saved with palette name.")
        
        for fiber_type in fiber_types:
            print(f"\n=== Processing {fiber_type.upper()} Fibers ===")
            
            # Create CF-specific regressions
            fig, regression_df, db_level, palette_name = create_cf_specific_regressions(
                bez_data, cochlea_data, fiber_type=fiber_type, db_idx=db_idx, color_palette=color_palette)
            
            # Create output directory
            output_dir = "//subcorticalSTRF/cf_specific_regressions"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save plots
            plot_file = os.path.join(output_dir, f'cf_regressions_{fiber_type}_{db_level}dB_{color_palette}.png')
            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            fig.savefig(plot_file.replace('.png', '.pdf'), bbox_inches='tight')
            print(f"CF regression plots saved: {plot_file}")
            
            # Save regression data
            csv_file = os.path.join(output_dir, f'cf_regression_data_{fiber_type}_{db_level}dB_{color_palette}.csv')
            regression_df.to_csv(csv_file, index=False)
            print(f"Regression data saved: {csv_file}")
            
            # Save detailed regression statistics with confidence intervals
            save_detailed_regression_statistics(regression_df, fiber_type, db_level, color_palette, 
                                               output_dir, bez_data, cochlea_data, db_idx)
            
            # Print summary statistics
            print(f"Summary for {fiber_type.upper()}:")
            print(f"  Mean R²: {regression_df['r_squared'].mean():.3f}")
            print(f"  Median R²: {regression_df['r_squared'].median():.3f}")
            print(f"  Mean slope: {regression_df['slope'].mean():.3f}")
            print(f"  Best CF (R²): {regression_df.loc[regression_df['r_squared'].idxmax(), 'cf']:.0f} Hz (R²={regression_df['r_squared'].max():.3f})")
            print(f"  Worst CF (R²): {regression_df.loc[regression_df['r_squared'].idxmin(), 'cf']:.0f} Hz (R²={regression_df['r_squared'].min():.3f})")
            
            
            

            plt.show(block=False)
        
        print("\n=== Analysis Complete! ===")
        print("Individual condition temporal correlations calculated and visualized.")
        
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