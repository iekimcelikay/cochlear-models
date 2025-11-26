#!/usr/bin/env python3
"""
Simple Horizontal Violin Plot with Overlaid Distributions

This script creates a clean, publication-ready horizontal violin plot showing 
the d        # Calculate summary statistics for console output
    stats_text = []
    for fiber_type in fiber_types:
        fiber_data = df[df['Fiber_Type'] == fiber_type]
        mean_slope = fiber_data['Slope'].mean()
        std_slope = fiber_data['Slope'].std()
        near_perfect = len(fiber_data[(fiber_data['Slope'] >= 0.9) & (fiber_data['Slope'] <= 1.1)])
        total = len(fiber_data)
        
        stats_text.append(f"{fiber_type}: μ={mean_slope:.3f}±{std_slope:.3f}, "
                         f"Near perfect: {near_perfect}/{total} ({near_perfect/total*100:.0f}%)")
    
    # Print statistics to console instead of adding to plot
    print(f"\\nDetailed Statistics by Fiber Type:")
    for stat_line in stats_text:
        print(f"  {stat_line}")
    
    plt.tight_layout()
    
    return figegression slopes for all fiber types with overlaid individual 
data points, properly color-coded by fiber category.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import linregress, t
import os
from pathlib import Path
import argparse

# Set style for publication-quality plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

def load_and_process_data(db_idx=1):
    """Load data and calculate CF-specific slopes for all fiber types"""
    
    # Load BEZ model data
    bez_dir = Path("//subcorticalSTRF/BEZ2018_meanrate/results/processed_data")
    bez_file = bez_dir / 'bez_acrossruns_psths.mat'
    bez_data = loadmat(str(bez_file), struct_as_record=False, squeeze_me=True)
    
    # Load cochlea model data
    cochlea_dir = Path("//subcorticalSTRF/cochlea_meanrate/out/condition_psths")
    cochlea_file = cochlea_dir / 'cochlea_psths.mat'
    cochlea_data = loadmat(str(cochlea_file), struct_as_record=False, squeeze_me=True)
    
    # Extract parameters
    def convert_to_numeric(arr):
        arr = np.atleast_1d(arr).flatten()
        if arr.dtype.kind in ['U', 'S']:
            return np.array([float(str(x)) for x in arr])
        else:
            return arr.astype(float)
    
    cfs = np.sort(convert_to_numeric(bez_data['cfs']))
    frequencies = np.sort(convert_to_numeric(bez_data['frequencies']))
    dbs = np.sort(convert_to_numeric(bez_data['dbs']))
    
    db_level = dbs[db_idx]
    print(f"Processing data at {db_level} dB SPL...")
    
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
    
    # Calculate slopes for each fiber type and CF
    all_data = []
    
    for fiber_type in ['LSR', 'MSR', 'HSR']:
        fiber_key = fiber_type.lower()
        print(f"  Processing {fiber_type} fibers...")
        
        for cf_idx, cf_val in enumerate(cfs):
            # Collect data points for this CF across all tone frequencies
            cf_bez_points = []
            cf_cochlea_points = []
            
            for freq_idx, freq_val in enumerate(frequencies):
                try:
                    bez_psth = bez_rates[fiber_key][cf_idx, freq_idx, db_idx]
                    cochlea_psth = cochlea_rates[fiber_key][cf_idx, freq_idx, db_idx]
                    
                    bez_psth = np.atleast_1d(bez_psth).flatten()
                    cochlea_psth = np.atleast_1d(cochlea_psth).flatten()
                    
                    clean_mask = np.isfinite(bez_psth) & np.isfinite(cochlea_psth)
                    
                    if np.sum(clean_mask) > 0:
                        cf_bez_points.extend(bez_psth[clean_mask])
                        cf_cochlea_points.extend(cochlea_psth[clean_mask])
                        
                except (IndexError, ValueError):
                    continue
            
            # Perform regression for this CF
            if len(cf_bez_points) >= 5:
                slope, intercept, r_value, p_value, std_err = linregress(cf_bez_points, cf_cochlea_points)
                
                # Calculate 68% confidence interval for slope
                n = len(cf_bez_points)
                x_data = np.array(cf_bez_points)
                y_data = np.array(cf_cochlea_points)
                
                # Calculate standard error of slope for 68% CI
                y_pred = slope * x_data + intercept
                residuals = y_data - y_pred
                mse = np.sum(residuals**2) / (n - 2)
                x_mean = np.mean(x_data)
                sxx = np.sum((x_data - x_mean)**2)
                se_slope = np.sqrt(mse / sxx)
                
                # 68% CI is approximately ±1 standard error (for large n)
                ci_68_lower = slope - se_slope
                ci_68_upper = slope + se_slope
                ci_68_width = ci_68_upper - ci_68_lower
                
                all_data.append({
                    'Fiber_Type': fiber_type,
                    'CF_Hz': cf_val,
                    'Slope': slope,
                    'R_squared': r_value**2,
                    'P_value': p_value,
                    'N_points': len(cf_bez_points),
                    'Std_Error': std_err,
                    'SE_Slope': se_slope,
                    'CI_68_Lower': ci_68_lower,
                    'CI_68_Upper': ci_68_upper,
                    'CI_68_Width': ci_68_width
                })
    
    return pd.DataFrame(all_data), db_level

def create_clean_violin_plot(df, db_level, output_dir="violin_plot_results", 
                           color_palette='Set2', figsize=(12, 6)):
    """Create a clean horizontal violin plot with properly integrated scatter points"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Get colors for each fiber type
    fiber_types = ['HSR', 'MSR', 'LSR']  # Reordered for better visual flow
    colors = sns.color_palette(color_palette, len(fiber_types))
    color_map = dict(zip(fiber_types, colors))
    
    # Create horizontal violin plot with seaborn
    sns.violinplot(data=df, x='Slope', y='Fiber_Type', order=fiber_types,
                   hue='Fiber_Type', hue_order=fiber_types, palette=color_palette, 
                   inner='box', cut=0, ax=ax, legend=False, alpha=0.7)
    
    # Add individual CF data points with proper positioning
    for i, fiber_type in enumerate(fiber_types):
        fiber_data = df[df['Fiber_Type'] == fiber_type]
        
        # Create jitter that's constrained and looks natural
        base_y = i
        y_jitter = np.random.normal(0, 0.06, size=len(fiber_data))  # Small, controlled jitter
        y_positions = base_y + y_jitter
        
        # Plot individual CF points
        ax.scatter(fiber_data['Slope'], y_positions,
                  alpha=0.8, s=35, color=color_map[fiber_type],
                  edgecolors='white', linewidth=1, zorder=10)
    
    # Add reference lines
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.8, linewidth=1.5, 
               label='Perfect Agreement\\n(slope = 1.0)', zorder=5)
    ax.axvline(x=0.9, color='orange', linestyle=':', alpha=0.7, linewidth=1.2, 
               label='±10% Agreement\\n(slope = 0.9, 1.1)', zorder=5)
    ax.axvline(x=1.1, color='orange', linestyle=':', alpha=0.7, linewidth=1.2, zorder=5)
    
    # Customize plot
    ax.set_xlabel('Regression Slope (Cochlea vs BEZ Model)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Auditory Nerve Fiber Type', fontsize=14, fontweight='bold')
    ax.set_title(f'Distribution of Model Agreement by Fiber Type ({db_level} dB SPL)\\n'
                 f'Each point represents one characteristic frequency', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # Center x-axis around 1.0
    all_slopes = df['Slope'].values
    x_range = all_slopes.max() - all_slopes.min()
    buffer = max(0.2, x_range * 0.15)
    x_center = 1.0
    ax.set_xlim(x_center - x_range/2 - buffer, x_center + x_range/2 + buffer)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='upper left', fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f'clean_violin_plot_{db_level}dB_{timestamp}'
    
    png_path = Path(output_dir) / f'{filename_base}.png'
    pdf_path = Path(output_dir) / f'{filename_base}.pdf'
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"\\n✓ Clean violin plot saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    
    return fig

def create_slope_precision_plot(df, db_level, output_dir="violin_plot_results", 
                               color_palette='Set2', figsize=(12, 6)):
    """Create a strip plot showing the standard errors (precision) of regression slopes"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Get colors for each fiber type
    fiber_types = ['HSR', 'MSR', 'LSR']
    colors = sns.color_palette(color_palette, len(fiber_types))
    
    # Create horizontal strip plot showing individual standard errors
    sns.stripplot(data=df, y='Fiber_Type', x='SE_Slope', order=fiber_types,
                 palette=color_palette, size=8, alpha=0.8, ax=ax, jitter=True)
    
    # Add reference lines for precision levels
    ax.axvline(x=0.005, color='green', linestyle='--', alpha=0.8, linewidth=1.5, 
               label='Very High Precision (SE = 0.005)')
    ax.axvline(x=0.01, color='orange', linestyle=':', alpha=0.7, linewidth=1.2, 
               label='High Precision (SE = 0.01)')
    ax.axvline(x=0.015, color='red', linestyle='-.', alpha=0.7, linewidth=1.2, 
               label='Moderate Precision (SE = 0.015)')
    
    # Customize plot
    ax.set_xlabel('Standard Error of Regression Slope', fontsize=14, fontweight='bold')
    ax.set_ylabel('Auditory Nerve Fiber Type', fontsize=14, fontweight='bold')
    ax.set_title(f'Precision of Regression Slope Estimates by Fiber Type ({db_level} dB SPL)\\n'
                 f'Each point represents one characteristic frequency', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # Set x-axis limits to focus on the data range
    all_se = df['SE_Slope'].values
    x_max = max(0.02, all_se.max() * 1.1)
    ax.set_xlim(0, x_max)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f'slope_precision_stripplot_{db_level}dB_{timestamp}'
    
    png_path = Path(output_dir) / f'{filename_base}.png'
    pdf_path = Path(output_dir) / f'{filename_base}.pdf'
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"\\n✓ Slope precision strip plot saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    
    # Print detailed precision statistics
    print(f"\\nSlope Precision Statistics by Fiber Type:")
    for fiber_type in fiber_types:
        fiber_data = df[df['Fiber_Type'] == fiber_type]
        if len(fiber_data) > 0:
            mean_se = fiber_data['SE_Slope'].mean()
            std_se = fiber_data['SE_Slope'].std()
            median_se = fiber_data['SE_Slope'].median()
            min_se = fiber_data['SE_Slope'].min()
            max_se = fiber_data['SE_Slope'].max()
            very_high_precision = len(fiber_data[fiber_data['SE_Slope'] <= 0.005])
            high_precision = len(fiber_data[(fiber_data['SE_Slope'] > 0.005) & (fiber_data['SE_Slope'] <= 0.01)])
            moderate_precision = len(fiber_data[(fiber_data['SE_Slope'] > 0.01) & (fiber_data['SE_Slope'] <= 0.015)])
            
            print(f"  {fiber_type}: Mean SE = {mean_se:.4f} ± {std_se:.4f}, Median = {median_se:.4f}")
            print(f"      Range: [{min_se:.4f}, {max_se:.4f}]")
            print(f"      Very high precision (SE ≤ 0.005): {very_high_precision}/{len(fiber_data)} ({very_high_precision/len(fiber_data)*100:.1f}%)")
            print(f"      High precision (0.005 < SE ≤ 0.01): {high_precision}/{len(fiber_data)} ({high_precision/len(fiber_data)*100:.1f}%)")
            print(f"      Moderate precision (0.01 < SE ≤ 0.015): {moderate_precision}/{len(fiber_data)} ({moderate_precision/len(fiber_data)*100:.1f}%)")
    
    return fig

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create clean violin plot of regression slopes')
    parser.add_argument('--db_idx', type=int, default=1, choices=[0, 1, 2, 3],
                       help='dB level index: 0=50dB, 1=60dB, 2=70dB, 3=80dB (default: 1)')
    parser.add_argument('--color_palette', type=str, default='Set2',
                       help='Seaborn color palette (default: Set2)')
    parser.add_argument('--figsize', nargs=2, type=float, default=[12, 6],
                       help='Figure size as width height (default: 12 6)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CLEAN VIOLIN PLOT: Regression Slopes by Fiber Type")
    print("="*80)
    
    try:
        # Load and process data
        df, db_level = load_and_process_data(db_idx=args.db_idx)
        
        print(f"\\nData Summary:")
        print(f"  Total CFs analyzed: {len(df)}")
        print(f"  Fiber types: {', '.join(df['Fiber_Type'].unique())}")
        print(f"  dB level: {db_level} dB SPL")
        
        # Create the slope distribution plot
        fig1 = create_clean_violin_plot(df, db_level, 
                                        color_palette=args.color_palette,
                                        figsize=tuple(args.figsize))
        
        # Create the precision analysis plot
        fig2 = create_slope_precision_plot(df, db_level,
                                          color_palette=args.color_palette,
                                          figsize=tuple(args.figsize))
        
        # Display summary
        print(f"\\nSummary by Fiber Type:")
        for fiber_type in ['HSR', 'MSR', 'LSR']:
            fiber_data = df[df['Fiber_Type'] == fiber_type]
            if len(fiber_data) > 0:
                mean_slope = fiber_data['Slope'].mean()
                std_slope = fiber_data['Slope'].std()
                near_perfect = len(fiber_data[(fiber_data['Slope'] >= 0.9) & (fiber_data['Slope'] <= 1.1)])
                print(f"  {fiber_type}: {mean_slope:.3f} ± {std_slope:.3f} "
                      f"(near perfect: {near_perfect}/{len(fiber_data)}, {near_perfect/len(fiber_data)*100:.1f}%)")
        
        print(f"\\n{'='*80}")
        print("CLEAN VIOLIN PLOT COMPLETE!")
        print(f"{'='*80}")
        
        plt.show()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()