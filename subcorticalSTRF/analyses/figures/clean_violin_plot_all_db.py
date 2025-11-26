#!/usr/bin/env python3
"""
Clean Violin Plot with All dB Levels

This script creates publication-ready horizontal violin plots showing 
regression slopes for all fiber types across all dB levels, with visual
differentiation by brightness and size.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
from scipy.stats import linregress, t
import os
from pathlib import Path
import argparse

# Set style for publication-quality plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

def load_and_process_data(db_idx=None):
    """Load data and calculate CF-specific slopes for all fiber types
    
    Args:
        db_idx: If provided, process only that dB level. If None, process all dB levels.
    """
    
    # Load BEZ model data
    bez_dir = Path("//subcorticalSTRF/BEZ2018_meanrate/results/processed_data")
    bez_file = bez_dir / 'bez_acrossruns_psths.mat'
    bez_data = scipy.io.loadmat(str(bez_file), struct_as_record=False, squeeze_me=True)

    # Load cochlea model data
    cochlea_dir = Path("//subcorticalSTRF/cochlea_meanrate/out/condition_psths")
    cochlea_file = cochlea_dir / 'cochlea_psths.mat'
    cochlea_data = scipy.io.loadmat(str(cochlea_file), struct_as_record=False, squeeze_me=True)
    
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
    
    # Determine which dB levels to process
    if db_idx is not None:
        if db_idx < 0 or db_idx >= len(dbs):
            raise ValueError(f"db_idx must be between 0 and {len(dbs)-1}")
        db_levels_to_process = [dbs[db_idx]]
        db_indices_to_process = [db_idx]
        print(f"Processing data at {dbs[db_idx]} dB SPL...")
    else:
        db_levels_to_process = dbs
        db_indices_to_process = list(range(len(dbs)))
        print(f"Processing data at all dB levels: {dbs} dB SPL...")
    
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
    
    # Calculate slopes for each fiber type, CF, and dB level
    all_data = []
    
    for db_process_idx, (db_level, db_idx_actual) in enumerate(zip(db_levels_to_process, db_indices_to_process)):
        for fiber_type in ['LSR', 'MSR', 'HSR']:
            fiber_key = fiber_type.lower()
            if len(db_levels_to_process) == 1:
                print(f"  Processing {fiber_type} fibers...")
            
            for cf_idx, cf_val in enumerate(cfs):
                # Collect data points for this CF across all tone frequencies
                cf_bez_points = []
                cf_cochlea_points = []
                
                for freq_idx, freq_val in enumerate(frequencies):
                    try:
                        bez_psth = bez_rates[fiber_key][cf_idx, freq_idx, db_idx_actual]
                        cochlea_psth = cochlea_rates[fiber_key][cf_idx, freq_idx, db_idx_actual]
                        
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
                    
                    # Calculate standard error of slope for confidence intervals
                    n = len(cf_bez_points)
                    x_data = np.array(cf_bez_points)
                    y_data = np.array(cf_cochlea_points)
                    
                    # Calculate standard error of slope
                    y_pred = slope * x_data + intercept
                    residuals = y_data - y_pred
                    mse = np.sum(residuals**2) / (n - 2)
                    x_mean = np.mean(x_data)
                    sxx = np.sum((x_data - x_mean)**2)
                    se_slope = np.sqrt(mse / sxx)
                    
                    all_data.append({
                        'Fiber_Type': fiber_type,
                        'CF_Hz': cf_val,
                        'Slope': slope,
                        'R_squared': r_value**2,
                        'P_value': p_value,
                        'N_points': len(cf_bez_points),
                        'SE_Slope': se_slope,
                        'dB_Level': db_level
                    })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    print(f"\nData Summary:")
    print(f"  Total CFs analyzed: {len(df)}")
    print(f"  Fiber types: {', '.join(df['Fiber_Type'].unique())}")
    print(f"  dB levels: {', '.join(map(str, sorted(df['dB_Level'].unique())))} dB SPL")
    
    return df, db_levels_to_process

def create_clean_violin_plot(df, db_levels, output_dir="violin_plot_results", 
                            color_palette='Set2', figsize=(14, 8)):
    """Create a clean violin plot of regression slopes by fiber type with dB level differentiation"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Get unique fiber types and dB levels
    fiber_types = ['HSR', 'MSR', 'LSR']
    unique_db_levels = sorted(df['dB_Level'].unique())
    colors = sns.color_palette(color_palette, len(fiber_types))
    
    # Create brightness mapping for dB levels (higher dB = brighter)
    brightness_factors = np.linspace(0.4, 1.0, len(unique_db_levels))
    size_factors = np.linspace(0.6, 1.4, len(unique_db_levels))  # Size variation
    
    if len(unique_db_levels) == 1:
        # Single dB level - original behavior
        violin_parts = ax.violinplot([df[df['Fiber_Type'] == ft]['Slope'].values for ft in fiber_types],
                                    positions=range(len(fiber_types)), 
                                    vert=False, 
                                    widths=0.7,
                                    showmeans=False, 
                                    showmedians=True,
                                    showextrema=False)
        
        # Color the violins
        for i, (part, color) in enumerate(zip(violin_parts['bodies'], colors)):
            part.set_facecolor(color)
            part.set_alpha(0.8)
            part.set_edgecolor('black')
            part.set_linewidth(1.2)
        
        # Color the median lines
        violin_parts['cmedians'].set_colors('black')
        violin_parts['cmedians'].set_linewidth(2.5)
        
        title_suffix = f"({unique_db_levels[0]} dB SPL)"
    else:
        # Multiple dB levels - create layered violins
        for db_idx, db_level in enumerate(unique_db_levels):
            db_data = df[df['dB_Level'] == db_level]
            
            # Adjust violin width and position based on dB level
            width = 0.15 * size_factors[db_idx]
            offset = (db_idx - len(unique_db_levels)/2 + 0.5) * 0.12
            
            violin_parts = ax.violinplot([db_data[db_data['Fiber_Type'] == ft]['Slope'].values for ft in fiber_types],
                                        positions=[i + offset for i in range(len(fiber_types))], 
                                        vert=False, 
                                        widths=width,
                                        showmeans=False, 
                                        showmedians=False,
                                        showextrema=False)
            
            # Color the violins with brightness variation
            for i, part in enumerate(violin_parts['bodies']):
                color = colors[i]
                # Adjust brightness
                bright_color = tuple(c * brightness_factors[db_idx] + (1 - brightness_factors[db_idx]) for c in color[:3])
                part.set_facecolor(bright_color)
                part.set_alpha(0.7)
                part.set_edgecolor('black')
                part.set_linewidth(0.8)
        
        title_suffix = f"(50-80 dB SPL: {len(unique_db_levels)} intensity levels)"
    
    # Add individual data points with brightness and size differentiation
    np.random.seed(42)  # For reproducible jitter
    
    for i, fiber_type in enumerate(fiber_types):
        for db_idx, db_level in enumerate(unique_db_levels):
            fiber_db_data = df[(df['Fiber_Type'] == fiber_type) & (df['dB_Level'] == db_level)]
            
            if len(fiber_db_data) > 0:
                slopes = fiber_db_data['Slope'].values
                
                # Create jittered y positions
                base_y = i
                if len(unique_db_levels) > 1:
                    # Spread points across dB levels
                    offset = (db_idx - len(unique_db_levels)/2 + 0.5) * 0.08
                    jitter_range = 0.04
                else:
                    offset = 0
                    jitter_range = 0.08
                
                y_positions = np.random.normal(base_y + offset, jitter_range, len(slopes))
                
                # Color and size based on dB level
                color = colors[i]
                bright_color = tuple(c * brightness_factors[db_idx] + (1 - brightness_factors[db_idx]) for c in color[:3])
                point_size = 25 + 20 * size_factors[db_idx]
                
                # Plot points
                scatter = ax.scatter(slopes, y_positions, 
                          color=bright_color, alpha=0.8, s=point_size, 
                          edgecolors='white', linewidth=1.0, zorder=3,
                          label=f'{db_level} dB' if i == 0 else "")
    
    # Add reference line at slope = 1.0
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.8, linewidth=2.5, label='Perfect Linearity (Slope = 1.0)')
    
    # Customize the plot
    ax.set_yticks(range(len(fiber_types)))
    ax.set_yticklabels(fiber_types, fontsize=13, fontweight='bold')
    ax.set_xlabel('Regression Slope', fontsize=14, fontweight='bold')
    ax.set_ylabel('Auditory Nerve Fiber Type', fontsize=14, fontweight='bold')
    ax.set_title(f'Model Agreement by Fiber Type {title_suffix}\n'
                 f'Brightness and size represent intensity levels (brighter/larger = higher dB)', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # Set x-axis limits centered around 1.0
    all_slopes = df['Slope'].values
    x_min = min(0.4, all_slopes.min() - 0.1)
    x_max = max(1.4, all_slopes.max() + 0.1)
    ax.set_xlim(x_min, x_max)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3, axis='x')
    
    # Create legend
    legend_elements = [plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.8, linewidth=2.5, label='Perfect Linearity (Slope = 1.0)')]
    if len(unique_db_levels) > 1:
        legend_elements.extend([plt.scatter([], [], color='gray', s=25 + 20 * size_factors[db_idx], 
                                          alpha=0.6 + 0.4 * brightness_factors[db_idx],
                                          label=f'{db_level} dB SPL') 
                              for db_idx, db_level in enumerate(unique_db_levels)])
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    if len(unique_db_levels) == 1:
        filename_base = f'clean_violin_plot_{unique_db_levels[0]}dB_{timestamp}'
    else:
        filename_base = f'clean_violin_plot_all_dB_{timestamp}'
    
    png_path = Path(output_dir) / f'{filename_base}.png'
    pdf_path = Path(output_dir) / f'{filename_base}.pdf'
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"\n✓ Clean violin plot saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    
    return fig

def create_slope_precision_plot(df, db_levels, output_dir="violin_plot_results", 
                               color_palette='Set2', figsize=(14, 6)):
    """Create a strip plot showing the standard errors (precision) of regression slopes with dB level differentiation"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Get colors for each fiber type and dB levels
    fiber_types = ['HSR', 'MSR', 'LSR']
    unique_db_levels = sorted(df['dB_Level'].unique())
    colors = sns.color_palette(color_palette, len(fiber_types))
    
    # Create brightness and size mapping for dB levels
    brightness_factors = np.linspace(0.4, 1.0, len(unique_db_levels))
    size_factors = np.linspace(0.6, 1.4, len(unique_db_levels))
    
    # Create horizontal strip plot with dB level differentiation
    np.random.seed(42)  # For reproducible jitter
    for i, fiber_type in enumerate(fiber_types):
        for db_idx, db_level in enumerate(unique_db_levels):
            fiber_db_data = df[(df['Fiber_Type'] == fiber_type) & (df['dB_Level'] == db_level)]
            
            if len(fiber_db_data) > 0:
                se_values = fiber_db_data['SE_Slope'].values
                
                # Create jittered y positions
                base_y = i
                if len(unique_db_levels) > 1:
                    # Spread points across dB levels
                    offset = (db_idx - len(unique_db_levels)/2 + 0.5) * 0.08
                    jitter_range = 0.04
                else:
                    offset = 0
                    jitter_range = 0.08
                
                y_positions = np.random.normal(base_y + offset, jitter_range, len(se_values))
                
                # Color and size based on dB level
                color = colors[i]
                bright_color = tuple(c * brightness_factors[db_idx] + (1 - brightness_factors[db_idx]) for c in color[:3])
                point_size = 25 + 30 * size_factors[db_idx]
                
                # Plot points
                ax.scatter(se_values, y_positions, 
                          color=bright_color, alpha=0.8, s=point_size, 
                          edgecolors='white', linewidth=1.0, zorder=3)
    
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
    
    if len(unique_db_levels) == 1:
        title_suffix = f"({unique_db_levels[0]} dB SPL)"
    else:
        title_suffix = f"(50-80 dB SPL: {len(unique_db_levels)} intensity levels)"
    
    ax.set_title(f'Regression Slope Precision by Fiber Type {title_suffix}\n'
                 f'Brightness and size represent intensity levels (brighter/larger = higher dB)', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # Set fiber type labels
    ax.set_yticks(range(len(fiber_types)))
    ax.set_yticklabels(fiber_types, fontsize=13, fontweight='bold')
    
    # Set x-axis limits to focus on the data range
    all_se = df['SE_Slope'].values
    x_max = max(0.02, all_se.max() * 1.1)
    ax.set_xlim(0, x_max)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3, axis='x')
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], color='green', linestyle='--', alpha=0.8, linewidth=1.5, label='Very High Precision (SE = 0.005)'),
        plt.Line2D([0], [0], color='orange', linestyle=':', alpha=0.7, linewidth=1.2, label='High Precision (SE = 0.01)'),
        plt.Line2D([0], [0], color='red', linestyle='-.', alpha=0.7, linewidth=1.2, label='Moderate Precision (SE = 0.015)')
    ]
    
    if len(unique_db_levels) > 1:
        legend_elements.extend([plt.scatter([], [], color='gray', s=25 + 30 * size_factors[db_idx], 
                                          alpha=0.6 + 0.4 * brightness_factors[db_idx],
                                          label=f'{db_level} dB SPL') 
                              for db_idx, db_level in enumerate(unique_db_levels)])
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    if len(unique_db_levels) == 1:
        filename_base = f'slope_precision_stripplot_{unique_db_levels[0]}dB_{timestamp}'
    else:
        filename_base = f'slope_precision_stripplot_all_dB_{timestamp}'
    
    png_path = Path(output_dir) / f'{filename_base}.png'
    pdf_path = Path(output_dir) / f'{filename_base}.pdf'
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"\n✓ Slope precision strip plot saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    
    # Print detailed precision statistics
    print(f"\nSlope Precision Statistics by Fiber Type:")
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
    """Main function to create clean violin plots"""
    parser = argparse.ArgumentParser(description='Create clean violin plots of regression slopes')
    parser.add_argument('--db_idx', type=int, default=None, 
                       help='dB level index (0=50dB, 1=60dB, 2=70dB, 3=80dB). If not provided, all intensity levels will be processed.')
    parser.add_argument('--color_palette', type=str, default='Set2', 
                       help='Seaborn color palette name')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CLEAN VIOLIN PLOT: Regression Slopes by Fiber Type")
    print("=" * 80)
    
    try:
        # Load and process data
        df, db_levels = load_and_process_data(args.db_idx)
        
        # Create violin plot
        fig1 = create_clean_violin_plot(df, db_levels, color_palette=args.color_palette)
        
        # Create precision plot
        fig2 = create_slope_precision_plot(df, db_levels, color_palette=args.color_palette)
        
        # Display summary
        print(f"\nSummary by Fiber Type:")
        for fiber_type in ['HSR', 'MSR', 'LSR']:
            fiber_data = df[df['Fiber_Type'] == fiber_type]
            if len(fiber_data) > 0:
                mean_slope = fiber_data['Slope'].mean()
                std_slope = fiber_data['Slope'].std()
                near_perfect = len(fiber_data[(fiber_data['Slope'] >= 0.9) & (fiber_data['Slope'] <= 1.1)])
                print(f"  {fiber_type}: {mean_slope:.3f} ± {std_slope:.3f} "
                      f"(near perfect: {near_perfect}/{len(fiber_data)}, {near_perfect/len(fiber_data)*100:.1f}%)")
        
        print(f"\n{'=' * 80}")
        print("CLEAN VIOLIN PLOT COMPLETE!")
        print(f"{'=' * 80}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()