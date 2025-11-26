#!/usr/bin/env python3
"""
CF-Level Regression Analysis and Fiber Type Comparison

This script calculates CF-specific regression slopes (matching the violin plots)
and performs statistical tests to compare fiber type differences in model agreement.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
from scipy.stats import linregress, f_oneway, ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Set style for publication-quality plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

def calculate_cf_specific_slopes():
    """Calculate CF-specific slopes exactly like the violin plot script"""
    
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
    
    print(f"Data dimensions:")
    print(f"  CFs: {len(cfs)} ({cfs.min():.0f} - {cfs.max():.0f} Hz)")
    print(f"  Frequencies: {len(frequencies)} ({frequencies.min():.0f} - {frequencies.max():.0f} Hz)")
    print(f"  dB levels: {len(dbs)} ({dbs.min():.0f} - {dbs.max():.0f} dB SPL)")
    
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
    
    # Calculate CF-specific slopes (EXACTLY like violin plot script)
    all_data = []
    
    print(f"\nCalculating CF-specific slopes...")
    
    for db_idx, db_val in enumerate(dbs):
        for fiber_type in ['LSR', 'MSR', 'HSR']:
            fiber_key = fiber_type.lower()
            print(f"  Processing {fiber_type} fibers at {db_val} dB...")
            
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
                
                # Perform regression for this CF (SAME AS VIOLIN PLOT)
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
                        'dB_Level': db_val,
                        'Slope': slope,
                        'R_squared': r_value**2,
                        'P_value': p_value,
                        'N_points': len(cf_bez_points),
                        'SE_Slope': se_slope,
                        'Log_CF': np.log10(cf_val),
                        'dB_Centered': db_val - np.mean(dbs),
                        'CF_Index': cf_idx,
                        'dB_Index': db_idx
                    })
    
    df = pd.DataFrame(all_data)
    
    print(f"\nCF-specific slopes calculated:")
    print(f"  Total CF-dB-Fiber combinations: {len(df)}")
    print(f"  Expected: {len(cfs)} CFs × {len(dbs)} dB × 3 fibers = {len(cfs) * len(dbs) * 3}")
    
    # Print fiber type distribution
    for fiber_type in ['HSR', 'MSR', 'LSR']:
        count = len(df[df['Fiber_Type'] == fiber_type])
        mean_slope = df[df['Fiber_Type'] == fiber_type]['Slope'].mean()
        print(f"  {fiber_type}: {count} observations, mean slope = {mean_slope:.4f}")
    
    return df

def test_fiber_type_differences(df, output_dir):
    """Test statistical differences between fiber types"""
    
    print(f"\n{'='*80}")
    print("STATISTICAL TESTS FOR FIBER TYPE DIFFERENCES")
    print(f"{'='*80}")
    
    results = {}
    
    # 1. One-Way ANOVA: Test if fiber types have different slopes
    print(f"\n1. ONE-WAY ANOVA: Do fiber types have different slopes?")
    print("-" * 60)
    
    hsr_slopes = df[df['Fiber_Type'] == 'HSR']['Slope'].values
    msr_slopes = df[df['Fiber_Type'] == 'MSR']['Slope'].values
    lsr_slopes = df[df['Fiber_Type'] == 'LSR']['Slope'].values
    
    f_stat, p_value = f_oneway(hsr_slopes, msr_slopes, lsr_slopes)
    
    print(f"F-statistic = {f_stat:.3f}")
    print(f"p-value = {p_value:.2e}")
    
    if p_value < 0.001:
        print("*** Highly significant differences between fiber types (p < 0.001)")
    elif p_value < 0.05:
        print("** Significant differences between fiber types (p < 0.05)")
    else:
        print("No significant differences between fiber types")
    
    results['anova'] = {'F': f_stat, 'p': p_value}
    
    # 2. Pairwise t-tests between fiber types
    print(f"\n2. PAIRWISE T-TESTS: Which fiber types differ?")
    print("-" * 50)
    
    comparisons = [('HSR', 'MSR'), ('HSR', 'LSR'), ('MSR', 'LSR')]
    pairwise_results = {}
    
    for fiber1, fiber2 in comparisons:
        slopes1 = df[df['Fiber_Type'] == fiber1]['Slope'].values
        slopes2 = df[df['Fiber_Type'] == fiber2]['Slope'].values
        
        t_stat, p_val = ttest_ind(slopes1, slopes2)
        
        mean1 = np.mean(slopes1)
        mean2 = np.mean(slopes2)
        diff = mean1 - mean2
        
        print(f"{fiber1} vs {fiber2}:")
        print(f"  Means: {mean1:.4f} vs {mean2:.4f} (diff = {diff:.4f})")
        print(f"  t-statistic = {t_stat:.3f}, p = {p_val:.2e}")
        
        if p_val < 0.001:
            print(f"  *** Highly significant difference (p < 0.001)")
        elif p_val < 0.05:
            print(f"  ** Significant difference (p < 0.05)")
        else:
            print(f"  No significant difference")
        print()
        
        pairwise_results[f'{fiber1}_vs_{fiber2}'] = {
            't': t_stat, 'p': p_val, 'mean_diff': diff
        }
    
    results['pairwise'] = pairwise_results
    
    # 3. Tukey HSD post-hoc test
    print(f"\n3. TUKEY HSD POST-HOC TEST: Multiple comparisons correction")
    print("-" * 65)
    
    tukey_result = pairwise_tukeyhsd(df['Slope'], df['Fiber_Type'])
    print(tukey_result)
    results['tukey'] = str(tukey_result)
    
    # 4. ANOVA with dB level as covariate
    print(f"\n4. ANCOVA: Fiber type differences controlling for dB level")
    print("-" * 60)
    
    ancova_model = smf.ols('Slope ~ C(Fiber_Type) + dB_Level', data=df).fit()
    print(f"Model R² = {ancova_model.rsquared:.4f}")
    print(f"F-statistic = {ancova_model.fvalue:.3f}, p = {ancova_model.f_pvalue:.2e}")
    
    print(f"\nFiber type effects (controlling for dB):")
    for param_name in ancova_model.params.index:
        if 'Fiber_Type' in param_name:
            coef = ancova_model.params[param_name]
            pval = ancova_model.pvalues[param_name]
            print(f"  {param_name}: {coef:.4f}, p = {pval:.3e}")
    
    results['ancova'] = ancova_model
    
    # 5. ANOVA with CF as covariate
    print(f"\n5. ANCOVA: Fiber type differences controlling for CF")
    print("-" * 55)
    
    ancova_cf_model = smf.ols('Slope ~ C(Fiber_Type) + Log_CF', data=df).fit()
    print(f"Model R² = {ancova_cf_model.rsquared:.4f}")
    print(f"F-statistic = {ancova_cf_model.fvalue:.3f}, p = {ancova_cf_model.f_pvalue:.2e}")
    
    print(f"\nFiber type effects (controlling for CF):")
    for param_name in ancova_cf_model.params.index:
        if 'Fiber_Type' in param_name:
            coef = ancova_cf_model.params[param_name]
            pval = ancova_cf_model.pvalues[param_name]
            print(f"  {param_name}: {coef:.4f}, p = {pval:.3e}")
    
    results['ancova_cf'] = ancova_cf_model
    
    # 6. Full model: Fiber type + dB + CF + interactions
    print(f"\n6. FULL MODEL: All factors and interactions")
    print("-" * 45)
    
    full_model = smf.ols('Slope ~ C(Fiber_Type) * dB_Level + Log_CF', data=df).fit()
    print(f"Model R² = {full_model.rsquared:.4f}")
    print(f"F-statistic = {full_model.fvalue:.3f}, p = {full_model.f_pvalue:.2e}")
    
    # Test significance of fiber type main effect
    reduced_model = smf.ols('Slope ~ dB_Level + Log_CF', data=df).fit()
    anova_comparison = anova_lm(reduced_model, full_model)
    print(f"\nFiber type effect test:")
    print(anova_comparison)
    
    results['full_model'] = full_model
    results['model_comparison'] = anova_comparison
    
    return results

def test_slopes_against_unity(df):
    """Test if each fiber type slope is significantly different from 1.0"""
    
    print(f"\n{'='*80}")
    print("TESTING SLOPES AGAINST PERFECT AGREEMENT (SLOPE = 1.0)")
    print(f"{'='*80}")
    
    results = {}
    
    for fiber_type in ['HSR', 'MSR', 'LSR']:
        print(f"\n{fiber_type} FIBER TYPE:")
        print("-" * 20)
        
        fiber_data = df[df['Fiber_Type'] == fiber_type]
        slopes = fiber_data['Slope'].values
        
        # One-sample t-test against 1.0
        from scipy.stats import ttest_1samp
        t_stat, p_value = ttest_1samp(slopes, 1.0)
        
        mean_slope = np.mean(slopes)
        se_slope = np.std(slopes) / np.sqrt(len(slopes))
        
        print(f"Mean slope = {mean_slope:.4f} ± {se_slope:.4f}")
        print(f"t-statistic = {t_stat:.3f}")
        print(f"p-value = {p_value:.2e}")
        
        if p_value < 0.001:
            direction = "greater" if mean_slope > 1.0 else "less"
            print(f"*** Highly significant difference from 1.0 (slope {direction} than 1.0)")
        elif p_value < 0.05:
            direction = "greater" if mean_slope > 1.0 else "less"
            print(f"** Significant difference from 1.0 (slope {direction} than 1.0)")
        else:
            print(f"No significant difference from 1.0")
        
        # Effect size (Cohen's d)
        cohens_d = (mean_slope - 1.0) / np.std(slopes)
        print(f"Effect size (Cohen's d) = {cohens_d:.3f}")
        
        if abs(cohens_d) > 0.8:
            print(f"*** Large effect size")
        elif abs(cohens_d) > 0.5:
            print(f"** Medium effect size")
        elif abs(cohens_d) > 0.2:
            print(f"* Small effect size")
        else:
            print(f"Negligible effect size")
        
        results[fiber_type] = {
            'mean': mean_slope,
            'se': se_slope,
            't': t_stat,
            'p': p_value,
            'cohens_d': cohens_d
        }
    
    return results

def create_statistical_plots(df, stats_results, output_dir):
    """Create plots showing statistical comparisons"""
    
    print(f"\n{'='*60}")
    print("CREATING STATISTICAL COMPARISON PLOTS")
    print(f"{'='*60}")
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Statistical Analysis of Fiber Type Differences in Model Agreement', 
                 fontsize=16, fontweight='bold')
    
    colors = sns.color_palette('Set2', 3)
    fiber_types = ['HSR', 'MSR', 'LSR']
    
    # Plot 1: Box plot with individual points
    ax1 = axes[0, 0]
    
    box_plot = ax1.boxplot([df[df['Fiber_Type'] == ft]['Slope'].values for ft in fiber_types],
                          labels=fiber_types, patch_artist=True)
    
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual points with jitter
    for i, fiber_type in enumerate(fiber_types):
        slopes = df[df['Fiber_Type'] == fiber_type]['Slope'].values
        y_jitter = np.random.normal(i+1, 0.04, len(slopes))
        ax1.scatter(y_jitter, slopes, alpha=0.3, s=8, color=colors[i])
    
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Perfect Agreement')
    ax1.set_ylabel('Regression Slope', fontweight='bold')
    ax1.set_title('Slope Distribution by Fiber Type')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Mean slopes with error bars
    ax2 = axes[0, 1]
    
    means = [df[df['Fiber_Type'] == ft]['Slope'].mean() for ft in fiber_types]
    sems = [df[df['Fiber_Type'] == ft]['Slope'].std() / 
            np.sqrt(len(df[df['Fiber_Type'] == ft])) for ft in fiber_types]
    
    bars = ax2.bar(fiber_types, means, yerr=sems, capsize=5, 
                   color=colors, alpha=0.7, edgecolor='black')
    
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Perfect Agreement')
    ax2.set_ylabel('Mean Regression Slope', fontweight='bold')
    ax2.set_title('Mean Slopes with Standard Error')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # Add significance stars if available
    if 'anova' in stats_results and stats_results['anova']['p'] < 0.05:
        ax2.text(0.5, max(means) + max(sems) + 0.05, '***', 
                ha='center', fontsize=16, fontweight='bold')
    
    # Plot 3: Effect sizes (Cohen's d from unity test)
    ax3 = axes[0, 2]
    
    if 'unity_test' in stats_results:
        cohens_d = [stats_results['unity_test'][ft]['cohens_d'] for ft in fiber_types]
        
        bars = ax3.bar(fiber_types, cohens_d, color=colors, alpha=0.7, edgecolor='black')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.axhline(y=0.2, color='orange', linestyle=':', alpha=0.7, label='Small effect')
        ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium effect')
        ax3.axhline(y=0.8, color='red', linestyle='-', alpha=0.7, label='Large effect')
        ax3.axhline(y=-0.2, color='orange', linestyle=':', alpha=0.7)
        ax3.axhline(y=-0.5, color='orange', linestyle='--', alpha=0.7)
        ax3.axhline(y=-0.8, color='red', linestyle='-', alpha=0.7)
        
        ax3.set_ylabel("Cohen's d (vs slope = 1.0)", fontweight='bold')
        ax3.set_title('Effect Sizes vs Perfect Agreement')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend(fontsize=9)
    
    # Plot 4: Slopes vs dB level
    ax4 = axes[1, 0]
    
    for i, fiber_type in enumerate(fiber_types):
        fiber_data = df[df['Fiber_Type'] == fiber_type]
        db_levels = sorted(fiber_data['dB_Level'].unique())
        
        mean_slopes = [fiber_data[fiber_data['dB_Level'] == db]['Slope'].mean() 
                      for db in db_levels]
        
        ax4.plot(db_levels, mean_slopes, 'o-', color=colors[i], 
                label=fiber_type, linewidth=2, markersize=6)
    
    ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax4.set_xlabel('dB Level', fontweight='bold')
    ax4.set_ylabel('Mean Regression Slope', fontweight='bold')
    ax4.set_title('Slope vs Stimulus Intensity')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Plot 5: Slopes vs CF
    ax5 = axes[1, 1]
    
    for i, fiber_type in enumerate(fiber_types):
        fiber_data = df[df['Fiber_Type'] == fiber_type]
        ax5.scatter(fiber_data['Log_CF'], fiber_data['Slope'], 
                   alpha=0.6, s=15, color=colors[i], label=fiber_type)
    
    ax5.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax5.set_xlabel('Log₁₀(CF) [Hz]', fontweight='bold')
    ax5.set_ylabel('Regression Slope', fontweight='bold')
    ax5.set_title('Slope vs Characteristic Frequency')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Plot 6: Statistical summary text
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create summary text
    summary_text = "STATISTICAL SUMMARY\n" + "="*25 + "\n\n"
    
    if 'anova' in stats_results:
        f_val = stats_results['anova']['F']
        p_val = stats_results['anova']['p']
        summary_text += f"One-way ANOVA:\n"
        summary_text += f"F({2}, {len(df)-3}) = {f_val:.3f}\n"
        summary_text += f"p = {p_val:.2e}\n"
        if p_val < 0.001:
            summary_text += "*** Highly significant\n\n"
        elif p_val < 0.05:
            summary_text += "** Significant\n\n"
        else:
            summary_text += "Not significant\n\n"
    
    if 'pairwise' in stats_results:
        summary_text += "Pairwise comparisons:\n"
        for comparison, result in stats_results['pairwise'].items():
            p_val = result['p']
            mean_diff = result['mean_diff']
            summary_text += f"{comparison.replace('_vs_', ' vs ')}: "
            summary_text += f"Δ = {mean_diff:.3f}, "
            if p_val < 0.001:
                summary_text += "p < 0.001 ***\n"
            elif p_val < 0.05:
                summary_text += f"p = {p_val:.3f} **\n"
            else:
                summary_text += f"p = {p_val:.3f}\n"
        summary_text += "\n"
    
    if 'unity_test' in stats_results:
        summary_text += "Tests vs slope = 1.0:\n"
        for fiber_type in fiber_types:
            result = stats_results['unity_test'][fiber_type]
            p_val = result['p']
            mean_slope = result['mean']
            summary_text += f"{fiber_type}: {mean_slope:.3f}, "
            if p_val < 0.001:
                summary_text += "p < 0.001 ***\n"
            elif p_val < 0.05:
                summary_text += f"p = {p_val:.3f} **\n"
            else:
                summary_text += f"p = {p_val:.3f}\n"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    png_path = Path(output_dir) / f'fiber_type_statistical_analysis_{timestamp}.png'
    pdf_path = Path(output_dir) / f'fiber_type_statistical_analysis_{timestamp}.pdf'
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"✓ Statistical analysis plots saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    
    return fig

def main():
    """Main function to run CF-level analysis and fiber type comparisons"""
    parser = argparse.ArgumentParser(description='CF-level regression analysis and fiber type comparison')
    parser.add_argument('--output_dir', type=str, default='cf_level_analysis_results', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CF-LEVEL REGRESSION ANALYSIS AND FIBER TYPE COMPARISON")
    print("=" * 80)
    
    try:
        # Create output directory
        Path(args.output_dir).mkdir(exist_ok=True)
        
        # Step 1: Calculate CF-specific slopes (like violin plots)
        df = calculate_cf_specific_slopes()
        
        # Step 2: Test fiber type differences
        fiber_stats = test_fiber_type_differences(df, args.output_dir)
        
        # Step 3: Test slopes against unity (perfect agreement)
        unity_stats = test_slopes_against_unity(df)
        
        # Combine all results
        all_stats = {**fiber_stats, 'unity_test': unity_stats}
        
        # Step 4: Create statistical plots
        fig = create_statistical_plots(df, all_stats, args.output_dir)
        
        # Step 5: Save detailed results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CF-level data
        csv_path = Path(args.output_dir) / f'cf_level_slopes_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ CF-level slope data saved: {csv_path}")
        
        # Save statistical summary
        with open(Path(args.output_dir) / f'statistical_summary_{timestamp}.txt', 'w') as f:
            f.write("CF-LEVEL REGRESSION ANALYSIS AND FIBER TYPE COMPARISON\n")
            f.write("="*60 + "\n\n")
            
            f.write("DATA SUMMARY:\n")
            f.write("-"*15 + "\n")
            f.write(f"Total observations: {len(df)}\n")
            for fiber_type in ['HSR', 'MSR', 'LSR']:
                fiber_data = df[df['Fiber_Type'] == fiber_type]
                f.write(f"{fiber_type}: {len(fiber_data)} observations, mean slope = {fiber_data['Slope'].mean():.4f}\n")
            f.write("\n")
            
            if 'anova' in all_stats:
                f.write("ONE-WAY ANOVA RESULTS:\n")
                f.write("-"*25 + "\n")
                f.write(f"F-statistic = {all_stats['anova']['F']:.3f}\n")
                f.write(f"p-value = {all_stats['anova']['p']:.2e}\n\n")
            
            if 'pairwise' in all_stats:
                f.write("PAIRWISE COMPARISONS:\n")
                f.write("-"*22 + "\n")
                for comparison, result in all_stats['pairwise'].items():
                    f.write(f"{comparison}: t = {result['t']:.3f}, p = {result['p']:.2e}, diff = {result['mean_diff']:.4f}\n")
                f.write("\n")
            
            if 'unity_test' in all_stats:
                f.write("TESTS AGAINST SLOPE = 1.0:\n")
                f.write("-"*27 + "\n")
                for fiber_type in ['HSR', 'MSR', 'LSR']:
                    result = all_stats['unity_test'][fiber_type]
                    f.write(f"{fiber_type}: mean = {result['mean']:.4f}, t = {result['t']:.3f}, p = {result['p']:.2e}\n")
        
        # Print final summary
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        
        print(f"\nKey Findings:")
        if 'anova' in all_stats:
            anova_p = all_stats['anova']['p']
            if anova_p < 0.001:
                print(f"  *** HIGHLY SIGNIFICANT fiber type differences (p = {anova_p:.2e})")
            elif anova_p < 0.05:
                print(f"  ** Significant fiber type differences (p = {anova_p:.3f})")
            else:
                print(f"  No significant fiber type differences (p = {anova_p:.3f})")
        
        for fiber_type in ['HSR', 'MSR', 'LSR']:
            fiber_data = df[df['Fiber_Type'] == fiber_type]
            mean_slope = fiber_data['Slope'].mean()
            if 'unity_test' in all_stats:
                unity_p = all_stats['unity_test'][fiber_type]['p']
                sig_text = "***" if unity_p < 0.001 else "**" if unity_p < 0.05 else "ns"
                print(f"  {fiber_type}: {mean_slope:.4f} (vs 1.0: {sig_text})")
        
        print(f"\nAll results saved to: {args.output_dir}/")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()