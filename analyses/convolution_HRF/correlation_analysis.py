#!/usr/bin/env python3
"""
Correlation analysis between Cochlea and BEZ models.

This script:
1. Calculates pairwise correlations within BEZ runs (within-model)
2. Compares Cochlea output to mean BEZ response (between-model)
3. Evaluates where Cochlea-BEZ correlation falls in within-BEZ 
   distribution

Usage:
    python correlation_analysis.py --bez-file <path> --cochlea-file <path>
"""

import pickle
import numpy as np
import argparse
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import logging


def load_extracted_data(filepath):
    """
    Load extracted data from pickle file.
    
    Parameters:
    - filepath: Path to pickle file
    
    Returns:
    - extracted_data: Nested dict [fiber_type][cf][freq][run] or [cf][freq]
    - metadata: Additional metadata from file
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data['extracted_data'], data


def calculate_within_bez_correlations(bez_data, fiber_type):
    """
    Calculate pairwise correlations between all BEZ runs.
    
    Parameters:
    - bez_data: BEZ data dict [cf][freq][run]
    - fiber_type: Fiber type being processed
    
    Returns:
    - correlations: Dict [cf][freq] = list of correlation coefficients
    - run_pairs: Dict [cf][freq] = list of (run_i, run_j) tuples
    """
    correlations = {}
    run_pairs = {}
    
    logging.info(f"\nDEBUG: Calculating correlations for {fiber_type}")
    logging.info(f"Number of CFs: {len(bez_data)}")
    
    for cf_val in bez_data:
        correlations[cf_val] = {}
        run_pairs[cf_val] = {}
        
        for freq_val in bez_data[cf_val]:
            runs = bez_data[cf_val][freq_val]
            run_indices = sorted(runs.keys())
            n_runs = len(run_indices)
            
            logging.info(
                f"CF={cf_val}Hz, Freq={freq_val}Hz: {n_runs} runs"
            )
            
            corr_list = []
            pairs_list = []
            
            # Calculate all pairwise correlations
            for i in range(n_runs):
                for j in range(i + 1, n_runs):
                    run_i = runs[run_indices[i]]
                    run_j = runs[run_indices[j]]
                    
                    # DEBUG: Check if arrays are identical
                    if i == 0 and j == 1:  # Check first pair
                        logging.info(
                            f"  First pair (run {run_indices[i]} vs "
                            f"{run_indices[j]}):"
                        )
                        logging.info(f"    Run i shape: {run_i.shape}")
                        logging.info(f"    Run j shape: {run_j.shape}")
                        logging.info(
                            f"    First 5 values i: {run_i[:5]}"
                        )
                        logging.info(
                            f"    First 5 values j: {run_j[:5]}"
                        )
                        logging.info(
                            f"    Arrays equal? "
                            f"{np.array_equal(run_i, run_j)}"
                        )
                    
                    # Ensure same length
                    if len(run_i) == len(run_j):
                        r, _ = stats.pearsonr(run_i, run_j)
                        corr_list.append(r)
                        pairs_list.append(
                            (run_indices[i], run_indices[j])
                        )
            
            correlations[cf_val][freq_val] = np.array(corr_list)
            run_pairs[cf_val][freq_val] = pairs_list
            
            # DEBUG: Check correlation values
            if len(corr_list) > 0:
                logging.info(
                    f"  Correlations: min={np.min(corr_list):.3f}, "
                    f"max={np.max(corr_list):.3f}, "
                    f"mean={np.mean(corr_list):.3f}"
                )
    
    return correlations, run_pairs


def calculate_correlation_matrix(bez_data, cf_val, freq_val):
    """
    Calculate full correlation matrix for all runs at a specific condition.
    
    Parameters:
    - bez_data: BEZ data dict [cf][freq][run]
    - cf_val: CF value
    - freq_val: Frequency value
    
    Returns:
    - corr_matrix: NxN correlation matrix (N = number of runs)
    - run_indices: List of run indices
    """
    runs = bez_data[cf_val][freq_val]
    run_indices = sorted(runs.keys())
    n_runs = len(run_indices)
    
    # Initialize correlation matrix
    corr_matrix = np.ones((n_runs, n_runs))
    
    # Calculate pairwise correlations
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            run_i = runs[run_indices[i]]
            run_j = runs[run_indices[j]]
            
            if len(run_i) == len(run_j):
                r, _ = stats.pearsonr(run_i, run_j)
                corr_matrix[i, j] = r
                corr_matrix[j, i] = r  # Symmetric
    
    return corr_matrix, run_indices


def plot_within_bez_correlation_heatmaps(
    bez_data,
    fiber_type,
    output_dir,
    max_freqs_per_cf=6
):
    """
    Create and save correlation heatmaps for within-BEZ run correlations.
    One main plot per CF with frequency subplots.
    
    Parameters:
    - bez_data: BEZ data dict [cf][freq][run]
    - fiber_type: Fiber type being analyzed
    - output_dir: Directory to save plots
    - max_freqs_per_cf: Maximum frequencies to plot per CF (default: 6)
    """
    output_path = Path(output_dir) / 'heatmaps'
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating correlation heatmaps for {fiber_type.upper()}...")
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Process each CF
    for cf_val in sorted(bez_data.keys()):
        frequencies = sorted(bez_data[cf_val].keys())
        
        # Limit number of frequencies per plot
        freq_subset = frequencies[:max_freqs_per_cf]
        n_freqs = len(freq_subset)
        
        if n_freqs == 0:
            continue
        
        # Calculate subplot grid
        n_cols = min(3, n_freqs)
        n_rows = int(np.ceil(n_freqs / n_cols))
        
        # Create figure
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 4 * n_rows)
        )
        
        # Ensure axes is always 2D array
        if n_freqs == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        # Plot heatmap for each frequency
        for idx, freq_val in enumerate(freq_subset):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Calculate correlation matrix
            corr_matrix, run_indices = calculate_correlation_matrix(
                bez_data, cf_val, freq_val
            )
            
            # Create heatmap
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='RdYlBu_r',
                vmin=0,
                vmax=1,
                square=True,
                cbar_kws={'label': 'Correlation'},
                ax=ax,
                xticklabels=run_indices,
                yticklabels=run_indices
            )
            
            ax.set_title(f'Freq: {freq_val:.0f} Hz')
            ax.set_xlabel('Run Index')
            ax.set_ylabel('Run Index')
        
        # Hide unused subplots
        for idx in range(n_freqs, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        # Add main title
        fig.suptitle(
            f'{fiber_type.upper()} - CF: {cf_val:.0f} Hz - '
            f'Within-BEZ Run Correlations',
            fontsize=14,
            y=1.00
        )
        
        plt.tight_layout()
        
        # Save figure
        filename = (f'correlation_heatmap_{fiber_type}_'
                   f'CF{cf_val:.0f}Hz.png')
        filepath = output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {filename}")
    
    print(f"✓ Heatmaps saved to: {output_path}")


def calculate_cochlea_bez_correlation(cochlea_data, bez_data):
    """
    Calculate correlation between Cochlea and mean BEZ response.
    
    Parameters:
    - cochlea_data: Cochlea data dict [cf][freq]
    - bez_data: BEZ data dict [cf][freq][run]
    
    Returns:
    - correlations: Dict [cf][freq] = correlation coefficient
    """
    correlations = {}
    
    for cf_val in cochlea_data:
        if cf_val not in bez_data:
            continue
            
        correlations[cf_val] = {}
        
        for freq_val in cochlea_data[cf_val]:
            if freq_val not in bez_data[cf_val]:
                continue
            
            cochlea_response = cochlea_data[cf_val][freq_val]
            
            # Calculate mean BEZ response across runs
            bez_runs = bez_data[cf_val][freq_val]
            bez_arrays = [bez_runs[k] for k in sorted(bez_runs.keys())]
            
            # Stack and average
            bez_stacked = np.stack(bez_arrays)
            mean_bez = np.mean(bez_stacked, axis=0)
            
            # Ensure same length
            min_len = min(len(cochlea_response), len(mean_bez))
            
            r, _ = stats.pearsonr(
                cochlea_response[:min_len], 
                mean_bez[:min_len]
            )
            correlations[cf_val][freq_val] = r
    
    return correlations


def calculate_percentile_rank(value, distribution):
    """
    Calculate percentile rank of value in distribution.
    
    Parameters:
    - value: Single value
    - distribution: Array of values
    
    Returns:
    - percentile: Percentile rank (0-100)
    """
    return stats.percentileofscore(distribution, value)


def analyze_correlations(bez_file, cochlea_file, fiber_type='hsr'):
    """
    Main correlation analysis function.
    
    Parameters:
    - bez_file: Path to extracted BEZ data
    - cochlea_file: Path to extracted Cochlea data
    - fiber_type: Fiber type to analyze
    
    Returns:
    - results: Dictionary with all correlation results
    """
    logging.info(f"=== Correlation Analysis: {fiber_type.upper()} ===\n")
    
    # Load data
    logging.info("Loading data...")
    bez_data, bez_meta = load_extracted_data(bez_file)
    cochlea_data, cochlea_meta = load_extracted_data(cochlea_file)
    
    # DEBUG: Check data structure
    logging.info("\nDEBUG: Checking BEZ data structure...")
    logging.info(f"Top-level keys: {list(bez_data.keys())}")
    
    # Extract fiber-specific data
    bez_fiber_data = bez_data[fiber_type]
    cochlea_fiber_data = cochlea_data[fiber_type]
    
    # DEBUG: Inspect first CF/Freq combination
    first_cf = list(bez_fiber_data.keys())[0]
    first_freq = list(bez_fiber_data[first_cf].keys())[0]
    first_runs = bez_fiber_data[first_cf][first_freq]
    
    logging.info(
        f"\nDEBUG: First condition (CF={first_cf}Hz, "
        f"Freq={first_freq}Hz):"
    )
    logging.info(f"  Run keys: {list(first_runs.keys())}")
    logging.info(f"  Number of runs: {len(first_runs)}")
    
    # Check if runs are different
    if len(first_runs) >= 2:
        run_keys = sorted(first_runs.keys())
        run0 = first_runs[run_keys[0]]
        run1 = first_runs[run_keys[1]]
        
        logging.info(f"  Run {run_keys[0]} shape: {run0.shape}")
        logging.info(f"  Run {run_keys[1]} shape: {run1.shape}")
        logging.info(f"  Run {run_keys[0]} first 5: {run0[:5]}")
        logging.info(f"  Run {run_keys[1]} first 5: {run1[:5]}")
        logging.info(
            f"  Arrays identical? {np.array_equal(run0, run1)}"
        )
        
        if np.array_equal(run0, run1):
            logging.warning(
                "\n*** WARNING: Runs appear to be identical! ***"
            )
            logging.warning(
                "This suggests an issue with data extraction."
            )
            logging.warning("Check extract_db_utility.py\n")
    
    logging.info(
        f"\n  BEZ dB level: "
        f"{bez_meta['extraction_summary']['target_db']}"
    )
    logging.info(
        f"  Cochlea dB level: "
        f"{cochlea_meta['extraction_summary']['target_db']}"
    )
    
    # Calculate within-BEZ correlations
    logging.info("\nCalculating within-BEZ correlations...")
    within_bez, pairs = calculate_within_bez_correlations(
        bez_fiber_data, fiber_type
    )
    
    # Calculate Cochlea-BEZ correlations
    logging.info("Calculating Cochlea-BEZ correlations...")
    cochlea_bez = calculate_cochlea_bez_correlation(
        cochlea_fiber_data, bez_fiber_data
    )
    
    # Compare distributions
    logging.info("\nAnalyzing correlation distributions...")
    results = {
        'fiber_type': fiber_type,
        'within_bez': within_bez,
        'cochlea_bez': cochlea_bez,
        'percentile_ranks': {},
        'summary_stats': {},
        'bez_fiber_data': bez_fiber_data  # Add for heatmap plotting
    }
    
    # For each CF x Freq combination
    for cf_val in cochlea_bez:
        results['percentile_ranks'][cf_val] = {}
        
        for freq_val in cochlea_bez[cf_val]:
            cb_corr = cochlea_bez[cf_val][freq_val]
            wb_corrs = within_bez[cf_val][freq_val]
            
            percentile = calculate_percentile_rank(cb_corr, wb_corrs)
            results['percentile_ranks'][cf_val][freq_val] = percentile
    
    # Calculate summary statistics
    all_within_bez = np.concatenate([
        within_bez[cf][freq] 
        for cf in within_bez 
        for freq in within_bez[cf]
    ])
    
    all_cochlea_bez = np.array([
        cochlea_bez[cf][freq] 
        for cf in cochlea_bez 
        for freq in cochlea_bez[cf]
    ])
    
    results['summary_stats'] = {
        'within_bez_mean': np.mean(all_within_bez),
        'within_bez_std': np.std(all_within_bez),
        'within_bez_median': np.median(all_within_bez),
        'cochlea_bez_mean': np.mean(all_cochlea_bez),
        'cochlea_bez_std': np.std(all_cochlea_bez),
        'cochlea_bez_median': np.median(all_cochlea_bez),
        'overall_percentile': calculate_percentile_rank(
            np.mean(all_cochlea_bez), all_within_bez
        )
    }
    
    # Print summary
    print(f"\n=== Summary Statistics ===")
    print(f"Within-BEZ correlations:")
    print(f"  Mean: {results['summary_stats']['within_bez_mean']:.3f}")
    print(f"  Std:  {results['summary_stats']['within_bez_std']:.3f}")
    print(f"  Median: "
          f"{results['summary_stats']['within_bez_median']:.3f}")
    print(f"\nCochlea-BEZ correlations:")
    print(f"  Mean: {results['summary_stats']['cochlea_bez_mean']:.3f}")
    print(f"  Std:  {results['summary_stats']['cochlea_bez_std']:.3f}")
    print(f"  Median: "
          f"{results['summary_stats']['cochlea_bez_median']:.3f}")
    print(f"\nOverall percentile rank of Cochlea-BEZ: "
          f"{results['summary_stats']['overall_percentile']:.1f}%")
    
    return results


def plot_correlation_comparison(results, output_dir=None):
    """
    Plot comparison of within-BEZ and Cochlea-BEZ correlations.
    
    Parameters:
    - results: Results dictionary from analyze_correlations
    - output_dir: Optional directory to save plot
    """
    # Aggregate correlations
    all_within = np.concatenate([
        results['within_bez'][cf][freq] 
        for cf in results['within_bez'] 
        for freq in results['within_bez'][cf]
    ])
    
    all_cochlea_bez = np.array([
        results['cochlea_bez'][cf][freq] 
        for cf in results['cochlea_bez'] 
        for freq in results['cochlea_bez'][cf]
    ])
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram comparison
    ax = axes[0]
    ax.hist(all_within, bins=30, alpha=0.6, label='Within-BEZ', 
            color='blue', density=True)
    ax.hist(all_cochlea_bez, bins=30, alpha=0.6, 
            label='Cochlea-BEZ', color='red', density=True)
    ax.axvline(np.mean(all_within), color='blue', 
               linestyle='--', label='Within-BEZ mean')
    ax.axvline(np.mean(all_cochlea_bez), color='red', 
               linestyle='--', label='Cochlea-BEZ mean')
    ax.set_xlabel('Correlation Coefficient')
    ax.set_ylabel('Density')
    ax.set_title(f'Correlation Distributions - '
                 f'{results["fiber_type"].upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box plot comparison
    ax = axes[1]
    ax.boxplot([all_within, all_cochlea_bez], 
               labels=['Within-BEZ', 'Cochlea-BEZ'])
    ax.set_ylabel('Correlation Coefficient')
    ax.set_title('Correlation Comparison')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_path / f'correlation_comparison_'
                          f'{results["fiber_type"]}.png',
            dpi=300, bbox_inches='tight'
        )
    
    plt.show()


def plot_correlation_distribution_with_cochlea(
    results,
    output_dir,
    show_individual_conditions=False
):
    """
    Plot distribution of within-BEZ correlations with Cochlea-BEZ overlay.
    Uses KDE for smooth distribution visualization and scatter for points.
    
    Parameters:
    - results: Results dictionary from analyze_correlations
    - output_dir: Directory to save plots
    - show_individual_conditions: If True, show per-condition analysis
    """
    from scipy.stats import gaussian_kde
    
    output_path = Path(output_dir) / 'distributions'
    output_path.mkdir(parents=True, exist_ok=True)
    
    fiber_type = results['fiber_type']
    
    # Aggregate all within-BEZ correlations
    all_within = np.concatenate([
        results['within_bez'][cf][freq] 
        for cf in results['within_bez'] 
        for freq in results['within_bez'][cf]
    ])
    
    # Aggregate all Cochlea-BEZ correlations
    all_cochlea_bez = np.array([
        results['cochlea_bez'][cf][freq] 
        for cf in results['cochlea_bez'] 
        for freq in results['cochlea_bez'][cf]
    ])
    
    # Calculate statistics
    within_mean = np.mean(all_within)
    within_std = np.std(all_within)
    cb_mean = np.mean(all_cochlea_bez)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create KDE for within-BEZ distribution
    if len(all_within) > 1:
        kde_within = gaussian_kde(all_within)
        x_range = np.linspace(
            max(0, all_within.min() - 0.1),
            min(1, all_within.max() + 0.1),
            200
        )
        kde_values = kde_within(x_range)
        
        # Plot KDE as filled curve
        ax.fill_between(
            x_range, kde_values,
            alpha=0.3,
            color='blue',
            label=f'Within-BEZ distribution\n'
                  f'(μ={within_mean:.3f}, σ={within_std:.3f})'
        )
        
        # Plot KDE line
        ax.plot(x_range, kde_values, color='blue', linewidth=2)
    
    # Scatter plot of within-BEZ correlations
    y_jitter = np.random.normal(0, 0.02, size=len(all_within))
    ax.scatter(
        all_within,
        y_jitter,
        alpha=0.3,
        s=20,
        color='blue',
        marker='o',
        label=f'Within-BEZ correlations (n={len(all_within)})'
    )
    
    # Plot Cochlea-BEZ correlations
    y_jitter_cb = np.random.normal(0, 0.02, size=len(all_cochlea_bez))
    ax.scatter(
        all_cochlea_bez,
        y_jitter_cb,
        alpha=0.6,
        s=50,
        color='red',
        marker='D',
        edgecolors='darkred',
        linewidth=1,
        label=f'Cochlea-BEZ correlations (n={len(all_cochlea_bez)})'
    )
    
    # Add mean lines
    ax.axvline(
        within_mean,
        color='blue',
        linestyle='--',
        linewidth=2,
        alpha=0.7,
        label=f'Within-BEZ mean: {within_mean:.3f}'
    )
    
    ax.axvline(
        cb_mean,
        color='red',
        linestyle='--',
        linewidth=2,
        alpha=0.7,
        label=f'Cochlea-BEZ mean: {cb_mean:.3f}'
    )
    
    # Add shaded region for ±1 std
    ax.axvspan(
        within_mean - within_std,
        within_mean + within_std,
        alpha=0.1,
        color='blue',
        label=f'Within-BEZ ±1σ'
    )
    
    # Styling
    ax.set_xlabel('Correlation Coefficient', fontsize=12)
    ax.set_ylabel('Density / Sample Points', fontsize=12)
    ax.set_title(
        f'{fiber_type.upper()} - Distribution of Correlations\n'
        f'Within-BEZ vs Cochlea-BEZ',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xlim(max(0, all_within.min() - 0.1), 
                min(1, all_within.max() + 0.1))
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'correlation_distribution_{fiber_type}.png'
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logging.info(f"Saved distribution plot: {filename}")
    
    # Optionally plot individual conditions
    if show_individual_conditions:
        plot_per_condition_distributions(
            results, output_path, fiber_type
        )


def plot_per_condition_distributions(results, output_path, fiber_type):
    """
    Plot distributions for individual CF × Frequency conditions.
    
    Parameters:
    - results: Results dictionary
    - output_path: Path to save plots
    - fiber_type: Fiber type being analyzed
    """
    from scipy.stats import gaussian_kde
    
    # Create subfolder for per-condition plots
    condition_path = output_path / 'per_condition'
    condition_path.mkdir(parents=True, exist_ok=True)
    
    within_bez = results['within_bez']
    cochlea_bez = results['cochlea_bez']
    
    # Limit to first few CFs for visualization
    cf_values = sorted(list(within_bez.keys()))[:5]
    
    for cf_val in cf_values:
        frequencies = sorted(list(within_bez[cf_val].keys()))[:6]
        
        n_freqs = len(frequencies)
        if n_freqs == 0:
            continue
        
        # Create subplot grid
        n_cols = min(3, n_freqs)
        n_rows = int(np.ceil(n_freqs / n_cols))
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 4 * n_rows)
        )
        
        # Ensure axes is array
        if n_freqs == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        for idx, freq_val in enumerate(frequencies):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Get within-BEZ correlations for this condition
            wb_corrs = within_bez[cf_val][freq_val]
            
            # Get Cochlea-BEZ correlation
            if (cf_val in cochlea_bez and 
                freq_val in cochlea_bez[cf_val]):
                cb_corr = cochlea_bez[cf_val][freq_val]
            else:
                cb_corr = None
            
            # Plot KDE if enough points
            if len(wb_corrs) > 1:
                kde = gaussian_kde(wb_corrs)
                x_range = np.linspace(
                    max(0, wb_corrs.min() - 0.05),
                    min(1, wb_corrs.max() + 0.05),
                    100
                )
                kde_values = kde(x_range)
                
                ax.fill_between(
                    x_range, kde_values,
                    alpha=0.3, color='blue'
                )
                ax.plot(x_range, kde_values, color='blue', linewidth=2)
            
            # Scatter within-BEZ points
            y_jitter = np.random.normal(0, 0.01, size=len(wb_corrs))
            ax.scatter(
                wb_corrs, y_jitter,
                alpha=0.4, s=15, color='blue'
            )
            
            # Plot Cochlea-BEZ correlation
            if cb_corr is not None:
                ax.scatter(
                    [cb_corr], [0],
                    s=80, color='red', marker='D',
                    edgecolors='darkred', linewidth=1.5,
                    label=f'Cochlea-BEZ: {cb_corr:.3f}'
                )
                
                # Add percentile info
                percentile = results['percentile_ranks'][cf_val][freq_val]
                ax.text(
                    0.95, 0.95,
                    f'{percentile:.0f}th percentile',
                    transform=ax.transAxes,
                    ha='right', va='top',
                    fontsize=8,
                    bbox=dict(
                        boxstyle='round',
                        facecolor='wheat',
                        alpha=0.5
                    )
                )
            
            ax.set_title(f'Freq: {freq_val:.0f} Hz', fontsize=10)
            ax.set_xlabel('Correlation', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_freqs, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        # Main title
        fig.suptitle(
            f'{fiber_type.upper()} - CF: {cf_val:.0f} Hz - '
            f'Correlation Distributions by Frequency',
            fontsize=12,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        # Save
        filename = (f'correlation_dist_percondition_{fiber_type}_'
                   f'CF{cf_val:.0f}Hz.png')
        filepath = condition_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logging.info(f"  Saved per-condition plot: {filename}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Correlation analysis between Cochlea and BEZ models'
    )
    
    parser.add_argument(
        '--bez-file',
        type=str,
        required=True,
        help='Path to extracted BEZ data pickle file'
    )
    
    parser.add_argument(
        '--cochlea-file',
        type=str,
        required=True,
        help='Path to extracted Cochlea data pickle file'
    )
    
    parser.add_argument(
        '--fiber-type',
        type=str,
        choices=['hsr', 'msr', 'lsr'],
        default='hsr',
        help='Fiber type to analyze (default: hsr)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/correlations',
        help='Output directory for plots and results'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to pickle file'
    )
    
    parser.add_argument(
        '--heatmaps',
        action='store_true',
        help='Generate correlation heatmaps'
    )
    
    parser.add_argument(
        '--distributions',
        action='store_true',
        help='Generate correlation distribution plots'
    )
    
    parser.add_argument(
        '--per-condition',
        action='store_true',
        help='Generate per-condition distribution plots'
    )
    
    parser.add_argument(
        '--max-freqs-per-cf',
        type=int,
        default=6,
        help='Maximum frequencies per CF in heatmaps (default: 6)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    args = parse_arguments()
    
    # Run analysis
    results = analyze_correlations(
        args.bez_file,
        args.cochlea_file,
        args.fiber_type
    )
    
    # Plot results
    plot_correlation_comparison(results, args.output_dir)
    
    # Generate distribution plots if requested
    if args.distributions:
        logging.info("\nGenerating correlation distribution plots...")
        plot_correlation_distribution_with_cochlea(
            results,
            args.output_dir,
            show_individual_conditions=args.per_condition
        )
    
    # Generate heatmaps if requested
    if args.heatmaps:
        plot_within_bez_correlation_heatmaps(
            results['bez_fiber_data'],
            args.fiber_type,
            args.output_dir,
            max_freqs_per_cf=args.max_freqs_per_cf
        )
    
    # Save results if requested
    if args.save:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Remove raw data before saving to reduce file size
        results_to_save = {
            k: v for k, v in results.items() 
            if k != 'bez_fiber_data'
        }
        
        output_file = (output_path / 
                      f'correlation_results_{args.fiber_type}.pkl')
        
        with open(output_file, 'wb') as f:
            pickle.dump(results_to_save, f)
        
        logging.info(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    results = main()
