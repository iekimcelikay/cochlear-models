"""
Statistical Testing Script for BEZ Run Analysis
Created: 2025-10-24

This script performs leave-one-out correlation analysis on BEZ runs to assess:
1. Consistency of responses across runs
2. Reliability of individual runs
3. Statistical significance of correlations
4. Identification of outlier runs

Uses the convolved HRF responses from the BEZ run-by-run analysis.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import argparse
import pandas as pd
from itertools import combinations
from convolution_utils import (
    save_plot,
    generate_timestamp,
    get_frequency_colors
)


def load_convolved_responses(data_file):
    """
    Load convolved responses from pickle file saved by the convolution script.
    
    Parameters:
    - data_file: Path to the pickle file containing convolved responses
    
    Returns:
    - convolved_responses: Dictionary with convolved responses for each run
    """
    print(f"Loading convolved responses from: {data_file}")
    
    with open(data_file, 'rb') as f:
        convolved_responses = pickle.load(f)
    
    print("✓ Convolved responses loaded successfully")
    return convolved_responses


def extract_run_data(convolved_responses, fiber_type, cf_val, freq_val, db_val):
    """
    Extract data for all runs of a specific condition.
    
    Parameters:
    - convolved_responses: Dictionary with convolved responses
    - fiber_type: Fiber type ('hsr', 'msr', 'lsr')
    - cf_val: CF value
    - freq_val: Frequency value  
    - db_val: dB level value
    
    Returns:
    - run_data: List of convolved responses for each run
    - valid_run_indices: List of run indices that have valid data
    """
    try:
        condition_data = convolved_responses[fiber_type][cf_val][freq_val][db_val]
        
        run_data = []
        valid_run_indices = []
        
        for run_idx, response in condition_data.items():
            if response is not None and len(response) > 0:
                run_data.append(response)
                valid_run_indices.append(run_idx)
        
        return run_data, valid_run_indices
    
    except KeyError:
        print(f"No data found for {fiber_type} CF={cf_val}Hz, Freq={freq_val}Hz, dB={db_val}")
        return [], []


def pad_responses_to_same_length(responses):
    """
    Pad all responses to the same length with NaN for consistent correlation analysis.
    
    Parameters:
    - responses: List of response arrays with potentially different lengths
    
    Returns:
    - padded_responses: Array of shape (n_runs, max_length) with NaN padding
    """
    if not responses:
        return np.array([])
    
    max_length = max(len(response) for response in responses)
    n_runs = len(responses)
    
    padded_responses = np.full((n_runs, max_length), np.nan)
    
    for i, response in enumerate(responses):
        padded_responses[i, :len(response)] = response
    
    return padded_responses


def calculate_leave_one_out_correlations(responses, correlation_type='pearson'):
    """
    Calculate leave-one-out correlations for a set of runs.
    
    For each run, correlate it with the mean of all other runs.
    
    Parameters:
    - responses: Array of shape (n_runs, n_timepoints)
    - correlation_type: 'pearson' or 'spearman'
    
    Returns:
    - correlations: Array of correlations for each run
    - p_values: Array of p-values for each correlation
    - mean_responses: Array of mean responses used for each correlation
    """
    n_runs, n_timepoints = responses.shape
    correlations = np.full(n_runs, np.nan)
    p_values = np.full(n_runs, np.nan)
    mean_responses = np.full((n_runs, n_timepoints), np.nan)
    
    for i in range(n_runs):
        # Get all runs except the current one
        other_runs = np.delete(responses, i, axis=0)
        
        # Calculate mean of other runs (ignoring NaN values)
        mean_other = np.nanmean(other_runs, axis=0)
        mean_responses[i] = mean_other
        
        # Get current run
        current_run = responses[i]
        
        # Find valid (non-NaN) timepoints in both signals
        valid_mask = ~(np.isnan(current_run) | np.isnan(mean_other))
        
        if np.sum(valid_mask) > 10:  # Need at least 10 valid points for correlation
            current_valid = current_run[valid_mask]
            mean_valid = mean_other[valid_mask]
            
            # Calculate correlation
            if correlation_type == 'pearson':
                corr, p_val = pearsonr(current_valid, mean_valid)
            else:  # spearman
                corr, p_val = spearmanr(current_valid, mean_valid)
            
            correlations[i] = corr
            p_values[i] = p_val
    
    return correlations, p_values, mean_responses


def calculate_pairwise_correlations(responses, correlation_type='pearson'):
    """
    Calculate all pairwise correlations between runs.
    
    Parameters:
    - responses: Array of shape (n_runs, n_timepoints)
    - correlation_type: 'pearson' or 'spearman'
    
    Returns:
    - correlation_matrix: Symmetric matrix of correlations
    - p_value_matrix: Symmetric matrix of p-values
    """
    n_runs = responses.shape[0]
    correlation_matrix = np.full((n_runs, n_runs), np.nan)
    p_value_matrix = np.full((n_runs, n_runs), np.nan)
    
    for i in range(n_runs):
        for j in range(i, n_runs):
            if i == j:
                correlation_matrix[i, j] = 1.0
                p_value_matrix[i, j] = 0.0
            else:
                # Find valid timepoints for both runs
                valid_mask = ~(np.isnan(responses[i]) | np.isnan(responses[j]))
                
                if np.sum(valid_mask) > 10:
                    run_i_valid = responses[i][valid_mask]
                    run_j_valid = responses[j][valid_mask]
                    
                    if correlation_type == 'pearson':
                        corr, p_val = pearsonr(run_i_valid, run_j_valid)
                    else:
                        corr, p_val = spearmanr(run_i_valid, run_j_valid)
                    
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr  # Symmetric
                    p_value_matrix[i, j] = p_val
                    p_value_matrix[j, i] = p_val
    
    return correlation_matrix, p_value_matrix


def identify_outlier_runs(correlations, threshold=2.0):
    """
    Identify outlier runs based on z-score of correlations.
    
    Parameters:
    - correlations: Array of correlation values
    - threshold: Z-score threshold for outlier detection
    
    Returns:
    - outlier_indices: Indices of outlier runs
    - z_scores: Z-scores for all runs
    """
    valid_correlations = correlations[~np.isnan(correlations)]
    
    if len(valid_correlations) < 3:
        return [], np.full_like(correlations, np.nan)
    
    mean_corr = np.mean(valid_correlations)
    std_corr = np.std(valid_correlations)
    
    z_scores = np.full_like(correlations, np.nan)
    valid_mask = ~np.isnan(correlations)
    z_scores[valid_mask] = (correlations[valid_mask] - mean_corr) / std_corr
    
    outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
    
    return outlier_indices, z_scores


def analyze_condition_statistics(convolved_responses, fiber_type, cf_val, freq_val, db_val, 
                                correlation_type='pearson', outlier_threshold=2.0):
    """
    Perform complete statistical analysis for a single condition.
    
    Parameters:
    - convolved_responses: Dictionary with convolved responses
    - fiber_type: Fiber type to analyze
    - cf_val: CF value
    - freq_val: Frequency value
    - db_val: dB level
    - correlation_type: Type of correlation ('pearson' or 'spearman')
    - outlier_threshold: Z-score threshold for outlier detection
    
    Returns:
    - results: Dictionary with all statistical results
    """
    # Extract run data
    run_data, valid_run_indices = extract_run_data(
        convolved_responses, fiber_type, cf_val, freq_val, db_val
    )
    
    if len(run_data) < 3:
        print(f"Insufficient data for {fiber_type} CF={cf_val}Hz, Freq={freq_val}Hz, dB={db_val}")
        return None
    
    # Pad responses to same length
    padded_responses = pad_responses_to_same_length(run_data)
    
    # Calculate leave-one-out correlations
    loo_correlations, loo_p_values, loo_means = calculate_leave_one_out_correlations(
        padded_responses, correlation_type
    )
    
    # Calculate pairwise correlations
    pairwise_corr_matrix, pairwise_p_matrix = calculate_pairwise_correlations(
        padded_responses, correlation_type
    )
    
    # Identify outliers
    outlier_indices, z_scores = identify_outlier_runs(loo_correlations, outlier_threshold)
    
    # Compile results
    results = {
        'condition': {
            'fiber_type': fiber_type,
            'cf_val': cf_val,
            'freq_val': freq_val,
            'db_val': db_val
        },
        'data': {
            'responses': padded_responses,
            'valid_run_indices': valid_run_indices,
            'n_runs': len(run_data)
        },
        'leave_one_out': {
            'correlations': loo_correlations,
            'p_values': loo_p_values,
            'mean_responses': loo_means,
            'mean_correlation': np.nanmean(loo_correlations),
            'std_correlation': np.nanstd(loo_correlations)
        },
        'pairwise': {
            'correlation_matrix': pairwise_corr_matrix,
            'p_value_matrix': pairwise_p_matrix,
            'mean_correlation': np.nanmean(pairwise_corr_matrix[np.triu_indices_from(pairwise_corr_matrix, k=1)])
        },
        'outliers': {
            'indices': outlier_indices,
            'z_scores': z_scores,
            'threshold': outlier_threshold
        },
        'statistics': {
            'correlation_type': correlation_type,
            'valid_correlations': np.sum(~np.isnan(loo_correlations)),
            'significant_correlations': np.sum(loo_p_values < 0.05) if not np.all(np.isnan(loo_p_values)) else 0
        }
    }
    
    return results


def plot_correlation_results(results, tr=0.1, figsize=(15, 12), save_plots=False, 
                           output_dir=None, timestamp=None, plot_format='png'):
    """
    Create comprehensive plots for correlation analysis results.
    
    Parameters:
    - results: Results dictionary from analyze_condition_statistics
    - tr: Temporal resolution for time axis
    - figsize: Figure size
    - save_plots: Whether to save plots
    - output_dir: Output directory for saving
    - timestamp: Timestamp for filenames
    - plot_format: Format for saved plots
    """
    if results is None:
        return
    
    condition = results['condition']
    responses = results['data']['responses']
    loo_corr = results['leave_one_out']['correlations']
    loo_means = results['leave_one_out']['mean_responses']
    pairwise_matrix = results['pairwise']['correlation_matrix']
    outlier_indices = results['outliers']['indices']
    valid_run_indices = results['data']['valid_run_indices']
    
    # Create time axis
    time_axis = np.arange(responses.shape[1]) * tr
    
    # Create subplot layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: All runs with outliers highlighted
    ax1 = fig.add_subplot(gs[0, :2])
    for i, response in enumerate(responses):
        color = 'red' if i in outlier_indices else 'blue'
        alpha = 0.8 if i in outlier_indices else 0.3
        linewidth = 2 if i in outlier_indices else 1
        label = f'Run {valid_run_indices[i]} (outlier)' if i in outlier_indices else None
        ax1.plot(time_axis, response, color=color, alpha=alpha, linewidth=linewidth, label=label)
    
    # Plot mean response
    mean_response = np.nanmean(responses, axis=0)
    ax1.plot(time_axis, mean_response, 'k-', linewidth=3, label='Mean across runs')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('HRF Response')
    ax1.set_title(f'{condition["fiber_type"].upper()} - CF: {condition["cf_val"]}Hz, '
                  f'Freq: {condition["freq_val"]}Hz, dB: {condition["db_val"]}\n'
                  f'All Runs (n={results["data"]["n_runs"]})')
    ax1.grid(True, alpha=0.3)
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend()
    
    # Plot 2: Leave-one-out correlations
    ax2 = fig.add_subplot(gs[0, 2])
    valid_corr_mask = ~np.isnan(loo_corr)
    x_pos = np.arange(len(loo_corr))
    colors = ['red' if i in outlier_indices else 'blue' for i in range(len(loo_corr))]
    
    bars = ax2.bar(x_pos[valid_corr_mask], loo_corr[valid_corr_mask], 
                   color=np.array(colors)[valid_corr_mask], alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Run Index')
    ax2.set_ylabel('Correlation with\nOther Runs')
    ax2.set_title('Leave-One-Out\nCorrelations')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation statistics
    mean_corr = results['leave_one_out']['mean_correlation']
    ax2.text(0.02, 0.98, f'Mean: {mean_corr:.3f}', transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 3: Pairwise correlation matrix
    ax3 = fig.add_subplot(gs[1, :2])
    im = ax3.imshow(pairwise_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax3.set_xlabel('Run Index')
    ax3.set_ylabel('Run Index')
    ax3.set_title('Pairwise Correlation Matrix')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('Correlation')
    
    # Plot 4: Correlation distribution
    ax4 = fig.add_subplot(gs[1, 2])
    valid_correlations = loo_corr[~np.isnan(loo_corr)]
    if len(valid_correlations) > 0:
        ax4.hist(valid_correlations, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(np.mean(valid_correlations), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(valid_correlations):.3f}')
        ax4.set_xlabel('Correlation')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Correlation Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Example leave-one-out comparison
    ax5 = fig.add_subplot(gs[2, :])
    if len(valid_correlations) > 0:
        # Find run with median correlation for example
        median_idx = np.argsort(valid_correlations)[len(valid_correlations)//2]
        median_run_idx = np.where(~np.isnan(loo_corr))[0][median_idx]
        
        example_run = responses[median_run_idx]
        example_mean = loo_means[median_run_idx]
        example_corr = loo_corr[median_run_idx]
        
        ax5.plot(time_axis, example_run, 'b-', linewidth=2, 
                label=f'Run {valid_run_indices[median_run_idx]}')
        ax5.plot(time_axis, example_mean, 'r-', linewidth=2, 
                label=f'Mean of other runs (r={example_corr:.3f})')
        
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('HRF Response')
        ax5.set_title('Example Leave-One-Out Comparison (Median Correlation)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    plt.suptitle(f'Statistical Analysis: {condition["fiber_type"].upper()} Fibers\n'
                 f'CF: {condition["cf_val"]}Hz, Freq: {condition["freq_val"]}Hz, '
                 f'dB: {condition["db_val"]}', fontsize=14, y=0.98)
    
    # Save plot if requested
    if save_plots and output_dir and timestamp:
        plot_name = (f'statistical_analysis_{condition["fiber_type"]}_'
                    f'{condition["cf_val"]}Hz_{condition["freq_val"]}Hz_{condition["db_val"]}dB')
        save_plot(plot_name, output_dir, timestamp, model_name='BEZ_Stats', 
                 fiber_type=condition["fiber_type"], plot_format=plot_format)
    
    plt.show()


def create_summary_statistics_table(all_results):
    """
    Create a summary table of statistics across all conditions.
    
    Parameters:
    - all_results: List of results dictionaries from multiple conditions
    
    Returns:
    - summary_df: Pandas DataFrame with summary statistics
    """
    summary_data = []
    
    for results in all_results:
        if results is None:
            continue
            
        condition = results['condition']
        loo_stats = results['leave_one_out']
        pairwise_stats = results['pairwise']
        outlier_stats = results['outliers']
        
        summary_data.append({
            'Fiber_Type': condition['fiber_type'],
            'CF_Hz': condition['cf_val'],
            'Freq_Hz': condition['freq_val'],
            'dB_Level': condition['db_val'],
            'N_Runs': results['data']['n_runs'],
            'Mean_LOO_Correlation': loo_stats['mean_correlation'],
            'Std_LOO_Correlation': loo_stats['std_correlation'],
            'Mean_Pairwise_Correlation': pairwise_stats['mean_correlation'],
            'N_Outliers': len(outlier_stats['indices']),
            'N_Significant_Correlations': results['statistics']['significant_correlations'],
            'Percent_Significant': (results['statistics']['significant_correlations'] / 
                                  results['statistics']['valid_correlations'] * 100) if results['statistics']['valid_correlations'] > 0 else 0
        })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df


def main():
    """Main execution function with command line arguments."""
    parser = argparse.ArgumentParser(description='Statistical testing of BEZ runs with leave-one-out correlations')
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to convolved responses pickle file')
    parser.add_argument('--output_dir', type=str, default='./statistical_results',
                       help='Output directory for results')
    parser.add_argument('--fiber_types', nargs='+', default=['hsr'],
                       help='Fiber types to analyze')
    parser.add_argument('--correlation_type', choices=['pearson', 'spearman'], default='pearson',
                       help='Type of correlation to compute')
    parser.add_argument('--outlier_threshold', type=float, default=2.0,
                       help='Z-score threshold for outlier detection')
    parser.add_argument('--max_conditions', type=int, default=None,
                       help='Maximum number of conditions to analyze (for testing)')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to file')
    parser.add_argument('--tr', type=float, default=0.1,
                       help='Temporal resolution (seconds)')
    
    args = parser.parse_args()
    
    # Generate timestamp for this analysis
    timestamp = generate_timestamp()
    print(f"Starting BEZ statistical analysis - {timestamp}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load convolved responses
    print("\n=== Loading Convolved Responses ===")
    convolved_responses = load_convolved_responses(args.data_file)
    
    # Analyze all conditions
    print(f"\n=== Analyzing Conditions ===")
    all_results = []
    condition_count = 0
    
    for fiber_type in args.fiber_types:
        if fiber_type not in convolved_responses:
            print(f"No data for fiber type: {fiber_type}")
            continue
            
        fiber_data = convolved_responses[fiber_type]
        
        for cf_val in fiber_data.keys():
            for freq_val in fiber_data[cf_val].keys():
                for db_val in fiber_data[cf_val][freq_val].keys():
                    
                    print(f"\nAnalyzing {fiber_type.upper()} CF={cf_val}Hz, "
                          f"Freq={freq_val}Hz, dB={db_val}")
                    
                    results = analyze_condition_statistics(
                        convolved_responses, fiber_type, cf_val, freq_val, db_val,
                        correlation_type=args.correlation_type,
                        outlier_threshold=args.outlier_threshold
                    )
                    
                    if results is not None:
                        all_results.append(results)
                        
                        # Create plots for first few conditions or if explicitly requested
                        if condition_count < 5 or args.save_plots:
                            plot_correlation_results(
                                results, tr=args.tr, save_plots=args.save_plots,
                                output_dir=args.output_dir, timestamp=timestamp
                            )
                    
                    condition_count += 1
                    
                    # Limit conditions for testing
                    if args.max_conditions and condition_count >= args.max_conditions:
                        break
                if args.max_conditions and condition_count >= args.max_conditions:
                    break
            if args.max_conditions and condition_count >= args.max_conditions:
                break
        if args.max_conditions and condition_count >= args.max_conditions:
            break
    
    # Create summary statistics
    print(f"\n=== Creating Summary Statistics ===")
    summary_df = create_summary_statistics_table(all_results)
    
    # Save summary table
    summary_file = Path(args.output_dir) / f'summary_statistics_{timestamp}.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to: {summary_file}")
    
    # Display summary
    print("\nSUMMARY STATISTICS:")
    print(summary_df.describe())
    
    # Save all results
    results_file = Path(args.output_dir) / f'all_statistical_results_{timestamp}.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"All results saved to: {results_file}")
    
    print(f"\n✓ Statistical analysis complete. Results saved to: {args.output_dir}")
    
    return all_results, summary_df


if __name__ == "__main__":
    results, summary = main()
