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
    - correlations: Array of correlations
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
    
    if std_corr == 0:
        return [], np.zeros_like(correlations)

    z_scores = np.abs((correlations - mean_corr) / std_corr)
    outlier_indices = np.where(z_scores > threshold)[0]

    return outlier_indices, z_scores


def analyze_condition_statistics(convolved_responses, fiber_type, cf_val, freq_val, db_val, 
                                correlation_type='pearson', outlier_threshold=2.0):
    """
    Perform statistical analysis for a single condition using pairwise correlations.

    Parameters:
    - convolved_responses: Dictionary with convolved responses
    - fiber_type: Fiber type to analyze
    - cf_val: CF value
    - freq_val: Frequency value
    - db_val: dB level
    - correlation_type: Type of correlation ('pearson' or 'spearman')
    - outlier_threshold: Z-score threshold for outlier detection
    
    Returns:
    - results: Dictionary with statistical results
    """
    # Extract run data
    run_data, valid_run_indices = extract_run_data(
        convolved_responses, fiber_type, cf_val, freq_val, db_val
    )
    
    if len(run_data) < 2:
        print(f"Insufficient data for {fiber_type} CF={cf_val}Hz, Freq={freq_val}Hz, dB={db_val}")
        return None
    
    # Pad responses to same length
    padded_responses = pad_responses_to_same_length(run_data)
    
    # Calculate pairwise correlations
    pairwise_corr_matrix, pairwise_p_matrix = calculate_pairwise_correlations(
        padded_responses, correlation_type
    )
    
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
        'pairwise': {
            'correlation_matrix': pairwise_corr_matrix,
            'p_value_matrix': pairwise_p_matrix,
            'mean_correlation': np.nanmean(pairwise_corr_matrix[np.triu_indices_from(pairwise_corr_matrix, k=1)])
        },
        'statistics': {
            'correlation_type': correlation_type,
            'n_pairwise_correlations': len(pairwise_corr_matrix[np.triu_indices_from(pairwise_corr_matrix, k=1)])
        }
    }
    
    return results


def plot_correlation_results(results, tr=0.1, figsize=(12, 8), save_plots=False,
                           output_dir=None, timestamp=None, plot_format='png'):
    """
    Create plots for correlation analysis results.

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
    pairwise_matrix = results['pairwise']['correlation_matrix']
    valid_run_indices = results['data']['valid_run_indices']
    
    # Create time axis
    time_axis = np.arange(responses.shape[1]) * tr
    
    # Create subplot layout (2x2)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: All runs
    for i, response in enumerate(responses):
        ax1.plot(time_axis, response, alpha=0.5, linewidth=1)

    # Plot mean response
    mean_response = np.nanmean(responses, axis=0)
    ax1.plot(time_axis, mean_response, 'k-', linewidth=3, label='Mean across runs')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('HRF Response')
    ax1.set_title(f'All Runs (n={results["data"]["n_runs"]})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Pairwise correlation matrix
    im = ax2.imshow(pairwise_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax2.set_xlabel('Run Index')
    ax2.set_ylabel('Run Index')
    ax2.set_title('Pairwise Correlation Matrix')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Correlation')
    
    # Plot 3: Pairwise correlation distribution
    upper_tri_indices = np.triu_indices_from(pairwise_matrix, k=1)
    pairwise_correlations = pairwise_matrix[upper_tri_indices]
    valid_pairwise = pairwise_correlations[~np.isnan(pairwise_correlations)]

    if len(valid_pairwise) > 0:
        ax3.hist(valid_pairwise, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.axvline(np.mean(valid_pairwise), color='red', linestyle='--',
                   label=f'Mean: {np.mean(valid_pairwise):.3f}')
        ax3.set_xlabel('Pairwise Correlation')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Pairwise Correlation Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Statistics summary
    ax4.axis('off')
    stats_text = f"""Statistical Summary:
    
Condition: {condition["fiber_type"].upper()}
CF: {condition["cf_val"]} Hz
Freq: {condition["freq_val"]} Hz  
dB: {condition["db_val"]}

Number of runs: {results["data"]["n_runs"]}
Correlation type: {results["statistics"]["correlation_type"]}

Pairwise correlations:
  Mean: {np.mean(valid_pairwise):.3f}
  Std: {np.std(valid_pairwise):.3f}
  Range: [{np.min(valid_pairwise):.3f}, {np.max(valid_pairwise):.3f}]
  N pairs: {len(valid_pairwise)}"""

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.suptitle(f'BEZ Run Analysis: {condition["fiber_type"].upper()} Fibers', fontsize=14)
    plt.tight_layout()

    # Save plot if requested
    if save_plots and output_dir and timestamp:
        plot_name = (f'statistical_analysis_{condition["fiber_type"]}_'
                    f'{condition["cf_val"]}Hz_{condition["freq_val"]}Hz_{condition["db_val"]}dB')
        save_plot(plot_name, output_dir, timestamp, model_name='BEZ_Stats', 
                 fiber_type=condition["fiber_type"], plot_format=plot_format)
    
    plt.show()


def plot_pairwise_correlation_matrix(correlation_matrix, valid_run_indices=None,
                                    condition_info=None, figsize=(8, 6),
                                    save_plot=False, output_dir=None, timestamp=None):
    """
    Plot pairwise correlation matrix as a heatmap with correlation values in cells.

    Parameters:
    - correlation_matrix: Square matrix of pairwise correlations
    - valid_run_indices: List of valid run indices for labeling
    - condition_info: Dictionary with condition information for title
    - figsize: Figure size tuple
    - save_plot: Whether to save the plot
    - output_dir: Directory to save plot
    - timestamp: Timestamp for filename

    Returns:
    - fig, ax: Figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    # Add correlation values as text in cells
    n_runs = correlation_matrix.shape[0]
    for i in range(n_runs):
        for j in range(n_runs):
            if not np.isnan(correlation_matrix[i, j]):
                # Use white text for dark cells, black for light cells
                text_color = 'white' if abs(correlation_matrix[i, j]) > 0.5 else 'black'
                ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                       ha='center', va='center', color=text_color, fontsize=8)

    # Set labels
    if valid_run_indices is not None:
        ax.set_xticks(range(len(valid_run_indices)))
        ax.set_yticks(range(len(valid_run_indices)))
        ax.set_xticklabels([f'Run {idx}' for idx in valid_run_indices], rotation=45)
        ax.set_yticklabels([f'Run {idx}' for idx in valid_run_indices])
    else:
        ax.set_xticks(range(n_runs))
        ax.set_yticks(range(n_runs))
        ax.set_xticklabels([f'Run {i}' for i in range(n_runs)], rotation=45)
        ax.set_yticklabels([f'Run {i}' for i in range(n_runs)])

    ax.set_xlabel('Run Index')
    ax.set_ylabel('Run Index')

    # Create title
    if condition_info:
        title = (f'Pairwise Correlation Matrix - {condition_info.get("fiber_type", "").upper()}\n'
                f'CF: {condition_info.get("cf_val", "N/A")}Hz, '
                f'Freq: {condition_info.get("freq_val", "N/A")}Hz, '
                f'dB: {condition_info.get("db_val", "N/A")}')
    else:
        title = 'Pairwise Correlation Matrix'

    ax.set_title(title)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient')

    plt.tight_layout()

    # Save plot if requested
    if save_plot and output_dir and timestamp:
        plot_name = 'pairwise_correlation_matrix'
        if condition_info:
            plot_name += f'_{condition_info.get("fiber_type", "unknown")}'
            plot_name += f'_{condition_info.get("cf_val", "unknown")}Hz'
            plot_name += f'_{condition_info.get("freq_val", "unknown")}Hz'
            plot_name += f'_{condition_info.get("db_val", "unknown")}dB'

        plt.savefig(f'{output_dir}/{plot_name}_{timestamp}.png', dpi=300, bbox_inches='tight')

    return fig, ax


def plot_pairwise_correlation_distribution(correlation_matrix, condition_info=None,
                                         figsize=(8, 6), bins=20, save_plot=False,
                                         output_dir=None, timestamp=None,
                                         cochlea_correlation=None, show_cochlea=False):
    """
    Plot distribution of pairwise correlations with mean and variance indicators.

    Parameters:
    - correlation_matrix: Square matrix of pairwise correlations
    - condition_info: Dictionary with condition information for title
    - figsize: Figure size tuple
    - bins: Number of histogram bins
    - save_plot: Whether to save the plot
    - output_dir: Directory to save plot
    - timestamp: Timestamp for filename
    - cochlea_correlation: Optional correlation value between cochlea and mean BEZ response
    - show_cochlea: Whether to display the cochlea correlation on the plot

    Returns:
    - fig, ax: Figure and axis objects
    - stats_dict: Dictionary with distribution statistics
    """
    # Extract upper triangular values (excluding diagonal)
    upper_tri_indices = np.triu_indices_from(correlation_matrix, k=1)
    pairwise_correlations = correlation_matrix[upper_tri_indices]
    valid_correlations = pairwise_correlations[~np.isnan(pairwise_correlations)]

    if len(valid_correlations) == 0:
        print("No valid correlations to plot")
        return None, None, None

    # Calculate statistics
    mean_corr = np.mean(valid_correlations)
    std_corr = np.std(valid_correlations)
    var_corr = np.var(valid_correlations)
    median_corr = np.median(valid_correlations)
    min_corr = np.min(valid_correlations)
    max_corr = np.max(valid_correlations)

    stats_dict = {
        'mean': mean_corr,
        'std': std_corr,
        'variance': var_corr,
        'median': median_corr,
        'min': min_corr,
        'max': max_corr,
        'n_pairs': len(valid_correlations)
    }

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram
    n, bins_edges, patches = ax.hist(valid_correlations, bins=bins, alpha=0.7,
                                    color='lightblue', edgecolor='black', density=True)

    # Add mean line
    ax.axvline(mean_corr, color='red', linestyle='-', linewidth=2,
              label=f'Mean: {mean_corr:.3f}')

    # Add standard deviation shading
    ax.axvspan(mean_corr - std_corr, mean_corr + std_corr, alpha=0.3,
              color='orange', label=f'±1 SD: {std_corr:.3f}')

    # Add median line
    ax.axvline(median_corr, color='green', linestyle='--', linewidth=2,
              label=f'Median: {median_corr:.3f}')

    # Add cochlea correlation line if provided
    if show_cochlea and cochlea_correlation is not None and not np.isnan(cochlea_correlation):
        ax.axvline(cochlea_correlation, color='purple', linestyle=':', linewidth=3,
                  label=f'Cochlea-BEZ: {cochlea_correlation:.3f}')

    # Fit and plot normal distribution for comparison
    x_fit = np.linspace(min_corr - 0.1, max_corr + 0.1, 100)
    y_fit = stats.norm.pdf(x_fit, mean_corr, std_corr)
    ax.plot(x_fit, y_fit, 'k--', alpha=0.8, linewidth=1.5,
           label=f'Normal fit (σ={std_corr:.3f})')

    # Set labels and title
    ax.set_xlabel('Pairwise Correlation Coefficient')
    ax.set_ylabel('Density')

    if condition_info:
        title = (f'Pairwise Correlation Distribution - {condition_info.get("fiber_type", "").upper()}\n'
                f'CF: {condition_info.get("cf_val", "N/A")}Hz, '
                f'Freq: {condition_info.get("freq_val", "N/A")}Hz, '
                f'dB: {condition_info.get("db_val", "N/A")}')
    else:
        title = 'Pairwise Correlation Distribution'

    ax.set_title(title)

    # Add statistics text box (updated to include cochlea correlation)
    stats_text = (f'N pairs: {len(valid_correlations)}\n'
                 f'Mean: {mean_corr:.3f}\n'
                 f'Std: {std_corr:.3f}\n'
                 f'Variance: {var_corr:.3f}\n'
                 f'Range: [{min_corr:.3f}, {max_corr:.3f}]')

    # Add cochlea correlation to stats text if available
    if show_cochlea and cochlea_correlation is not None and not np.isnan(cochlea_correlation):
        stats_text += f'\nCochlea-BEZ: {cochlea_correlation:.3f}'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot if requested
    if save_plot and output_dir and timestamp:
        plot_name = 'pairwise_correlation_distribution'
        if condition_info:
            plot_name += f'_{condition_info.get("fiber_type", "unknown")}'
            plot_name += f'_{condition_info.get("cf_val", "unknown")}Hz'
            plot_name += f'_{condition_info.get("freq_val", "unknown")}Hz'
            plot_name += f'_{condition_info.get("db_val", "unknown")}dB'

        plt.savefig(f'{output_dir}/{plot_name}_{timestamp}.png', dpi=300, bbox_inches='tight')

    return fig, ax, stats_dict

def load_cochlea_convolved_responses(cochlea_file):
    """
    Load cochlea responses convolved with HRF from pickle file.

    Parameters:
    - cochlea_file: Path to the pickle file containing cochlea convolved responses

    Returns:
    - cochlea_convolved: Dictionary with cochlea convolved responses
    """
    print(f"Loading cochlea convolved responses from: {cochlea_file}")

    with open(cochlea_file, 'rb') as f:
        cochlea_convolved = pickle.load(f)

    print("✓ Cochlea convolved responses loaded successfully")
    return cochlea_convolved


def extract_cochlea_data(cochlea_convolved, fiber_type, cf_val, freq_val, db_val):
    """
    Extract cochlea convolved response for a specific condition.

    Parameters:
    - cochlea_convolved: Dictionary with cochlea convolved responses
    - fiber_type: Fiber type ('hsr', 'msr', 'lsr')
    - cf_val: CF value
    - freq_val: Frequency value
    - db_val: dB level value

    Returns:
    - cochlea_response: Convolved response array for the condition
    """
    try:
        cochlea_response = cochlea_convolved[fiber_type][cf_val][freq_val][db_val]

        if cochlea_response is not None and len(cochlea_response) > 0:
            return cochlea_response
        else:
            print(f"No valid cochlea data for {fiber_type} CF={cf_val}Hz, Freq={freq_val}Hz, dB={db_val}")
            return None

    except KeyError:
        print(f"No cochlea data found for {fiber_type} CF={cf_val}Hz, Freq={freq_val}Hz, dB={db_val}")
        return None





def correlate_cochlea_with_bez_mean(cochlea_response, bez_runs, correlation_type='pearson'):
    """
    Correlate cochlea convolved response with mean of BEZ runs (convolved with HRF).

    Parameters:
    - cochlea_response: Cochlea convolved response array
    - bez_runs: List of BEZ run convolved responses
    - correlation_type: 'pearson' or 'spearman'

    Returns:
    - correlation: Correlation with mean BEZ response
    - p_value: P-value for the correlation
    - bez_mean: Mean BEZ response used for correlation
    """
    # Filter out None/empty runs
    valid_bez_runs = [run for run in bez_runs if run is not None and len(run) > 0]

    if len(valid_bez_runs) == 0:
        return np.nan, np.nan, None

    # Pad all responses to same length
    all_responses = valid_bez_runs + [cochlea_response]
    max_length = max(len(response) for response in all_responses)

    # Create padded array for BEZ runs
    padded_bez_runs = np.full((len(valid_bez_runs), max_length), np.nan)
    for i, bez_run in enumerate(valid_bez_runs):
        padded_bez_runs[i, :len(bez_run)] = bez_run

    # Pad cochlea response
    cochlea_padded = np.full(max_length, np.nan)
    cochlea_padded[:len(cochlea_response)] = cochlea_response

    # Calculate mean BEZ response (ignoring NaN values)
    bez_mean = np.nanmean(padded_bez_runs, axis=0)

    # Find valid timepoints for correlation
    valid_mask = ~(np.isnan(cochlea_padded) | np.isnan(bez_mean))

    if np.sum(valid_mask) > 10:  # Need at least 10 valid points
        cochlea_valid = cochlea_padded[valid_mask]
        bez_mean_valid = bez_mean[valid_mask]

        # Calculate correlation
        if correlation_type == 'pearson':
            correlation, p_value = pearsonr(cochlea_valid, bez_mean_valid)
        else:  # spearman
            correlation, p_value = spearmanr(cochlea_valid, bez_mean_valid)

        return correlation, p_value, bez_mean
    else:
        return np.nan, np.nan, bez_mean


def analyze_cochlea_bez_correlations(cochlea_file, bez_file, fiber_types=['hsr', 'msr', 'lsr'], 
                                   correlation_type='pearson', output_dir=None):
    """
    Analysis of correlations between cochlea (convolved with HRF) 
    and mean of BEZ runs (convolved with HRF).
    
    Parameters:
    - cochlea_file: Path to cochlea convolved responses pickle file
    - bez_file: Path to BEZ convolved responses pickle file
    - fiber_types: List of fiber types to analyze
    - correlation_type: 'pearson' or 'spearman'
    - output_dir: Directory to save results
    
    Returns:
    - results: Dictionary containing correlation results with mean BEZ
    """
    print("=== Cochlea-Mean BEZ Correlation Analysis ===\n")
    
    # Load data
    cochlea_convolved = load_cochlea_convolved_responses(cochlea_file)
    bez_convolved = load_convolved_responses(bez_file)
    
    results = {
        'mean_correlations': {},
        'statistics': {},
        'parameters': {
            'correlation_type': correlation_type,
            'fiber_types': fiber_types
        }
    }
    
    # Initialize result structures
    for fiber_type in fiber_types:
        results['mean_correlations'][fiber_type] = {}
        results['statistics'][fiber_type] = {}
    
    # Get all conditions from cochlea data
    total_conditions = 0
    processed_conditions = 0
    
    for fiber_type in fiber_types:
        print(f"\nAnalyzing {fiber_type.upper()} fibers...")
        
        if fiber_type not in cochlea_convolved:
            print(f"No {fiber_type} data in cochlea file")
            continue
            
        results['mean_correlations'][fiber_type] = {}
        results['statistics'][fiber_type] = {}
        
        for cf_val in cochlea_convolved[fiber_type]:
            results['mean_correlations'][fiber_type][cf_val] = {}
            results['statistics'][fiber_type][cf_val] = {}
            
            for freq_val in cochlea_convolved[fiber_type][cf_val]:
                results['mean_correlations'][fiber_type][cf_val][freq_val] = {}
                results['statistics'][fiber_type][cf_val][freq_val] = {}
                
                for db_val in cochlea_convolved[fiber_type][cf_val][freq_val]:
                    total_conditions += 1
                    
                    # Extract cochlea response
                    cochlea_response = extract_cochlea_data(
                        cochlea_convolved, fiber_type, cf_val, freq_val, db_val
                    )
                    
                    if cochlea_response is None:
                        continue
                    
                    # Extract BEZ runs
                    bez_runs, valid_run_indices = extract_run_data(
                        bez_convolved, fiber_type, cf_val, freq_val, db_val
                    )
                    
                    if len(bez_runs) == 0:
                        print(f"No BEZ data for {fiber_type} CF={cf_val}Hz, Freq={freq_val}Hz, dB={db_val}")
                        continue
                    
                    processed_conditions += 1
                    
                    # Correlate with mean BEZ response
                    mean_corr, mean_pval, bez_mean = correlate_cochlea_with_bez_mean(
                        cochlea_response, bez_runs, correlation_type
                    )
                    
                    # Store results
                    results['mean_correlations'][fiber_type][cf_val][freq_val][db_val] = {
                        'correlation': mean_corr,
                        'p_value': mean_pval,
                        'n_runs_used': len([r for r in bez_runs if r is not None and len(r) > 0]),
                        'cochlea_length': len(cochlea_response),
                        'bez_mean_length': len(bez_mean) if bez_mean is not None else 0
                    }
                    
                    # Store basic statistics
                    results['statistics'][fiber_type][cf_val][freq_val][db_val] = {
                        'correlation': mean_corr,
                        'p_value': mean_pval,
                        'significant': mean_pval < 0.05 if not np.isnan(mean_pval) else False,
                        'n_bez_runs': len([r for r in bez_runs if r is not None and len(r) > 0])
                    }
    
    print(f"\nProcessed {processed_conditions}/{total_conditions} conditions")
    
    # Save results if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = generate_timestamp()
        results_file = os.path.join(output_dir, f'cochlea_bez_mean_correlations_{timestamp}.pkl')
        
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to: {results_file}")
    
    return results



def main():
    """
    Main function for statistical testing of BEZ runs with command-line interface.
    """
    parser = argparse.ArgumentParser(
        description='Statistical Testing Script for BEZ Run Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with default parameters
  python statistical_testing_bez_runs.py --bez_file data/bez_convolved.pkl
  
  # Include cochlea comparison
  python statistical_testing_bez_runs.py --bez_file data/bez_convolved.pkl --cochlea_file data/cochlea_convolved.pkl
  
  # Analyze specific condition with custom parameters
  python statistical_testing_bez_runs.py --bez_file data/bez_convolved.pkl --fiber_type hsr --cf_val 1000 --freq_val 1000 --db_val 60
  
  # Generate all plots and save results
  python statistical_testing_bez_runs.py --bez_file data/bez_convolved.pkl --save_plots --save_results --output_dir results/
        """
    )

    # Required arguments
    parser.add_argument('--bez_file', type=str, required=True,
                       help='Path to BEZ convolved responses pickle file')

    # Optional data files
    parser.add_argument('--cochlea_file', type=str, default=None,
                       help='Path to cochlea convolved responses pickle file')

    # Analysis parameters
    parser.add_argument('--fiber_types', nargs='+', default=['hsr', 'msr', 'lsr'],
                       choices=['hsr', 'msr', 'lsr'],
                       help='Fiber types to analyze (default: all)')

    parser.add_argument('--correlation_type', type=str, default='pearson',
                       choices=['pearson', 'spearman'],
                       help='Type of correlation to compute (default: pearson)')

    parser.add_argument('--outlier_threshold', type=float, default=2.0,
                       help='Z-score threshold for outlier detection (default: 2.0)')

    parser.add_argument('--tr', type=float, default=0.1,
                       help='Temporal resolution in seconds (default: 0.1)')

    # Specific condition analysis
    parser.add_argument('--fiber_type', type=str, default=None,
                       choices=['hsr', 'msr', 'lsr'],
                       help='Analyze specific fiber type only')

    parser.add_argument('--cf_val', type=float, default=None,
                       help='Analyze specific CF value only')

    parser.add_argument('--freq_val', type=float, default=None,
                       help='Analyze specific frequency value only')

    parser.add_argument('--db_val', type=float, default=None,
                       help='Analyze specific dB level only')

    # Output options
    parser.add_argument('--output_dir', type=str, default='statistical_results',
                       help='Output directory for results and plots (default: statistical_results)')

    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to output directory')

    parser.add_argument('--save_results', action='store_true',
                       help='Save analysis results to pickle files')

    parser.add_argument('--plot_format', type=str, default='png',
                       choices=['png', 'pdf', 'svg'],
                       help='Format for saved plots (default: png)')

    parser.add_argument('--figsize', nargs=2, type=float, default=[12, 8],
                       help='Figure size for plots (width height, default: 12 8)')

    # Analysis modes
    parser.add_argument('--analyze_single_condition', action='store_true',
                       help='Analyze single condition (requires --fiber_type, --cf_val, --freq_val, --db_val)')

    parser.add_argument('--analyze_cochlea_correlations', action='store_true',
                       help='Analyze cochlea-BEZ correlations (requires --cochlea_file)')

    parser.add_argument('--analyze_all_conditions', action='store_true',
                       help='Analyze all available conditions')

    # Display options
    parser.add_argument('--show_plots', action='store_true', default=True,
                       help='Display plots interactively (default: True)')

    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    # Validate arguments
    if args.analyze_single_condition:
        required_args = [args.fiber_type, args.cf_val, args.freq_val, args.db_val]
        if any(arg is None for arg in required_args):
            parser.error("--analyze_single_condition requires --fiber_type, --cf_val, --freq_val, and --db_val")

    if args.analyze_cochlea_correlations and args.cochlea_file is None:
        parser.error("--analyze_cochlea_correlations requires --cochlea_file")

    # Create output directory
    if args.save_plots or args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")

    # Generate timestamp for this analysis session
    timestamp = generate_timestamp()
    print(f"Analysis timestamp: {timestamp}")

    # Load BEZ data
    print("\n=== Loading BEZ Data ===")
    bez_convolved = load_convolved_responses(args.bez_file)

    # Load cochlea data if provided
    cochlea_convolved = None
    if args.cochlea_file:
        print("\n=== Loading Cochlea Data ===")
        cochlea_convolved = load_cochlea_convolved_responses(args.cochlea_file)

    # Analysis execution
    if args.analyze_single_condition:
        print("\n=== Single Condition Analysis ===")
        results = analyze_condition_statistics(
            bez_convolved, args.fiber_type, args.cf_val, args.freq_val, args.db_val,
            correlation_type=args.correlation_type,
            outlier_threshold=args.outlier_threshold
        )

        if results:
            # Plot results
            plot_correlation_results(
                results, tr=args.tr, figsize=tuple(args.figsize),
                save_plots=args.save_plots, output_dir=args.output_dir,
                timestamp=timestamp, plot_format=args.plot_format
            )

            # Additional detailed plots
            plot_pairwise_correlation_matrix(
                results['pairwise']['correlation_matrix'],
                valid_run_indices=results['data']['valid_run_indices'],
                condition_info=results['condition'],
                figsize=(8, 6),
                save_plot=args.save_plots,
                output_dir=args.output_dir,
                timestamp=timestamp
            )

            # Get cochlea correlation if available
            cochlea_correlation = None
            if cochlea_convolved:
                cochlea_response = extract_cochlea_data(
                    cochlea_convolved, args.fiber_type, args.cf_val, args.freq_val, args.db_val
                )
                if cochlea_response is not None:
                    bez_runs, _ = extract_run_data(
                        bez_convolved, args.fiber_type, args.cf_val, args.freq_val, args.db_val
                    )
                    cochlea_correlation, _, _ = correlate_cochlea_with_bez_mean(
                        cochlea_response, bez_runs, args.correlation_type
                    )

            plot_pairwise_correlation_distribution(
                results['pairwise']['correlation_matrix'],
                condition_info=results['condition'],
                figsize=(10, 6),
                save_plot=args.save_plots,
                output_dir=args.output_dir,
                timestamp=timestamp,
                cochlea_correlation=cochlea_correlation,
                show_cochlea=cochlea_convolved is not None
            )

            # Save results
            if args.save_results:
                results_file = os.path.join(
                    args.output_dir,
                    f'single_condition_results_{args.fiber_type}_{args.cf_val}Hz_{args.freq_val}Hz_{args.db_val}dB_{timestamp}.pkl'
                )
                with open(results_file, 'wb') as f:
                    pickle.dump(results, f)
                print(f"Results saved to: {results_file}")

    elif args.analyze_cochlea_correlations:
        print("\n=== Cochlea-BEZ Correlation Analysis ===")
        cochlea_results = analyze_cochlea_bez_correlations(
            args.cochlea_file, args.bez_file,
            fiber_types=args.fiber_types,
            correlation_type=args.correlation_type,
            output_dir=args.output_dir if args.save_results else None
        )

        # Print summary statistics
        print("\n=== Cochlea-BEZ Correlation Summary ===")
        for fiber_type in args.fiber_types:
            if fiber_type in cochlea_results['mean_correlations']:
                correlations = []
                significant_count = 0
                total_count = 0

                for cf_val in cochlea_results['mean_correlations'][fiber_type]:
                    for freq_val in cochlea_results['mean_correlations'][fiber_type][cf_val]:
                        for db_val in cochlea_results['mean_correlations'][fiber_type][cf_val][freq_val]:
                            result = cochlea_results['mean_correlations'][fiber_type][cf_val][freq_val][db_val]
                            if not np.isnan(result['correlation']):
                                correlations.append(result['correlation'])
                                total_count += 1
                                if result['p_value'] < 0.05:
                                    significant_count += 1

                if correlations:
                    print(f"\n{fiber_type.upper()} fibers:")
                    print(f"  Total conditions: {total_count}")
                    print(f"  Significant correlations (p<0.05): {significant_count} ({100*significant_count/total_count:.1f}%)")
                    print(f"  Mean correlation: {np.mean(correlations):.3f} ± {np.std(correlations):.3f}")
                    print(f"  Range: [{np.min(correlations):.3f}, {np.max(correlations):.3f}]")

    elif args.analyze_all_conditions:
        print("\n=== Analyzing All Conditions ===")
        all_results = {}

        # Count total conditions first
        total_conditions = 0
        for fiber_type in args.fiber_types:
            if fiber_type in bez_convolved:
                for cf_val in bez_convolved[fiber_type]:
                    for freq_val in bez_convolved[fiber_type][cf_val]:
                        for db_val in bez_convolved[fiber_type][cf_val][freq_val]:
                            total_conditions += 1

        print(f"Found {total_conditions} conditions to analyze")

        processed = 0
        for fiber_type in args.fiber_types:
            if fiber_type not in bez_convolved:
                continue

            all_results[fiber_type] = {}

            for cf_val in bez_convolved[fiber_type]:
                all_results[fiber_type][cf_val] = {}

                for freq_val in bez_convolved[fiber_type][cf_val]:
                    all_results[fiber_type][cf_val][freq_val] = {}

                    for db_val in bez_convolved[fiber_type][cf_val][freq_val]:
                        processed += 1

                        if args.verbose:
                            print(f"Processing {processed}/{total_conditions}: {fiber_type} CF={cf_val}Hz, Freq={freq_val}Hz, dB={db_val}")

                        results = analyze_condition_statistics(
                            bez_convolved, fiber_type, cf_val, freq_val, db_val,
                            correlation_type=args.correlation_type,
                            outlier_threshold=args.outlier_threshold
                        )

                        all_results[fiber_type][cf_val][freq_val][db_val] = results

        print(f"Completed analysis of {processed} conditions")

        # Save all results
        if args.save_results:
            all_results_file = os.path.join(args.output_dir, f'all_conditions_results_{timestamp}.pkl')
            with open(all_results_file, 'wb') as f:
                pickle.dump(all_results, f)
            print(f"All results saved to: {all_results_file}")

    else:
        print("\nNo analysis mode specified. Use one of:")
        print("  --analyze_single_condition")
        print("  --analyze_cochlea_correlations")
        print("  --analyze_all_conditions")
        print("\nUse --help for more information.")

    print(f"\nAnalysis completed at {timestamp}")


if __name__ == "__main__":
    main()