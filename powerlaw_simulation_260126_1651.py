# Created on: 26.01.2026, 16:51
# In this script ======
#   I will load the firing rates for each CF, which gives me the AN output.
#   Then I will apply power function, - imitates the sharpening of tuning curves
# after AN stages, in subcortical.
# I want to find the n that gives me an optimum sharpening profile.
# It might be interesting to compare this with a lateral inhibition operation
# Idk though.
# Checking different papers to identify the steps they apply after AN simulation
# can be a good idea.
# Created by: Ekim Celikay

# 1. Apply power function
# 2. Re-normalize (divide by pooled activity)

# ==== POWER-LAW NONLINEARITY ====
#   - Applying a power function sharpens tuning curves by:
#       - Suppressing weak responses more than strong responses.
#       - Enhancing the peak relative to flanks.
#       - It's a pointwise nonlinearity, meaning that each channel is
#           processed independently.
# Power-law (n>1) enhances contrast by expanding the dynamic range.
# r_powered = r^n, where n>1
# Then you need to re-normalize:
# r_normalized = r_powered / (σ + Σ_j w_ij * r_j^n) (divide by pooled activity)
# or sometimes:
# r_normalized = r_powered / (σ + (Σ_j w_ij * r_j^n)^(1/n))
# Power function: Enhances strong responses, suppresses weak ones
# Normalization: Divides by the pooled activity of neighbors, creating gain control
# Net effect: Strong lateral inhibition - responses are suppressed based on overall population activity



# imports
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import logging
from pathlib import Path

# Project imports
from utils.misc_functions import find_latest_results_folder
from utils.result_saver import ResultSaver

# Define paths
# Add parent directory to path
sys.path.insert(0, str(Path.cwd().parent))

input_dir = "./models_output/cochlea_test015_approximate"

AN_firingrates_dir = find_latest_results_folder(input_dir)

# Load one file here
results_folder = AN_firingrates_dir

saver = ResultSaver(results_folder)

# Find all .npz files
npz_files = list(results_folder.glob('*.npz'))
print(f"Found {len(npz_files)} .npz files")

results = {}

for npz_file in npz_files:
    # Parse filename to extract freq and db
    # Format: test011_freq_125.0hz_db_60.npz
    parts = npz_file.stem.split('_')
    freq_idx = parts.index('freq') + 1
    db_idx = parts.index('db') + 1

    freq = float(parts[freq_idx].replace('hz', ''))
    db = float(parts[db_idx])

    # Load data
    data = saver.load_npz(npz_file.name)

    # Store with freq as key
    if freq not in results:
        results[freq] = {}
    results[freq][db] = data

    print(f"Loaded {len(results)} frequencies")

    first_freq = list(results.keys())[0]
    first_db = list(results[first_freq].keys())[0]
    cf_list = results[first_freq][first_db]['cf_list']
    mean_rates = results[first_freq][first_db]['mean_rates']

    population_results = {}
    hsr_weight = 0.65
    msr_weight = 0.28
    lsr_weight = 0.12


    for freq, db_dict in results.items():
        population_results[freq] = {}

        for db, data in db_dict.items():
            # Extract mean rates (it's a dictionary wrapped in numpy array)
            mean_rates = data['mean_rates'].item()

            # Get individual fiber types

            mean_rates_hsr = mean_rates['hsr']
            mean_rates_msr = mean_rates['msr']
            mean_rates_lsr = mean_rates['lsr']

            # Compute population response
            population_mean = (hsr_weight * mean_rates_hsr +
                                msr_weight * mean_rates_msr +
                                lsr_weight * mean_rates_lsr
                                )

            population_results[freq][db] = population_mean


###

# ================= Function drafts ====================

def load_cochlea_results(input_dir, weights=None):
    """Load cochlea simulation results and compute population responses.
    Usage:

    results, population_results, cf_list = load_cochlea_results(input_dir)

    Args:
        input_dir: Path to directory containing cochlea results
        weights: Dict with 'hsr', 'msr', 'lsr' weights. If None, uses defaults.

    Returns:
        results: Dict[freq][db] -> raw data from .npz files
        population_results: Dict[freq][db] -> population mean rates array (num_cf,)
        cf_list: Array of CF values
    """
    # Default weights
    if weights is None:
        weights = {
            'hsr': 0.65,
            'msr': 0.28,
            'lsr': 0.12
        }

    # Find latest results folder
    AN_firingrates_dir = find_latest_results_folder(input_dir)
    results_folder = AN_firingrates_dir
    saver = ResultSaver(results_folder)

    # Find all .npz files
    npz_files = list(results_folder.glob('*.npz'))
    print(f"Found {len(npz_files)} .npz files")

    results = {}

    # Load all data files
    for npz_file in npz_files:
        # Parse filename to extract freq and db
        # Format: test011_freq_125.0hz_db_60.npz
        parts = npz_file.stem.split('_')
        freq_idx = parts.index('freq') + 1
        db_idx = parts.index('db') + 1

        freq = float(parts[freq_idx].replace('hz', ''))
        db = float(parts[db_idx])

        # Load data
        data = saver.load_npz(npz_file.name)

        # Store with freq as key
        if freq not in results:
            results[freq] = {}
        results[freq][db] = data

    print(f"Loaded {len(results)} frequencies")

    # Extract CF list from first file
    first_freq = list(results.keys())[0]
    first_db = list(results[first_freq].keys())[0]
    cf_list = results[first_freq][first_db]['cf_list']

    # Compute population responses
    population_results = {}

    for freq, db_dict in results.items():
        population_results[freq] = {}

        for db, data in db_dict.items():
            # Extract mean rates (it's a dictionary wrapped in numpy array)
            mean_rates = data['mean_rates'].item()

            # Get individual fiber types
            mean_rates_hsr = mean_rates['hsr']
            mean_rates_msr = mean_rates['msr']
            mean_rates_lsr = mean_rates['lsr']

            # Compute weighted population response
            population_mean = (weights['hsr'] * mean_rates_hsr +
                             weights['msr'] * mean_rates_msr +
                             weights['lsr'] * mean_rates_lsr)

            population_results[freq][db] = population_mean

    return results, population_results, cf_list

def organize_for_eachtone_allCFs(population_results, cf_list, target_db):
    """Organize population results into a 2D spectrogram matrix for a specific dB level.

    Args:
        population_results: Dict[freq][db] -> array of firing rates (num_cf,)
        cf_list: Array of CF values (num_cf,)
        target_db: dB level to extract

    Returns:
        response_matrix: 2D array of shape (num_cf, num_tones)
                        Rows = CF channels, Columns = tone frequencies
        tone_freqs: Sorted array of tone frequencies used as stimuli
    """
    # Get all tone frequencies that have the target dB level
    tone_freqs = []
    for freq, db_dict in population_results.items():
        if target_db in db_dict:
            tone_freqs.append(freq)

    # Sort tone frequencies
    tone_freqs = np.array(sorted(tone_freqs))

    # Initialize response matrix: (num_cf, num_tones)
    num_cf = len(cf_list)
    num_tones = len(tone_freqs)
    response_matrix = np.zeros((num_cf, num_tones))

    # Fill the matrix
    for i_tone, freq in enumerate(tone_freqs):
        response_matrix[:, i_tone] = population_results[freq][target_db]

    print(f"Organized matrix shape: {response_matrix.shape}")
    print(f"  - {num_cf} CF channels")
    print(f"  - {num_tones} tone frequencies")
    print(f"  - dB level: {target_db}")

    return response_matrix, tone_freqs

def plot_spectrogram_forCFS(cf_list, tone_freqs, response_matrix, db_level, save_path=None):
    """Plot spectrogram showing CF responses to different tone frequencies.

    Args:
        cf_list: Array of CF values (Hz)
        tone_freqs: Array of tone frequencies (Hz)
        response_matrix: 2D array of shape (num_cf, num_tones) with firing rates
        db_level: dB level of stimuli (for title)
        save_path: Optional path to save figure

    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Transpose matrix to have CFs on x-axis
    # Original: (num_cf, num_tones), Transposed: (num_tones, num_cf)
    response_matrix_T = response_matrix.T

    # Create heatmap with CFs on x-axis
    im = ax.imshow(response_matrix_T,
                   aspect='auto',
                   origin='lower',
                   cmap='viridis',
                   interpolation='nearest',
                   extent=[cf_list[0], cf_list[-1], tone_freqs[0], tone_freqs[-1]])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Firing Rate (spikes/s)')

    # Labels and title
    ax.set_xlabel('CF (Hz)', fontsize=12)
    ax.set_ylabel('Tone Frequency (Hz)', fontsize=12)
    ax.set_title(f'Cochlear Spectrogram - Population Response at {db_level} dB SPL',
                 fontsize=14, fontweight='bold')

    # Set log scale for better visualization
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set ticks to show all values
    ax.set_xticks(cf_list)
    ax.set_yticks(tone_freqs)

    # Format tick labels to show actual Hz values
    from matplotlib.ticker import FuncFormatter
    def hz_formatter(x, pos):
        if x >= 1000:
            return f'{x/1000:.1f}k'
        else:
            return f'{int(x)}'

    ax.xaxis.set_major_formatter(FuncFormatter(hz_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(hz_formatter))

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, ax

def plot_tuning_curves(cf_list, tone_freqs, response_matrix, db_level, save_path=None):
    """Plot tuning curves for each CF channel in subplots.

    Args:
        cf_list: Array of CF values (Hz)
        tone_freqs: Array of tone frequencies (Hz)
        response_matrix: 2D array of shape (num_cf, num_tones) with firing rates
        db_level: dB level of stimuli (for suptitle)
        save_path: Optional path to save figure

    Returns:
        fig, axes: Matplotlib figure and axes array
    """
    num_cf = len(cf_list)

    # Create subplot grid (8 rows x 5 columns for 40 CFs)
    n_rows = 8
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 24))
    axes = axes.flatten()

    # Plot tuning curve for each CF
    for i_cf, cf in enumerate(cf_list):
        ax = axes[i_cf]

        # Get firing rates for this CF across all tone frequencies
        tuning_curve = response_matrix[i_cf, :]

        # Plot tuning curve
        ax.plot(tone_freqs, tuning_curve, 'o-', linewidth=2, markersize=4, color='steelblue')

        # Mark the CF on the plot
        ax.axvline(cf, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label=f'CF={cf:.0f} Hz')

        # Labels and formatting
        ax.set_xscale('log')
        ax.set_xlabel('Tone Freq (Hz)', fontsize=8)
        ax.set_ylabel('Rate (sp/s)', fontsize=8)
        ax.set_title(f'CF = {cf:.0f} Hz', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=7)

        # Format x-axis tick labels
        from matplotlib.ticker import FuncFormatter
        def hz_formatter(x, pos):
            if x >= 1000:
                return f'{x/1000:.1f}k'
            else:
                return f'{int(x)}'
        ax.xaxis.set_major_formatter(FuncFormatter(hz_formatter))

    # Overall title
    fig.suptitle(f'Tuning Curves for All CF Channels at {db_level} dB SPL',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.995])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes

def plot_AN_tuning_curves():
    return

def power_function_sharpening():
    return

def plot_power_sharpened_curves():
    return


# Testing
