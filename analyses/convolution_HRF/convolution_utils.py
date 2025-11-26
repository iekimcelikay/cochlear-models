"""
Utilities for HRF convolution analysis scripts.
Contains common functions shared between different convolution analysis workflows.

Created: 2025-10-23
"""

import os
import pickle
import numpy as np
import nipy.modalities.fmri.hemodynamic_models
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
from subcorticalSTRF.compare_bez_cochlea_mainscript import convert_to_numeric  # Reuse existing function


def generate_canonical_hrf(tr=0.1, oversampling=1, time_length=32.0, onset=0.0):
    """
    Generate canonical HRF using the Glover model.

    Parameters:
    - tr: scan repeat time in seconds
    - oversampling: temporal oversampling factor
    - time_length: HRF kernel length in seconds
    - onset: onset of the response

    Returns:
    - canonical_hrf: the generated HRF array
    - time_points: corresponding time axis
    """
    canonical_hrf = nipy.modalities.fmri.hemodynamic_models.glover_hrf(
        tr=tr, oversampling=oversampling, time_length=time_length, onset=onset
    )

    time_points = np.arange(len(canonical_hrf)) * tr
    time_points += onset

    return canonical_hrf, time_points


def convolve_brief_stimulus_with_hrf(psth_data, stimulus_duration=0.2, hrf_duration=32.0, tr=0.1):
    """
    Convolve brief PSTH data with full HRF response.

    Parameters:
    - psth_data: 200ms PSTH array
    - stimulus_duration: Duration of PSTH data (0.2 seconds)
    - hrf_duration: Duration of HRF in seconds
    - tr: Temporal resolution (0.1 seconds)

    Returns:
    - convolved_response: Full BOLD response prediction
    - time_vector: Time axis for the response
    """
    # Create time vector for full response
    time_vector = np.arange(0, stimulus_duration, tr)
    dt = tr
    psth_data = np.atleast_1d(psth_data).flatten()

    # Generate full HRF
    hrf, _ = generate_canonical_hrf(tr=tr, time_length=hrf_duration)

    # Convolve to get BOLD response
    convolved_response = np.convolve(psth_data, hrf, mode='full')
    #convolved_response = convolved_response * dt  # Scale by time resolution

    return convolved_response, time_vector


def save_plot(plot_name, output_dir, timestamp, model_name=None, fiber_type=None, plot_format='png'):
    """
    Save the current matplotlib plot to file with timestamp and model info.

    Parameters:
    - plot_name: Base name for the plot (e.g., 'canonical_hrf', 'hrf_responses')
    - output_dir: Directory to save plots
    - timestamp: Timestamp string for filename
    - model_name: Model name for filename (optional)
    - fiber_type: Fiber type for filename (optional)
    - plot_format: File format ('png', 'pdf', 'svg')
    """
    # Create plots subdirectory
    plots_dir = Path(output_dir) / 'plots'
    os.makedirs(plots_dir, exist_ok=True)

    # Build filename with model and fiber info
    filename_parts = []
    if model_name:
        filename_parts.append(model_name.upper())
    filename_parts.append(plot_name)
    if fiber_type:
        filename_parts.append(fiber_type.lower())
    filename_parts.append(timestamp)

    filename = '_'.join(filename_parts) + f'.{plot_format}'
    filepath = plots_dir / filename

    # Save the plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {filepath}")

    return str(filepath)



def validate_psth_data(psth_data, expected_length=200, fiber_type=None, 
                      cf_val=None, freq_val=None, db_val=None, run_idx=None):
    """
    Validate PSTH data array for convolution.

    Parameters:
    - psth_data: PSTH array to validate
    - expected_length: Expected length of PSTH (default: 200 for 200ms)
    - fiber_type, cf_val, freq_val, db_val, run_idx: Parameters for error reporting

    Returns:
    - is_valid: Boolean indicating if data is valid
    - processed_data: Processed PSTH array (flattened and cleaned)
    """
    try:
        if psth_data is None:
            return False, None

        # Convert to numpy array and flatten
        if isinstance(psth_data, np.ndarray):
            psth_array = np.atleast_1d(psth_data).flatten()
        else:
            psth_array = np.array(psth_data).flatten()

        # Check length
        if len(psth_array) != expected_length:
            if any(x is not None for x in [fiber_type, cf_val, freq_val, db_val, run_idx]):
                print(f"Warning: Unexpected PSTH length {len(psth_array)} (expected {expected_length}) "
                      f"for {fiber_type or 'unknown'} CF={cf_val}Hz, Freq={freq_val}Hz, "
                      f"dB={db_val}, Run={run_idx}")
            return False, None

        # Check for finite values
        if not np.any(np.isfinite(psth_array)):
            if any(x is not None for x in [fiber_type, cf_val, freq_val, db_val, run_idx]):
                print(f"Warning: No finite values for {fiber_type or 'unknown'} "
                      f"CF={cf_val}Hz, Freq={freq_val}Hz, dB={db_val}, Run={run_idx}")
            return False, None

        return True, psth_array

    except Exception as e:
        if any(x is not None for x in [fiber_type, cf_val, freq_val, db_val, run_idx]):
            print(f"Warning: Error validating PSTH data for {fiber_type or 'unknown'} "
                  f"CF={cf_val}Hz, Freq={freq_val}Hz, dB={db_val}, Run={run_idx}: {e}")
        return False, None


def plot_hrf(canonical_hrf, time_points, figsize=(10, 6), save_plots=False, 
             output_dir=None, timestamp=None, plot_format='png'):
    """
    Plot the canonical HRF.

    Parameters:
    - canonical_hrf: HRF array to plot
    - time_points: corresponding time axis
    - figsize: figure size tuple
    - save_plots: whether to save the plot to file
    - output_dir: directory to save plots
    - timestamp: timestamp for filename
    - plot_format: format for saved plots
    """
    plt.figure(figsize=figsize)
    plt.plot(time_points, canonical_hrf, 'b-', linewidth=2, label='Canonical HRF (Glover)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Response amplitude')
    plt.title('Canonical Hemodynamic Response Function')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save plot if requested
    if save_plots and output_dir and timestamp:
        save_plot('canonical_hrf', output_dir, timestamp, plot_format=plot_format)

    plt.show()


def save_convolved_data(convolved_responses, output_dir, timestamp=None, model_name=None):
    """
    Save convolved responses to files with timestamps.

    Parameters:
    - convolved_responses: Dictionary with convolved data
    - output_dir: Directory to save files
    - timestamp: Optional timestamp string (if None, generates current timestamp)
    - model_name: Optional model name for filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Add model name to filename if provided
    model_prefix = f"{model_name}_" if model_name else ""

    for fiber_type in convolved_responses:
        output_file = os.path.join(output_dir, f'{model_prefix}convolved_responses_{fiber_type}_{timestamp}.pkl')

        with open(output_file, 'wb') as f:
            pickle.dump(convolved_responses[fiber_type], f)

        print(f"Saved {fiber_type.upper()} convolved responses to: {output_file}")


def generate_timestamp():
    """Generate timestamp string for filenames."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def get_frequency_colors(frequencies, colormap='viridis'):
    """
    Generate consistent color mapping for frequencies.

    Parameters:
    - frequencies: List or array of frequency values
    - colormap: Matplotlib colormap name

    Returns:
    - freq_colors: Dictionary mapping frequencies to colors
    """
    import matplotlib.cm as cm
    
    frequencies = sorted(frequencies)
    n_freqs = len(frequencies)

    # Use appropriate colormap based on number of frequencies
    if n_freqs <= 20 and colormap == 'viridis':
        colors = cm.tab20(np.linspace(0, 1, n_freqs))
    else:
        cmap = getattr(cm, colormap)
        colors = cmap(np.linspace(0, 1, n_freqs))

    return {freq: colors[i] for i, freq in enumerate(frequencies)}
