# This is the second script created for convolution analysis.
# created: 23.10.2025
# IT is after the 1-1 meeting, where I discussed with Alex about the HRF convolution analysis.
# In this script, I will perform convolution with HRF for each run of BEZ model simulation separately.

import os
import pickle
import numpy as np
import nipy.modalities.fmri.hemodynamic_models
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path
import argparse
from convolution_utils import (
    generate_canonical_hrf,
    convolve_brief_stimulus_with_hrf,
    save_plot,
    validate_psth_data,
    convert_to_numeric,
    save_convolved_data,
    generate_timestamp,
    plot_hrf,
    get_frequency_colors
)

def load_bez_psth_data():
    """
    Load BEZ model PSTH data from MATLAB file psth_data_128fibers.mat.
    The data contains hsr_all, lsr_all, msr_all as 4D cell arrays.

    Returns:
    - bez_data: Dictionary containing PSTH data for all fiber types
    - parameters: Dictionary with CFs, frequencies, and dB levels
    """
    # Load BEZ model data from psth_data_128fibers.mat
    # Use proper path resolution
    current_dir = Path(__file__).parent
    
    # Try multiple possible locations
    possible_paths = [
        # Relative to convolution_HRF directory
        current_dir.parent / 'BEZ2018_meanrate' / 'results' / 
        'processed_data' / 'psth_data_128fibers.mat',
        
        # Absolute path from project root
        Path('/home/ekim/PycharmProjects/phd_firstyear/subcorticalSTRF/BEZ2018_meanrate/results/processed_data/psth_data_128fibers.mat'),
        
        # Try from audio_periph_models
        Path('/home/ekim/audio_periph_models/BEZ2018a_model/results/processed_data/psth_data_128fibers.mat'),
    ]
    
    # Find the first existing path
    bez_file = None
    for path in possible_paths:
        if path.exists():
            bez_file = path
            break
    
    if bez_file is None:
        raise FileNotFoundError(
            f"Could not find psth_data_128fibers.mat in any of:\n" +
            '\n'.join(f"  - {p}" for p in possible_paths) +
            "\n\nPlease check the file location."
        )

    print(f"Loading BEZ PSTH data from: {bez_file}")
    bez_data = loadmat(
        str(bez_file), struct_as_record=False, squeeze_me=True
    )
    print("✓ BEZ PSTH data loaded successfully")

    # Extract parameters

    parameters = {
        'cfs': np.sort(convert_to_numeric(bez_data['cfs'])),
        'frequencies': np.sort(convert_to_numeric(bez_data['frequencies'])),
        'dbs': np.sort(convert_to_numeric(bez_data['dbs']))
    }

    print(f"Parameters extracted:")
    print(f"  CFs: {len(parameters['cfs'])} values")
    print(f"  Frequencies: {len(parameters['frequencies'])} values")
    print(f"  dB levels: {parameters['dbs']}")

    # Verify the data structure
    print(f"Data structure:")
    print(f"  hsr_all shape: {bez_data['hsr_all'].shape}")
    print(f"  msr_all shape: {bez_data['msr_all'].shape}")
    print(f"  lsr_all shape: {bez_data['lsr_all'].shape}")

    return bez_data, parameters


def convolve_bez_runs_with_hrf(bez_data, parameters, fiber_types=['hsr', 'msr', 'lsr'],
                               tr=0.1, hrf_length=32.0):
    """
    Convolve each run of BEZ PSTH data separately with canonical HRF.

    Parameters:
    - bez_data: Loaded BEZ MATLAB data structure from psth_data_128fibers.mat
    - parameters: Parameter dictionary with CFs, frequencies, and dB levels
    - fiber_types: List of fiber types to process
    - tr: Temporal resolution in seconds
    - hrf_length: HRF duration in seconds

    Returns:
    - convolved_responses: Dictionary with HRF-convolved responses for each run
    - hrf: The canonical HRF used
    - time_points: Time axis for HRF
    """
    # Generate canonical HRF
    hrf, time_points = generate_canonical_hrf(tr=tr, time_length=hrf_length)

    # Get number of runs from data structure
    # Get dimensions from one of the fiber type arrays
    sample_data = bez_data['hsr_all']
    n_cfs, n_freqs, n_dbs, n_runs = sample_data.shape
    print(f"Processing {n_runs} runs for each condition...")
    print(f"Data dimensions: {n_cfs} CFs x {n_freqs} frequencies x {n_dbs} dB levels x {n_runs} runs")

    # Extract PSTH data for each fiber type - correct field names
    bez_rates = {
        'lsr': bez_data['lsr_all'],
        'msr': bez_data['msr_all'],
        'hsr': bez_data['hsr_all']
    }

    convolved_responses = {}

    for fiber_type in fiber_types:
        print(f"\nProcessing {fiber_type.upper()} fibers...")
        convolved_responses[fiber_type] = {}

        psth_data = bez_rates[fiber_type]

        # Process each CF
        for cf_idx, cf_val in enumerate(parameters['cfs']):
            convolved_responses[fiber_type][cf_val] = {}

            # Process each tone frequency
            for freq_idx, freq_val in enumerate(parameters['frequencies']):
                convolved_responses[fiber_type][cf_val][freq_val] = {}

                # Process each dB level
                for db_idx, db_val in enumerate(parameters['dbs']):
                    convolved_responses[fiber_type][cf_val][freq_val][db_val] = {}

                    # Process each run separately
                    for run_idx in range(n_runs):
                        try:
                            # Extract PSTH for this specific run from 4D cell array
                            psth_cell = psth_data[cf_idx, freq_idx, db_idx, run_idx]

                            # Validate PSTH data using utility function
                            is_valid, psth_array = validate_psth_data(
                                psth_cell, expected_length=200,
                                fiber_type=fiber_type, cf_val=cf_val,
                                freq_val=freq_val, db_val=db_val, run_idx=run_idx
                            )

                            if is_valid:
                                # Convolve this run's PSTH with HRF
                                convolved, time_vec = convolve_brief_stimulus_with_hrf(
                                    psth_array, stimulus_duration=0.2,
                                    hrf_duration=hrf_length, tr=tr
                                )
                                convolved_responses[fiber_type][cf_val][freq_val][db_val][run_idx] = convolved
                            else:
                                convolved_responses[fiber_type][cf_val][freq_val][db_val][run_idx] = None

                        except (IndexError, ValueError, AttributeError) as e:
                            print(f"Warning: Error processing {fiber_type.upper()} CF={cf_val}Hz, "
                                  f"Freq={freq_val}Hz, dB={db_val}, Run={run_idx}: {e}")
                            convolved_responses[fiber_type][cf_val][freq_val][db_val][run_idx] = None
                            continue

            # Progress indicator
            print(f"✓ Completed CF {cf_idx + 1}/{len(parameters['cfs'])} for {fiber_type.upper()}")

        print(f"✓ {fiber_type.upper()} convolution complete")

    return convolved_responses, hrf, time_points


def plot_convolved_runs(convolved_responses, parameters, fiber_type='hsr', cf_val=None,
                       freq_val=None, db_val=60.0, tr=0.1, figsize=(15, 10),
                       plot_individual_runs=True, plot_mean_and_sem=True,
                       max_runs_display=None, save_plots=False, output_dir=None,
                       timestamp=None, plot_format='png'):
    """
    Plot HRF-convolved responses for each run of BEZ simulation.

    Parameters:
    - convolved_responses: Dictionary with convolved data for each run
    - parameters: Dictionary with stimulus parameters
    - fiber_type: Fiber type to plot ('hsr', 'msr', 'lsr')
    - cf_val: Specific CF to plot (if None, uses first available)
    - freq_val: Specific frequency to plot (if None, plots all frequencies)
    - db_val: dB level to plot
    - tr: Temporal resolution for time axis
    - figsize: Figure size
    - plot_individual_runs: Whether to show individual run traces
    - plot_mean_and_sem: Whether to show mean ± SEM across runs
    - max_runs_display: Maximum number of individual runs to display (None for all)
    - save_plots: Whether to save the plot to file
    - output_dir: Directory to save plots
    - timestamp: Timestamp for filename
    - plot_format: Format for saved plots
    """

    if fiber_type not in convolved_responses:
        print(f"No data for fiber type: {fiber_type}")
        return

    fiber_data = convolved_responses[fiber_type]

    # Select CF to plot
    if cf_val is None:
        cf_val = list(fiber_data.keys())[0]
        print(f"Using CF = {cf_val}Hz (first available)")

    if cf_val not in fiber_data:
        print(f"CF {cf_val}Hz not found in data")
        return

    cf_data = fiber_data[cf_val]

    # Determine frequencies to plot
    if freq_val is None:
        frequencies_to_plot = sorted(parameters['frequencies'])
        plot_title_suffix = "All Frequencies"
    else:
        frequencies_to_plot = [freq_val]
        plot_title_suffix = f"Frequency {freq_val}Hz"

    # Create color mapping for frequencies
    freq_colors = get_frequency_colors(frequencies_to_plot)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    for freq_idx, freq in enumerate(frequencies_to_plot):
        if freq not in cf_data or db_val not in cf_data[freq]:
            continue

        run_data = cf_data[freq][db_val]

        # Collect valid runs
        valid_runs = []
        run_indices = []

        for run_idx, convolved_response in run_data.items():
            if convolved_response is not None:
                valid_runs.append(convolved_response)
                run_indices.append(run_idx)

        if not valid_runs:
            print(f"No valid runs for frequency {freq}Hz")
            continue

        # Create time axis
        max_length = max(len(run) for run in valid_runs)
        time_axis = np.arange(max_length) * tr

        # Pad shorter runs with NaN
        padded_runs = []
        for run in valid_runs:
            if len(run) < max_length:
                padded_run = np.full(max_length, np.nan)
                padded_run[:len(run)] = run
                padded_runs.append(padded_run)
            else:
                padded_runs.append(run)

        runs_array = np.array(padded_runs)

        # Plot individual runs (with transparency and limited number)
        if plot_individual_runs and len(valid_runs) > 0:
            runs_to_display = valid_runs
            if max_runs_display is not None:
                runs_to_display = valid_runs[:max_runs_display]

            for i, run in enumerate(runs_to_display):
                run_time_axis = np.arange(len(run)) * tr
                alpha = 0.3 if plot_mean_and_sem else 0.7
                linewidth = 0.5 if plot_mean_and_sem else 1.0

                label = f'Freq {freq}Hz - Run {run_indices[i]}' if not plot_mean_and_sem else None
                ax.plot(run_time_axis, run, color=freq_colors[freq], alpha=alpha,
                       linewidth=linewidth, label=label)

        # Plot mean and SEM across runs
        if plot_mean_and_sem and len(valid_runs) > 1:
            # Calculate mean and SEM across runs
            mean_response = np.nanmean(runs_array, axis=0)
            sem_response = np.nanstd(runs_array, axis=0) / np.sqrt(np.sum(~np.isnan(runs_array), axis=0))

            # Plot mean line
            ax.plot(time_axis, mean_response, color=freq_colors[freq], linewidth=2.5,
                   label=f'Freq {freq}Hz (n={len(valid_runs)} runs)')

            # Plot SEM as shaded area
            ax.fill_between(time_axis, mean_response - sem_response, mean_response + sem_response,
                           color=freq_colors[freq], alpha=0.2)

        elif len(valid_runs) == 1:
            # Only one run available
            single_run = valid_runs[0]
            single_time_axis = np.arange(len(single_run)) * tr
            ax.plot(single_time_axis, single_run, color=freq_colors[freq], linewidth=2.5,
                   label=f'Freq {freq}Hz (1 run)')

    # Formatting
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('HRF-Convolved Response', fontsize=12)
    ax.set_title(f'{fiber_type.upper()} Fibers - CF: {cf_val}Hz, {db_val}dB\n{plot_title_suffix}', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Legend
    if ax.get_legend_handles_labels()[0]:  # Check if there are any legend items
        n_freqs = len(frequencies_to_plot)
        if n_freqs <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        else:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)

    plt.tight_layout()

    # Save plot if requested
    if save_plots and output_dir and timestamp:
        plot_name = f'bez_runs_convolved_{fiber_type}_{cf_val}Hz_{db_val}dB'
        save_plot(plot_name, output_dir, timestamp, model_name='BEZ',
                 fiber_type=fiber_type, plot_format=plot_format)

    plt.show()


def plot_runs_comparison_across_cfs(convolved_responses, parameters, fiber_type='hsr',
                                   freq_val=None, db_val=60.0, tr=0.1,
                                   max_cfs=6, figsize=(18, 12),
                                   save_plots=False, output_dir=None,
                                   timestamp=None, plot_format='png'):
    """
    Plot convolved responses across multiple CFs for comparison.
    Shows mean ± SEM for each CF in separate subplots.

    Parameters:
    - convolved_responses: Dictionary with convolved data for each run
    - parameters: Dictionary with stimulus parameters
    - fiber_type: Fiber type to plot
    - freq_val: Specific frequency to plot (if None, uses first frequency)
    - db_val: dB level to plot
    - tr: Temporal resolution
    - max_cfs: Maximum number of CFs to plot
    - figsize: Figure size
    - save_plots: Whether to save plot
    - output_dir: Directory to save plots
    - timestamp: Timestamp for filename
    - plot_format: Format for saved plots
    """

    if fiber_type not in convolved_responses:
        print(f"No data for fiber type: {fiber_type}")
        return

    fiber_data = convolved_responses[fiber_type]
    available_cfs = list(fiber_data.keys())[:max_cfs]

    if freq_val is None:
        freq_val = parameters['frequencies'][0]
        print(f"Using frequency = {freq_val}Hz (first available)")

    # Setup subplot grid
    n_cfs = len(available_cfs)
    ncols = min(3, n_cfs)
    nrows = int(np.ceil(n_cfs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, cf_val in enumerate(available_cfs):
        ax = axes[idx]

        if cf_val not in fiber_data or freq_val not in fiber_data[cf_val] or db_val not in fiber_data[cf_val][freq_val]:
            ax.text(0.5, 0.5, f'No data\nCF: {cf_val}Hz', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'CF: {cf_val}Hz')
            continue

        run_data = fiber_data[cf_val][freq_val][db_val]

        # Collect valid runs
        valid_runs = [convolved_response for convolved_response in run_data.values()
                     if convolved_response is not None]

        if not valid_runs:
            ax.text(0.5, 0.5, f'No valid runs\nCF: {cf_val}Hz', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'CF: {cf_val}Hz')
            continue

        # Create time axis and pad runs
        max_length = max(len(run) for run in valid_runs)
        time_axis = np.arange(max_length) * tr

        padded_runs = []
        for run in valid_runs:
            if len(run) < max_length:
                padded_run = np.full(max_length, np.nan)
                padded_run[:len(run)] = run
                padded_runs.append(padded_run)
            else:
                padded_runs.append(run)

        runs_array = np.array(padded_runs)

        # Plot individual runs with transparency
        for run in valid_runs:
            run_time_axis = np.arange(len(run)) * tr
            ax.plot(run_time_axis, run, color='blue', alpha=0.2, linewidth=0.8)

        # Plot mean and SEM
        if len(valid_runs) > 1:
            mean_response = np.nanmean(runs_array, axis=0)
            sem_response = np.nanstd(runs_array, axis=0) / np.sqrt(np.sum(~np.isnan(runs_array), axis=0))

            ax.plot(time_axis, mean_response, color='red', linewidth=2.5,
                   label=f'Mean (n={len(valid_runs)})')
            ax.fill_between(time_axis, mean_response - sem_response, mean_response + sem_response,
                           color='red', alpha=0.2)
        else:
            # Only one run
            single_run = valid_runs[0]
            single_time_axis = np.arange(len(single_run)) * tr
            ax.plot(single_time_axis, single_run, color='red', linewidth=2.5, label='Single run')

        ax.set_title(f'CF: {cf_val}Hz (n={len(valid_runs)})')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Response')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(n_cfs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'{fiber_type.upper()} Fibers - Freq: {freq_val}Hz, {db_val}dB\nMean ± SEM Across Runs',
                fontsize=16)
    plt.tight_layout()

    # Save plot if requested
    if save_plots and output_dir and timestamp:
        plot_name = f'bez_runs_comparison_{fiber_type}_{freq_val}Hz_{db_val}dB'
        save_plot(plot_name, output_dir, timestamp, model_name='BEZ',
                 fiber_type=fiber_type, plot_format=plot_format)

    plt.show()


def main():
    """Main execution function with command line arguments."""
    parser = argparse.ArgumentParser(description='Convolve BEZ PSTH data with canonical HRF - each run separately')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--fiber_types', nargs='+', default=['hsr', 'msr', 'lsr'],
                       help='Fiber types to process')
    parser.add_argument('--tr', type=float, default=0.1,
                       help='Temporal resolution (seconds)')
    parser.add_argument('--hrf_length', type=float, default=32.0,
                       help='HRF duration (seconds)')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to file')
    parser.add_argument('--show_hrf', action='store_true',
                       help='Show canonical HRF plot')

    args = parser.parse_args()

    # Generate timestamp for this run
    timestamp = generate_timestamp()
    print(f"Starting BEZ HRF convolution analysis - {timestamp}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load BEZ PSTH data
    print("\n=== Loading BEZ Data ===")
    bez_data, parameters = load_bez_psth_data()

    # Show canonical HRF if requested
    if args.show_hrf:
        print("\n=== Generating Canonical HRF ===")
        hrf, time_points = generate_canonical_hrf(tr=args.tr, time_length=args.hrf_length)
        plot_hrf(hrf, time_points, save_plots=args.save_plots,
                output_dir=args.output_dir, timestamp=timestamp)

    # Perform convolution for each run separately
    print(f"\n=== Convolving BEZ Data with HRF (each run separately) ===")
    convolved_responses, hrf, time_points = convolve_bez_runs_with_hrf(
        bez_data, parameters, fiber_types=args.fiber_types,
        tr=args.tr, hrf_length=args.hrf_length
    )

    # Save convolved data
    print(f"\n=== Saving Results ===")
    save_convolved_data(convolved_responses, args.output_dir, timestamp, model_name='BEZ')

    # Example plotting
    print(f"\n=== Example Plots ===")
    for fiber_type in args.fiber_types:
        if fiber_type in convolved_responses:
            # Plot individual runs for first CF and frequency
            plot_convolved_runs(
                convolved_responses, parameters,
                fiber_type=fiber_type,
                save_plots=args.save_plots,
                output_dir=args.output_dir,
                timestamp=timestamp
            )

            # Plot comparison across CFs
            plot_runs_comparison_across_cfs(
                convolved_responses, parameters,
                fiber_type=fiber_type,
                save_plots=args.save_plots,
                output_dir=args.output_dir,
                timestamp=timestamp
            )

    print(f"\n✓ Analysis complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
