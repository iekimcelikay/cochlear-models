# This is the first script that is written for this subfolder (for canonical HRF Convolution).
# This script performs HRF Convolution on the mean of the 10 runs of BEZ Model.

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
    convert_to_numeric,
    plot_hrf,
    save_convolved_data,
    generate_timestamp
)


def parse_arguments():
    """
    Parse command line arguments for subset testing.

    Returns:
    - args: Parsed arguments object
    """
    parser = argparse.ArgumentParser(description='Convolve PSTH data with canonical HRF')

    # Model selection arguments
    parser.add_argument('--model', choices=['cochlea', 'bez', 'both'], default='cochlea',
                        help='Which model to process (default: cochlea)')

    # Data subset arguments
    parser.add_argument('--max-cfs', type=int, default=None,
                        help='Maximum number of CFs to process (default: all)')
    parser.add_argument('--max-freqs', type=int, default=None,
                        help='Maximum number of frequencies to process (default: all)')
    parser.add_argument('--max-dbs', type=int, default=None,
                        help='Maximum number of dB levels to process (default: all)')
    parser.add_argument('--fiber-types', nargs='+', choices=['hsr', 'msr', 'lsr'],
                        default=['hsr', 'msr', 'lsr'],
                        help='Fiber types to process (default: all)')

    # Processing arguments
    parser.add_argument('--tr', type=float, default=0.1,
                        help='Temporal resolution in seconds (default: 0.1)')
    parser.add_argument('--hrf-length', type=float, default=32.0,
                        help='HRF duration in seconds (default: 32.0)')
    parser.add_argument('--db-level', type=float, default=60.0,
                        help='dB level for plotting (default: 60.0)')

    # Output arguments
    parser.add_argument('--output-dir', type=str,
                        default='/home/ekim/PycharmProjects/phd_firstyear/subcorticalSTRF/convolution_HRF/results',
                        help='Output directory for saved results')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting (useful for large datasets)')
    parser.add_argument('--no-save', action='store_true',
                        help='Skip saving results')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to files (in addition to displaying them)')
    parser.add_argument('--plot-format', type=str, choices=['png', 'pdf', 'svg'], default='png',
                        help='Format for saved plots (default: png)')

    # Test mode
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode (process only first 2 CFs, 3 frequencies, 2 dB levels)')

    return parser.parse_args()


def subset_parameters(parameters, args):
    """
    Create a subset of parameters based on command line arguments.

    Parameters:
    - parameters: Full parameter dictionary
    - args: Parsed command line arguments

    Returns:
    - subset_params: Subset of parameters
    - indices: Dictionary with indices for subsetting data arrays
    """
    subset_params = {}
    indices = {}

    # Handle test mode
    if args.test:
        print("Running in TEST MODE - processing subset of data...")
        max_cfs = 2
        max_freqs = 3
        max_dbs = 2
    else:
        max_cfs = args.max_cfs
        max_freqs = args.max_freqs
        max_dbs = args.max_dbs

    # Subset CFs
    if max_cfs is not None and max_cfs < len(parameters['cfs']):
        subset_params['cfs'] = parameters['cfs'][:max_cfs]
        indices['cf_indices'] = list(range(max_cfs))
        print(f"Using {max_cfs}/{len(parameters['cfs'])} CFs")
    else:
        subset_params['cfs'] = parameters['cfs']
        indices['cf_indices'] = list(range(len(parameters['cfs'])))
        print(f"Using all {len(parameters['cfs'])} CFs")

    # Subset frequencies
    if max_freqs is not None and max_freqs < len(parameters['frequencies']):
        subset_params['frequencies'] = parameters['frequencies'][:max_freqs]
        indices['freq_indices'] = list(range(max_freqs))
        print(f"Using {max_freqs}/{len(parameters['frequencies'])} frequencies")
    else:
        subset_params['frequencies'] = parameters['frequencies']
        indices['freq_indices'] = list(range(len(parameters['frequencies'])))
        print(f"Using all {len(parameters['frequencies'])} frequencies")

    # Subset dB levels
    if max_dbs is not None and max_dbs < len(parameters['dbs']):
        subset_params['dbs'] = parameters['dbs'][:max_dbs]
        indices['db_indices'] = list(range(max_dbs))
        print(f"Using {max_dbs}/{len(parameters['dbs'])} dB levels")
    else:
        subset_params['dbs'] = parameters['dbs']
        indices['db_indices'] = list(range(len(parameters['dbs'])))
        print(f"Using all {len(parameters['dbs'])} dB levels")

    return subset_params, indices


# FUNCTIONS
def load_cochlea_psth_data():
    """
    Load cochlea model PSTH data from MATLAB file.
    Uses the same loading approach as the regression analysis script.

    Returns:
    - cochlea_data: Dictionary containing PSTH data for all fiber types
    - parameters: Dictionary with CFs, frequencies, and dB levels
    """
    # Load cochlea model data - construct absolute path
    base_dir = Path("/home/ekim/PycharmProjects/phd_firstyear")
    cochlea_dir = (
        base_dir / "subcorticalSTRF" / "cochlea_meanrate" / 
        "out" / "condition_psths"
    )
    cochlea_file = cochlea_dir / 'cochlea_psths.mat'

    print(f"Loading cochlea PSTH data from: {cochlea_file}")
    cochlea_data = loadmat(
        str(cochlea_file), struct_as_record=False, squeeze_me=True
    )
    print("✓ Cochlea PSTH data loaded successfully")

    # Extract parameters
    parameters = {
        'cfs': np.sort(convert_to_numeric(cochlea_data['cfs'])),
        'frequencies': np.sort(
            convert_to_numeric(cochlea_data['frequencies'])
        ),
        'dbs': np.sort(convert_to_numeric(cochlea_data['dbs']))
    }

    print(f"Parameters extracted:")
    print(f"  CFs: {len(parameters['cfs'])} values")
    print(f"  Frequencies: {len(parameters['frequencies'])} values")
    print(f"  dB levels: {parameters['dbs']}")

    return cochlea_data, parameters


def load_bez_psth_data():
    """
    Load BEZ model PSTH data from MATLAB file.
    Uses the same loading approach as the regression analysis script.

    Returns:
    - bez_data: Dictionary containing PSTH data for all fiber types
    - parameters: Dictionary with CFs, frequencies, and dB levels
    """
    # Load BEZ model data - construct absolute path
    base_dir = Path("/home/ekim/PycharmProjects/phd_firstyear")
    bez_dir = (
        base_dir / "subcorticalSTRF" / "BEZ2018_meanrate" / 
        "results" / "processed_data"
    )
    bez_file = bez_dir / 'bez_acrossruns_psths.mat'

    print(f"Loading BEZ PSTH data from: {bez_file}")
    bez_data = loadmat(
        str(bez_file), struct_as_record=False, squeeze_me=True
    )
    print("✓ BEZ PSTH data loaded successfully")

    # Extract parameters
    parameters = {
        'cfs': np.sort(convert_to_numeric(bez_data['cfs'])),
        'frequencies': np.sort(
            convert_to_numeric(bez_data['frequencies'])
        ),
        'dbs': np.sort(convert_to_numeric(bez_data['dbs']))
    }

    print(f"Parameters extracted:")
    print(f"  CFs: {len(parameters['cfs'])} values")
    print(f"  Frequencies: {len(parameters['frequencies'])} values")
    print(f"  dB levels: {parameters['dbs']}")

    return bez_data, parameters


def load_both_models_data():
    """
    Load both BEZ and Cochlea model PSTH data.

    Returns:
    - combined_data: Dictionary containing both models' data
    - merged_parameters: Combined parameter dictionary
    """
    print("=== Loading Both Model Datasets ===\n")

    # Load BEZ data
    bez_data, bez_params = load_bez_psth_data()

    print()  # Add spacing

    # Load Cochlea data
    cochlea_data, cochlea_params = load_cochlea_psth_data()

    # Combine data into a single structure
    combined_data = {
        'bez': bez_data,
        'cochlea': cochlea_data
    }

    # Use cochlea parameters as base (they should be identical)
    # but verify they match
    merged_parameters = cochlea_params.copy()

    # Verify parameter consistency
    for key in ['cfs', 'frequencies', 'dbs']:
        if key in bez_params and key in cochlea_params:
            if not np.array_equal(bez_params[key], cochlea_params[key]):
                print(f"Warning: {key} parameters differ between models")
                print(f"  BEZ: {bez_params[key][:5]}... (length: {len(bez_params[key])})")
                print(f"  Cochlea: {cochlea_params[key][:5]}... (length: {len(cochlea_params[key])})")

    print("\n=== Both Datasets Loaded Successfully ===")

    return combined_data, merged_parameters









def convolve_cochlea_with_hrf(cochlea_data, parameters, indices, fiber_types, tr=0.1, hrf_length=32.0):
    """
    Convolve 200ms cochlea PSTH data with canonical HRF for each fiber type.

    Parameters:
    - cochlea_data: Loaded MATLAB data structure
    - parameters: Subset parameter dictionary
    - indices: Dictionary with indices for data subsetting
    - fiber_types: List of fiber types to process
    - tr: Temporal resolution in seconds
    - hrf_length: HRF duration in seconds
    """
    # Generate canonical HRF for full response duration
    hrf, time_points = generate_canonical_hrf(tr=tr, time_length=hrf_length)

    # Extract PSTH data for each fiber type
    cochlea_rates = {
        'lsr': cochlea_data['lsr_all'],
        'msr': cochlea_data['msr_all'],
        'hsr': cochlea_data['hsr_all']
    }

    convolved_responses = {}

    for fiber_type in fiber_types:
        print(f"\nProcessing {fiber_type.upper()} fibers...")
        convolved_responses[fiber_type] = {}

        psth_data = cochlea_rates[fiber_type]

        # Process each CF (subset)
        for cf_idx in indices['cf_indices']:
            cf_val = parameters['cfs'][cf_idx]
            convolved_responses[fiber_type][cf_val] = {}

            # Process each tone frequency (subset)
            for freq_idx in indices['freq_indices']:
                freq_val = parameters['frequencies'][freq_idx]
                convolved_responses[fiber_type][cf_val][freq_val] = {}

                # Process each dB level (subset)
                for db_idx in indices['db_indices']:
                    db_val = parameters['dbs'][db_idx]

                    try:
                        # Extract 200ms PSTH for this combination
                        psth = psth_data[cf_idx, freq_idx, db_idx]
                        psth = np.atleast_1d(psth).flatten()

                        # Remove non-finite values
                        if np.any(np.isfinite(psth)):
                            # Convolve brief PSTH with full HRF response
                            convolved, time_vec = convolve_brief_stimulus_with_hrf(
                                psth, stimulus_duration=0.2, hrf_duration=hrf_length, tr=tr
                            )
                            convolved_responses[fiber_type][cf_val][freq_val][db_val] = convolved
                        else:
                            print(f"Warning: No finite values for CF={cf_val}Hz, Freq={freq_val}Hz, dB={db_val}")

                    except (IndexError, ValueError) as e:
                        print(f"Warning: Skipping CF={cf_val}Hz, Freq={freq_val}Hz, dB={db_val} due to: {e}")
                        continue

        print(f"✓ {fiber_type.upper()} convolution complete")

    return convolved_responses, hrf, time_points


def convolve_bez_with_hrf(bez_data, parameters, indices, fiber_types, tr=0.1, hrf_length=32.0):
    """
    Convolve 200ms BEZ PSTH data with canonical HRF for each fiber type.

    Parameters:
    - bez_data: Loaded BEZ MATLAB data structure
    - parameters: Subset parameter dictionary
    - indices: Dictionary with indices for data subsetting
    - fiber_types: List of fiber types to process
    - tr: Temporal resolution in seconds
    - hrf_length: HRF duration in seconds

    Returns:
    - convolved_responses: Dictionary with HRF-convolved responses
    - hrf: The canonical HRF used
    - time_points: Time axis for HRF
    """
    # Generate canonical HRF for full response duration
    hrf, time_points = generate_canonical_hrf(tr=tr, time_length=hrf_length)

    # Extract PSTH data for each fiber type (BEZ naming convention)
    # Following the pattern from the regression analysis script
    bez_rates = {
        'lsr': bez_data['bez_rates'].lsr,
        'msr': bez_data['bez_rates'].msr,
        'hsr': bez_data['bez_rates'].hsr
    }

    convolved_responses = {}

    for fiber_type in fiber_types:
        print(f"\nProcessing BEZ {fiber_type.upper()} fibers...")
        convolved_responses[fiber_type] = {}

        psth_data = bez_rates[fiber_type]

        # Process each CF (subset)
        for cf_idx in indices['cf_indices']:
            cf_val = parameters['cfs'][cf_idx]
            convolved_responses[fiber_type][cf_val] = {}

            # Process each tone frequency (subset)
            for freq_idx in indices['freq_indices']:
                freq_val = parameters['frequencies'][freq_idx]
                convolved_responses[fiber_type][cf_val][freq_val] = {}

                # Process each dB level (subset)
                for db_idx in indices['db_indices']:
                    db_val = parameters['dbs'][db_idx]

                    try:
                        # Extract 200ms PSTH for this combination
                        psth = psth_data[cf_idx, freq_idx, db_idx]
                        psth = np.atleast_1d(psth).flatten()

                        # Remove non-finite values
                        if np.any(np.isfinite(psth)):
                            # Convolve brief PSTH with full HRF response
                            convolved, time_vec = convolve_brief_stimulus_with_hrf(
                                psth, stimulus_duration=0.2, hrf_duration=hrf_length, tr=tr
                            )
                            convolved_responses[fiber_type][cf_val][freq_val][db_val] = convolved
                        else:
                            print(f"Warning: No finite values for CF={cf_val}Hz, Freq={freq_val}Hz, dB={db_val}")

                    except (IndexError, ValueError) as e:
                        print(f"Warning: Skipping CF={cf_val}Hz, Freq={freq_val}Hz, dB={db_val} due to: {e}")
                        continue

        print(f"✓ BEZ {fiber_type.upper()} convolution complete")

    return convolved_responses, hrf, time_points


def convolve_both_models_with_hrf(combined_data, parameters, indices, fiber_types, tr=0.1, hrf_length=32.0):
    """
    Convolve both Cochlea and BEZ model PSTH data with canonical HRF.

    Parameters:
    - combined_data: Dictionary containing both models' data
    - parameters: Subset parameter dictionary
    - indices: Dictionary with indices for data subsetting
    - fiber_types: List of fiber types to process
    - tr: Temporal resolution in seconds
    - hrf_length: HRF duration in seconds

    Returns:
    - results: Dictionary with both model results
    - hrf: Canonical HRF array
    - time_points: Time axis for HRF
    """
    print("Processing both Cochlea and BEZ models...")

    # Generate canonical HRF
    hrf, time_points = generate_canonical_hrf(tr=tr, time_length=hrf_length)

    # Initialize results dictionary
    results = {}

    # Process Cochlea model
    print("  Processing Cochlea model...")
    cochlea_results, _, _ = convolve_cochlea_with_hrf(
        combined_data['cochlea'], parameters, indices, fiber_types, tr, hrf_length
    )
    results['cochlea'] = cochlea_results

    # Process BEZ model
    print("  Processing BEZ model...")
    bez_results, _, _ = convolve_bez_with_hrf(
        combined_data['bez'], parameters, indices, fiber_types, tr, hrf_length
    )
    results['bez'] = bez_results

    print("Both models processed successfully!")

    return results, hrf, time_points


def plot_convolved_responses(convolved_responses, parameters, fiber_type='hsr',
                             cf_val=None, db_val=60.0, figsize=(15, 10), max_subplots=None, model_name=None,
                             save_plots=False, output_dir=None, timestamp=None, plot_format='png'):
    """
    Plot HRF-convolved responses for visualization.

    Parameters:
    - convolved_responses: Dictionary with convolved data
    - parameters: Dictionary with stimulus parameters
    - fiber_type: Fiber type to plot ('hsr', 'msr', 'lsr')
    - cf_val: Specific CF to plot (if None, plots multiple CFs)
    - db_val: dB level to plot
    - figsize: Figure size
    - max_subplots: Maximum number of subplots (None for all)
    - model_name: Name of the model for plot titles (e.g., 'Cochlea', 'BEZ')
    - save_plots: whether to save the plot to file
    - output_dir: directory to save plots
    - timestamp: timestamp for filename
    - plot_format: format for saved plots
    """
    if fiber_type not in convolved_responses:
        print(f"No data for fiber type: {fiber_type}")
        return

    # Set model prefix once at the beginning
    model_prefix = f"{model_name} " if model_name else ""

    # Create consistent color mapping for all frequencies
    frequencies = sorted(parameters['frequencies'])
    n_freqs = len(frequencies)

    # Use a colormap that provides good distinction across many frequencies
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    # Use 'tab20' for up to 20 colors, or 'viridis' for more
    if n_freqs <= 20:
        colors = cm.tab20(np.linspace(0, 1, n_freqs))
    else:
        colors = cm.viridis(np.linspace(0, 1, n_freqs))

    # Create frequency to color mapping
    freq_colors = {freq: colors[i] for i, freq in enumerate(frequencies)}

    fiber_data = convolved_responses[fiber_type]

    if cf_val is None:
        # Plot all CFs (or limited by max_subplots)
        cfs_to_plot = list(fiber_data.keys())
        if max_subplots is not None:
            cfs_to_plot = cfs_to_plot[:max_subplots]

        # Adjust figure size based on number of subplots
        subplot_height = max(2, len(cfs_to_plot) * 2)
        fig = plt.figure(figsize=(figsize[0] + 3, subplot_height))

        # Store legend handles and labels from first subplot
        legend_handles = []
        legend_labels = []


        for i, cf in enumerate(cfs_to_plot):
            plt.subplot(len(cfs_to_plot), 1, i + 1)

            if cf in fiber_data:
                freq_data = fiber_data[cf]

                # Plot ALL frequencies with consistent colors
                for freq_val in frequencies:
                    if freq_val in freq_data and db_val in freq_data[freq_val]:
                        response = freq_data[freq_val][db_val]
                        time_vec = np.arange(len(response)) * 0.1  # 100ms resolution
                        line, =    plt.plot(time_vec, response, label=f'Tone: {freq_val:.0f}Hz',
                                alpha=0.8, color=freq_colors[freq_val], linewidth=1.5)

                        # Collect legen info only from first subplot
                        if i == 0:
                            legend_handles.append(line)
                            legend_labels.append(f'{freq_val:.0f}Hz')

                # Use the model prefix set at the beginning
                plt.title(f'{model_prefix}{fiber_type.upper()} - CF: {cf:.0f}Hz at {db_val}dB')
                plt.ylabel('Convolved Response')
                plt.grid(True, alpha=0.3)

        # Add single legend outside the plot area
        if legend_handles:
            # Determine legend layout based on number of frequencies
            if n_freqs <= 10:
                ncol = 1
                fontsize = 8
            elif n_freqs <= 20:
                ncol = 2
                fontsize = 7
            else:
                ncol = 3
                fontsize = 6

            fig.legend(legend_handles, legend_labels, loc='center right',
                       bbox_to_anchor=(0.98, 0.5), fontsize=fontsize, ncol=ncol,
                       title='Tone Frequency (Hz)', title_fontsize=fontsize+1)

        plt.xlabel('Time (seconds)')
        # Use the model prefix for the main title too
        plt.suptitle(f'{model_prefix}HRF-Convolved Responses: {fiber_type.upper()} Fibers (All CFs & Frequencies)', fontsize=14)

    else:
        # Plot single CF with ALL frequencies
        if cf_val in fiber_data:
            freq_data = fiber_data[cf_val]
            fig = plt.figure(figsize=(figsize[0] + 2, figsize[1]))

            legend_handles = []
            legend_labels = []

            # Plot ALL frequencies with consistent colors
            for freq_val in frequencies:
                if freq_val in freq_data and db_val in freq_data[freq_val]:
                    response = freq_data[freq_val][db_val]
                    time_vec = np.arange(len(response)) * 0.1
                    line, = plt.plot(time_vec, response, label=f'Tone: {freq_val:.0f}Hz',
                            alpha=0.8, color=freq_colors[freq_val], linewidth=1.5)
                    legend_handles.append(line)
                    legend_labels.append(f'{freq_val:.0f}Hz')

            # Use the model prefix for single CF plot title
            plt.title(f'{model_prefix}{fiber_type.upper()} - CF: {cf_val:.0f}Hz at {db_val}dB (All Frequencies)')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Convolved Response')
            plt.grid(True, alpha=0.3)

            # Add single legend outside the plot area
            if legend_handles:
                # Determine legend layout based on number of frequencies
                if n_freqs <= 15:
                    ncol = 1
                    fontsize = 8
                elif n_freqs <= 30:
                    ncol = 2
                    fontsize = 7
                else:
                    ncol = 3
                    fontsize = 6

                plt.legend(handles=legend_handles, labels=legend_labels,
                           bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=fontsize, ncol=ncol,
                           title='Tone Frequency (Hz)', title_fontsize=fontsize + 1)



    plt.tight_layout()

    # Save plot if requested
    if save_plots and output_dir and timestamp:
        save_plot('hrf_responses', output_dir, timestamp, model_name=model_name,
                 fiber_type=fiber_type, plot_format=plot_format)

    plt.show()


def save_convolved_data(convolved_responses, output_dir, timestamp=None, 
                       model_name=None):
    """
    Save convolved responses to files with timestamps.

    Parameters:
    - convolved_responses: Dictionary with convolved data
    - output_dir: Directory to save files
    - timestamp: Optional timestamp string (if None, generates current timestamp)
    - model_name: Optional model name to include in filename
    """
    import datetime

    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for fiber_type in convolved_responses:
        # Build filename with model name if provided
        if model_name:
            filename = (
                f'{model_name.upper()}_convolved_responses_'
                f'{fiber_type}_{timestamp}.pkl'
            )
        else:
            filename = f'convolved_responses_{fiber_type}_{timestamp}.pkl'
        
        output_file = os.path.join(output_dir, filename)

        with open(output_file, 'wb') as f:
            pickle.dump(convolved_responses[fiber_type], f)

        print(f"Saved {fiber_type.upper()} convolved responses to: "
              f"{output_file}")

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
    import datetime

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

# Main execution
def main(args=None):
    """Main pipeline to load PSTH data and convolve with HRF."""

    # Parse arguments if not provided
    if args is None:
        args = parse_arguments()

    print(f"=== PSTH-HRF Convolution Analysis ({args.model.upper()}) ===\n")
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  TR: {args.tr}s")
    print(f"  HRF length: {args.hrf_length}s")
    print(f"  Fiber types: {args.fiber_types}")
    print(f"  Test mode: {args.test}")
    print(f"  Output directory: {args.output_dir}")
    print()

    # Load data based on model selection
    if args.model == 'cochlea':
        print("Loading Cochlea model data...")
        model_data, full_parameters = load_cochlea_psth_data()
        parameters, indices = subset_parameters(full_parameters, args)

        print("Convolving Cochlea PSTH data with canonical HRF...")
        convolved_responses, hrf, time_points = convolve_cochlea_with_hrf(
            model_data, parameters, indices, args.fiber_types,
            tr=args.tr, hrf_length=args.hrf_length
        )

    elif args.model == 'bez':
        print("Loading BEZ model data...")
        model_data, full_parameters = load_bez_psth_data()
        parameters, indices = subset_parameters(full_parameters, args)

        print("Convolving BEZ PSTH data with canonical HRF...")
        convolved_responses, hrf, time_points = convolve_bez_with_hrf(
            model_data, parameters, indices, args.fiber_types,
            tr=args.tr, hrf_length=args.hrf_length
        )

    elif args.model == 'both':
        print("Loading both Cochlea and BEZ model data...")
        combined_data, full_parameters = load_both_models_data()
        parameters, indices = subset_parameters(full_parameters, args)

        print("Convolving both models' PSTH data with canonical HRF...")
        convolved_responses, hrf, time_points = convolve_both_models_with_hrf(
            combined_data, parameters, indices, args.fiber_types,
            tr=args.tr, hrf_length=args.hrf_length
        )

    # Plot the HRF first (unless skipped)
    if not args.no_plot:
        timestamp = generate_timestamp() if args.save_plots else None

        print("\nPlotting canonical HRF...")
        plot_hrf(hrf, time_points, save_plots=args.save_plots, 
                output_dir=args.output_dir,
                timestamp=timestamp, plot_format=args.plot_format)

        # Plot convolved responses for each fiber type
        print("\nPlotting convolved responses...")
        if args.model == 'both':
            # Plot both models for comparison
            for model_name in ['cochlea', 'bez']:
                print(f"\nPlotting {model_name.upper()} model responses...")
                for fiber_type in args.fiber_types:
                    if fiber_type in convolved_responses[model_name]:
                        print(f"  Plotting {fiber_type.upper()} responses...")
                        plot_convolved_responses(
                            convolved_responses[model_name], parameters,
                            fiber_type=fiber_type, db_val=args.db_level,
                            model_name=model_name.upper(), 
                            save_plots=args.save_plots,
                            output_dir=args.output_dir, timestamp=timestamp,
                            plot_format=args.plot_format
                        )
        else:
            # Single model plotting
            for fiber_type in args.fiber_types:
                if fiber_type in convolved_responses:
                    print(f"Plotting {fiber_type.upper()} responses...")
                    plot_convolved_responses(
                        convolved_responses, parameters,
                        fiber_type=fiber_type, db_val=args.db_level,
                        model_name=args.model.upper(), 
                        save_plots=args.save_plots,
                        output_dir=args.output_dir, timestamp=timestamp,
                        plot_format=args.plot_format
                    )
    else:
        print("Skipping plots as requested...")

    # Save results (unless skipped)
    if not args.no_save:
        timestamp = generate_timestamp()

        print(f"\nSaving convolved responses to: {args.output_dir}")
        if args.model == 'both':
            # Save both models' results separately
            for model_name in ['cochlea', 'bez']:
                model_output_dir = Path(args.output_dir) / model_name
                print(
                    f"  Saving {model_name.upper()} results to: "
                    f"{model_output_dir}"
                )
                save_convolved_data(
                    convolved_responses[model_name], 
                    str(model_output_dir), 
                    timestamp,
                    model_name=model_name
                )

            # Also save combined results with timestamp
            combined_output_file = (
                Path(args.output_dir) / 
                f"combined_models_results_{timestamp}.pkl"
            )
            print(f"  Saving combined results to: {combined_output_file}")
            with open(combined_output_file, 'wb') as f:
                pickle.dump(convolved_responses, f)
        else:
            # Save single model results with model name in filename
            save_convolved_data(
                convolved_responses, 
                args.output_dir, 
                timestamp,
                model_name=args.model
            )
    else:
        print("Skipping save as requested...")

    print(f"\n=== {args.model.upper()} Convolution Analysis Complete! ===")

    return convolved_responses, hrf, time_points, parameters

# Execute the pipeline
if __name__ == "__main__":
    convolved_data, canonical_hrf, time_axis, params = main()
