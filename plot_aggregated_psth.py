"""
Plot aggregated PSTH timecourses with error bars.

Shows mean ± SEM for population rates and individual fiber types.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_aggregated_psth(aggregated_dir, filename_stem):
    """
    Plot aggregated PSTH results with error bars.

    Parameters
    ----------
    aggregated_dir : str or Path
        Directory containing aggregated results
    filename_stem : str
        Stem of the filename (without .npz extension)
    """
    aggregated_dir = Path(aggregated_dir)

    # Load main file (mean values)
    main_file = aggregated_dir / f"{filename_stem}.npz"
    data = np.load(main_file, allow_pickle=True)

    # Load stats file (std/sem)
    stats_file = aggregated_dir / f"{filename_stem}_stats.npz"
    stats = np.load(stats_file, allow_pickle=True)

    # Extract data
    population_rate_mean = data['population_rate_psth']
    population_rate_sem = stats['population_rate_psth_sem']
    time_axis = data['time_axis']
    cf_list = data['cf_list']

    # Extract PSTH for fiber types
    psth_mean = data['psth'].item()
    psth_std = stats['psth_std'].item()
    psth_sem = stats['psth_sem'].item()

    # Get metadata
    metadata = data['aggregation_metadata'].item()
    num_runs = metadata['num_runs']

    num_cf = len(cf_list)

    # Create figure
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Population rate for all CFs
    ax1 = plt.subplot(2, 2, 1)
    for i, cf in enumerate(cf_list):
        mean = population_rate_mean[i, :]
        sem = population_rate_sem[i, :]

        # Plot mean line
        line = ax1.plot(time_axis, mean, label=f'CF={cf:.0f} Hz', linewidth=2, alpha=0.8)[0]

        # Plot SEM as shaded region
        ax1.fill_between(time_axis, mean - sem, mean + sem,
                         alpha=0.2, color=line.get_color())

    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Population Rate (spikes/s)', fontsize=12)
    ax1.set_title(f'Population Rate PSTH (mean ± SEM, n={num_runs} runs)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: First CF - all fiber types
    cf_idx = 0
    ax2 = plt.subplot(2, 2, 2)

    for fiber_type in ['hsr', 'msr', 'lsr']:
        mean = psth_mean[fiber_type][cf_idx, :]
        sem = psth_sem[fiber_type][cf_idx, :]

        line = ax2.plot(time_axis, mean, label=fiber_type.upper(), linewidth=2, alpha=0.8)[0]
        ax2.fill_between(time_axis, mean - sem, mean + sem,
                         alpha=0.2, color=line.get_color())

    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Firing Rate (spikes/s)', fontsize=12)
    ax2.set_title(f'Fiber Types at CF={cf_list[cf_idx]:.0f} Hz (mean ± SEM)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: HSR responses for all CFs
    ax3 = plt.subplot(2, 2, 3)

    for i, cf in enumerate(cf_list):
        mean = psth_mean['hsr'][i, :]
        sem = psth_sem['hsr'][i, :]

        line = ax3.plot(time_axis, mean, label=f'CF={cf:.0f} Hz', linewidth=2, alpha=0.8)[0]
        ax3.fill_between(time_axis, mean - sem, mean + sem,
                         alpha=0.2, color=line.get_color())

    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Firing Rate (spikes/s)', fontsize=12)
    ax3.set_title(f'HSR Fibers (mean ± SEM, n={num_runs} runs)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Comparison of mean vs std across time (for first CF)
    ax4 = plt.subplot(2, 2, 4)

    pop_mean = population_rate_mean[0, :]
    pop_std = stats['population_rate_psth_std'][0, :]

    ax4_twin = ax4.twinx()

    line1 = ax4.plot(time_axis, pop_mean, 'b-', linewidth=2, label='Mean', alpha=0.8)[0]
    line2 = ax4_twin.plot(time_axis, pop_std, 'r-', linewidth=2, label='Std Dev', alpha=0.8)[0]

    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Mean Rate (spikes/s)', fontsize=12, color='b')
    ax4_twin.set_ylabel('Std Dev (spikes/s)', fontsize=12, color='r')
    ax4.set_title(f'Variability Across Runs (CF={cf_list[0]:.0f} Hz)', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='b')
    ax4_twin.tick_params(axis='y', labelcolor='r')
    ax4.grid(True, alpha=0.3)

    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right', fontsize=10)

    plt.tight_layout()

    # Save figure
    output_file = aggregated_dir / f"{filename_stem}_timecourse.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure: {output_file}")

    plt.show()

    return fig


def main():
    """Main execution."""

    # Load from minimal test results
    aggregated_dir = Path("./models_output/test_multirun_minimal/aggregated")
    filename_stem = "sequence01_fc440hz_dur200ms_isi100ms_total0.3sec_numtones1"

    if not aggregated_dir.exists():
        print(f"ERROR: Aggregated directory not found: {aggregated_dir}")
        print("Run test_multirun_minimal.py first to generate results.")
        return

    main_file = aggregated_dir / f"{filename_stem}.npz"
    if not main_file.exists():
        print(f"ERROR: Aggregated file not found: {main_file}")
        return

    print("="*60)
    print("PLOTTING AGGREGATED PSTH TIMECOURSES")
    print("="*60)
    print(f"Directory: {aggregated_dir}")
    print(f"File: {filename_stem}")
    print("="*60)

    fig = plot_aggregated_psth(aggregated_dir, filename_stem)

    print("\n" + "="*60)
    print("PLOTTING COMPLETE")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
