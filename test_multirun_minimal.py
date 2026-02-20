"""
Multi-run PSTH test with moderate parameters.

Tests with:
- 10 CFs (characteristic frequencies from 125-2500 Hz)
- 30 fibers total (10 HSR, 10 MSR, 10 LSR)
- 100 runs (configurable via NUM_RUNS)
- 1 short WAV file
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from peripheral_models.cochlea_config import CochleaConfig
from cochlea_multiple_runs_pipeline_110226 import MultiRunPSTH
from utils.psth_aggregator import PSTHAggregator

# Test parameters
NUM_RUNS = 50
NUM_WORKERS = 3  # Number of parallel workers (set to 1 for sequential, or None for auto)
MAX_MEMORY_GB = 4.0  # Maximum memory per worker in GB (or None for 80% of available)


def custom_parser(filename: str) -> dict:
    """Parse custom filename format."""
    parts = filename.replace('.wav', '').split('_')
    return {
        'sequence': parts[0],
        'center_freq': parts[1],
        'tone_duration': parts[2],
        'isi': parts[3],
        'total_duration': parts[4],
        'num_tones': parts[5],
    }


def main():
    """Run minimal test."""

    print("\n" + "="*60)
    print("MULTI-RUN PSTH TEST (50 runs × 30 CFs)")
    print("="*60)
    print("Parameters:")
    print("  - CFs: 30")
    print("  - Fibers: 192 (64 HSR, 64 MSR, 64 LSR)")
    print(f"  - Runs: {NUM_RUNS}")
    print("  - Files: 1 (shortest)")
    print(f"  - Workers: {NUM_WORKERS or 'auto'}")
    print(f"  - Memory limit: {MAX_MEMORY_GB or 'auto'} GB per worker")
    print("="*60 + "\n")

    # Configuration with 10 CFs
    config = CochleaConfig(
        # Model parameters
        peripheral_fs=100000,
        min_cf=125,
        max_cf=2500,
        num_cf=30,  # 10 CFs
        num_ANF=(64, 64, 64),  # 30 fibers total (10 per type)
        powerlaw='approximate',
        seed=42,  # Base seed: each run gets unique seed (42, 12387, 24732, ...)
        fs_target=1000.0,

        # Output settings
        output_dir="./models_output/test_multirun_minimal",
        experiment_name="minimal_test",
        save_formats=['npz'],
        save_mean_rates=False,
        save_psth=True,

        # Logging
        log_console_level='INFO',
        log_file_level='DEBUG'
    )

    # Get only the first (shortest) WAV file
    wav_dir = Path("./stimuli/produced")
    all_wav_files = sorted(wav_dir.glob("*.wav"))

    if not all_wav_files:
        print(f"ERROR: No WAV files found in {wav_dir}")
        return

    # Use only the first file for quick testing
    wav_files = [all_wav_files[0]]
    print(f"Processing file: {wav_files[0].name}\n")

    # Step 1: Multi-run execution
    print("\nSTEP 1: Running multiple simulations...")
    multi_run = MultiRunPSTH(
        base_config=config,
        wav_files=wav_files,
        num_runs=NUM_RUNS,
        custom_parser=custom_parser,
        max_memory_gb=MAX_MEMORY_GB
    )
    multi_run.run_all_parallel(num_workers=NUM_WORKERS)

    # Step 2: View run summary
    print("\nSTEP 2: Checking run summary...")
    summary_json = Path(multi_run.base_output_dir) / "run_summary.json"
    summary_txt = Path(multi_run.base_output_dir) / "run_summary.txt"

    if summary_json.exists():
        print(f"  ✓ Detailed summary saved: {summary_json}")
        print(f"  ✓ Readable summary saved: {summary_txt}")

        # Print key statistics
        import json
        with open(summary_json, 'r') as f:
            summary = json.load(f)

        print(f"\n  Key Statistics:")
        print(f"    Wall time: {summary['timing']['wall_time_minutes']:.2f} minutes")
        print(f"    Avg runtime/run: {summary['timing']['average_runtime_per_run']:.2f} seconds")
        print(f"    Peak memory: {summary['memory']['peak_memory_gb']:.3f} GB")
        print(f"    Avg peak memory: {summary['memory']['average_peak_memory_gb']:.3f} GB")
    else:
        print(f"  ✗ Summary files not found")

    # Step 3: Aggregate results
    print("\nSTEP 3: Aggregating results...")
    aggregator = PSTHAggregator(
        base_output_dir=multi_run.base_output_dir,  # Use the auto-generated directory
        num_runs=NUM_RUNS
    )
    aggregated_files = aggregator.aggregate_all_files(wav_files)

    # Step 4: Verify outputs
    print("\nSTEP 4: Verifying outputs...")

    # Check individual run directories
    output_dir = Path(multi_run.base_output_dir)  # Use the auto-generated directory
    for i in range(NUM_RUNS):
        run_dir = output_dir / f"run_{i:03d}"
        if run_dir.exists():
            # Search recursively for npz files
            npz_files = list(run_dir.rglob("*.npz"))
            print(f"  ✓ run_{i:03d}: {len(npz_files)} file(s)")
        else:
            print(f"  ✗ run_{i:03d}: NOT FOUND")

    # Check aggregated directory
    agg_dir = output_dir / "aggregated"
    if agg_dir.exists():
        npz_files = list(agg_dir.glob("*.npz"))
        print(f"  ✓ aggregated: {len(npz_files)} file(s)")

        # Load and check aggregated data
        if aggregated_files:
            import numpy as np
            main_file = aggregated_files[0]
            stats_file = main_file.parent / f"{main_file.stem}_stats.npz"

            print(f"\n  Loading: {main_file.name}")
            data = np.load(main_file, allow_pickle=True)
            print(f"    Keys: {list(data.keys())}")

            if 'population_rate_psth' in data:
                psth_shape = data['population_rate_psth'].shape
                print(f"    population_rate_psth shape: {psth_shape}")
                print(f"    Expected: (2, n_bins) where n_bins depends on duration")

            if stats_file.exists():
                print(f"\n  Loading: {stats_file.name}")
                stats_data = np.load(stats_file, allow_pickle=True)
                print(f"    Keys: {list(stats_data.keys())}")
                if 'population_rate_psth_std' in stats_data:
                    print(f"    population_rate_psth_std shape: {stats_data['population_rate_psth_std'].shape}")
    else:
        print(f"  ✗ aggregated: NOT FOUND")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
