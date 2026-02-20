import sys
from pathlib import Path

# Add parent directory to path to import peripheral_models
sys.path.insert(0, str(Path(__file__).parent.parent))

from peripheral_models.cochlea_config import CochleaConfig
from peripheral_models.cochlea_simulation import CochleaWavSimulation
from cochlea_multiple_runs_pipeline_110226 import MultiRunPSTH
from utils.psth_aggregator import PSTHAggregator

def custom_parser(filename:str) -> dict:
    """Parse custon filename format: sequence01_fc440hz_dur200ms_isi100ms_total5sec_numtones1.wav """
    parts = filename.replace('.wav', '').split('_')
    return{
        'sequence': parts[0],
        'center_freq': parts[1],
        'tone_duration': parts[2],
        'isi': parts[3],
        'total_duration': parts[4],
        'num_tones': parts[5],

        }



def main(use_multi_run=True, num_runs=100):
    """
    Process WAV files through cochlea rate model.

    Parameters
    ----------
    use_multi_run : bool
        If True, run multiple times with different seeds and aggregate.
        If False, run single simulation (original behavior).
    num_runs : int
        Number of runs for multi-run mode (default: 100)
    """

    # Configure the model
    config = CochleaConfig(
        # Model parameters
        peripheral_fs=100000,
        min_cf=125,
        max_cf=2500,
        num_cf=20,
        num_ANF=(128, 128, 128),
        powerlaw = 'approximate',
        seed = 0,
        fs_target = 1000.0,

        # Output settings
        output_dir = "./models_output/cochlea_wavfiles_120226_05",
        experiment_name="customsequences_test05",
        save_formats=['npz'],
        save_mean_rates=False,
        save_psth=True,

        # Logging
        log_console_level = 'INFO',
        log_file_level = 'DEBUG'
        )

    # Get WAV files
    wav_dir = Path("./stimuli/produced")
    wav_files = sorted(wav_dir.glob("*.wav"))

    if not wav_files:
        print(f"No WAV files found in {wav_dir}")
        return

    if use_multi_run:
        # Multi-run workflow with aggregation
        print(f"\n{'='*60}")
        print(f"Running Multi-Run PSTH Pipeline")
        print(f"{'='*60}\n")

        # Step 1: Run all simulations in parallel
        multi_run = MultiRunPSTH(config, wav_files, num_runs=num_runs, custom_parser=custom_parser)
        multi_run.run_all_parallel()

        # Step 2: Aggregate results
        aggregator = PSTHAggregator(config.output_dir, num_runs=num_runs)
        aggregator.aggregate_all_files(wav_files)

        print(f"\nMulti-run pipeline completed!")
        print(f"Individual runs saved in: {config.output_dir}/run_XXX/")
        print(f"Aggregated results saved in: {config.output_dir}/aggregated/")

    else:
        # Original single-run simulation
        print("\nRunning single simulation...")
        simulation = CochleaWavSimulation(config, wav_files, auto_parse=True, parser_func=custom_parser)
        simulation.run()
        print(f"Results saved in: {config.output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Cochlea WAV simulation with optional multi-run")
    parser.add_argument('--single-run', action='store_true', help='Run single simulation instead of multi-run')
    parser.add_argument('--num-runs', type=int, default=100, help='Number of runs for multi-run mode (default: 100)')
    args = parser.parse_args()

    main(use_multi_run=not args.single_run, num_runs=args.num_runs)
