"""
Sequential CF batch processing for WAV stimuli with pause-and-continue.

Processes CFs in batches to manage memory usage for large-scale simulations.
After each batch completes, shows summary and waits for user confirmation.

Usage:
    python run_wav_cf_batched.py --num-cf 100 --batch-size 10 --num-batches 3
    python run_wav_cf_batched.py --num-cf 100 --batch-size 10 --start-from 3 --num-batches 2

Example:
    # Process first 3 batches (CFs 0-30)
    python run_wav_cf_batched.py --num-cf 100 --batch-size 10 --num-batches 3

    # Continue from batch 3 (CFs 30-50)
    python run_wav_cf_batched.py --num-cf 100 --batch-size 10 --start-from 3 --num-batches 2
"""
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from peripheral_models.cochlea_config import CochleaConfig
from peripheral_models.cochlea_simulation import CochleaWavSimulation
import psutil


def custom_parser(filename: str) -> dict:
    """Parse WAV filename format."""
    parts = filename.replace('.wav', '').split('_')
    return {
        'sequence': parts[0],
        'center_freq': parts[1] if len(parts) > 1 else '',
        'tone_duration': parts[2] if len(parts) > 2 else '',
        'isi': parts[3] if len(parts) > 3 else '',
        'total_duration': parts[4] if len(parts) > 4 else '',
        'num_tones': parts[5] if len(parts) > 5 else '',
    }


def print_batch_summary(batch_idx, config, elapsed_time, peak_memory_gb):
    """Print summary after batch completion."""
    start_idx, end_idx = config.get_batch_cf_range(batch_idx)
    cf_array = config.get_batch_cf_array(batch_idx)

    print("\n" + "="*70)
    print(f"BATCH {batch_idx} COMPLETED")
    print("="*70)
    print(f"CF Range: indices {start_idx}-{end_idx} ({len(cf_array)} CFs)")
    print(f"CF Values: {cf_array[0]:.1f} - {cf_array[-1]:.1f} Hz")
    print(f"Runtime: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Peak Memory: {peak_memory_gb:.2f} GB")
    print("="*70)


def wait_for_continue():
    """Wait for user input to continue or quit."""
    print("\nPress Enter to continue to next batch, or 'q' to quit: ", end='', flush=True)
    user_input = input().strip().lower()
    if user_input == 'q':
        print("Stopping batch processing.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sequential CF batch processing for WAV stimuli"
    )

    # CF batch parameters
    parser.add_argument('--num-cf', type=int, required=True,
                       help='Total number of CFs (e.g., 100)')
    parser.add_argument('--batch-size', type=int, required=True,
                       help='Number of CFs per batch (e.g., 10)')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Starting batch index (0-based, for resuming)')
    parser.add_argument('--num-batches', type=int, default=None,
                       help='Number of batches to process (default: all remaining)')

    # Model parameters
    parser.add_argument('--min-cf', type=float, default=125,
                       help='Minimum CF (Hz)')
    parser.add_argument('--max-cf', type=float, default=2500,
                       help='Maximum CF (Hz)')
    parser.add_argument('--num-anf', type=int, nargs=3, default=[128, 128, 128],
                       help='Number of ANFs per type: LSR MSR HSR')
    parser.add_argument('--peripheral-fs', type=float, default=100000,
                       help='Peripheral sampling rate (Hz)')
    parser.add_argument('--fs-target', type=float, default=1000.0,
                       help='Target PSTH sampling rate (Hz)')

    # I/O parameters
    parser.add_argument('--wav-files', type=str, nargs='+', default=None,
                       help='WAV file paths (default: first file in stimuli/produced/)')
    parser.add_argument('--output-dir', type=str, default='./models_output/wav_cf_batched',
                       help='Output directory')
    parser.add_argument('--experiment-name', type=str, default='wav_cf_batch',
                       help='Experiment name prefix')

    # Batch control
    parser.add_argument('--auto-continue', action='store_true',
                        help='Process all batches without pausing for confirmation')

    # Memory settings
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Calculate total batches
    import numpy as np
    total_batches = int(np.ceil(args.num_cf / args.batch_size))

    # Determine which batches to process
    start_batch = args.start_from
    if args.num_batches is None:
        end_batch = total_batches
    else:
        end_batch = min(start_batch + args.num_batches, total_batches)

    # Validate
    if start_batch >= total_batches:
        print(f"Error: start-from ({start_batch}) >= total batches ({total_batches})")
        return

    # Get WAV files
    if args.wav_files is None:
        wav_dir = Path("./stimuli/produced")
        all_wav_files = sorted(wav_dir.glob("*.wav"))
        if not all_wav_files:
            print(f"Error: No WAV files found in {wav_dir}")
            return
        wav_files = [all_wav_files[0]]  # Use first file
        print(f"Using default WAV file: {wav_files[0].name}")
    else:
        wav_files = [Path(f) for f in args.wav_files]

    # Print overall plan
    print("\n" + "="*70)
    print("CF BATCH PROCESSING PLAN")
    print("="*70)
    print(f"Total CFs: {args.num_cf} ({args.min_cf:.0f} - {args.max_cf:.0f} Hz)")
    print(f"Batch size: {args.batch_size} CFs per batch")
    print(f"Total batches: {total_batches}")
    print(f"Processing: Batches {start_batch} to {end_batch-1} ({end_batch - start_batch} batches)")
    print(f"ANF configuration: {args.num_anf[0]} LSR, {args.num_anf[1]} MSR, {args.num_anf[2]} HSR")
    print(f"WAV files: {len(wav_files)}")
    print(f"Output: {args.output_dir}")
    print("="*70 + "\n")

    # Process each batch sequentially
    for batch_idx in range(start_batch, end_batch):
        print(f"\n{'='*70}")
        print(f"STARTING BATCH {batch_idx} of {total_batches-1}")
        print(f"{'='*70}\n")

        batch_start_time = time.time()
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024**3)

        # Create config for this batch
        config = CochleaConfig(
            # Model parameters
            peripheral_fs=args.peripheral_fs,
            min_cf=args.min_cf,
            max_cf=args.max_cf,
            num_cf=args.num_cf,
            num_ANF=tuple(args.num_anf),
            powerlaw='approximate',
            seed=args.seed,
            fs_target=args.fs_target,

            # CF Batch settings
            cf_batch_enabled=True,
            cf_batch_size=args.batch_size,
            cf_batch_current=batch_idx,

            # Output settings
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            save_formats=['npz'],
            save_mean_rates=True,
            save_psth=True,

            # Logging
            log_console_level='INFO',
            log_file_level='DEBUG'
        )

        # Run simulation for this batch
        simulation = CochleaWavSimulation(
            config,
            wav_files,
            auto_parse=True,
            parser_func=custom_parser
        )
        simulation.run()

        batch_elapsed = time.time() - batch_start_time
        mem_after = process.memory_info().rss / (1024**3)
        peak_memory = max(mem_before, mem_after)

        # Clean up
        del simulation, config
        import gc
        gc.collect()

        # Show summary
        print_batch_summary(batch_idx, config, batch_elapsed, peak_memory)

        # Wait for user confirmation (except for last batch)
        if batch_idx < end_batch - 1:
            if not args.auto_continue:
                if not wait_for_continue():
                    break
            else:
                print("Auto-continuing to next batch...\n")

    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"Processed batches {start_batch} to {batch_idx}")
    print(f"Output directory: {args.output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

# USAGE EXAMPLES:
# Process first 3 batches (30 CFs: 0-29)
#python run_wav_cf_batched.py --num-cf 100 --batch-size 10 --num-batches 3

# Review results, then continue from batch 3
#python run_wav_cf_batched.py --num-cf 100 --batch-size 10 --start-from 3 --num-batches 2

# Process all remaining batches from batch 5 onward
#python run_wav_cf_batched.py --num-cf 100 --batch-size 10 --start-from 5

# Custom configuration
#python run_wav_cf_batched.py --num-cf 80 --batch-size 8 --num-anf 256 256 256 --fs-target 500

# High-resolution simulation: 80 CFs, 512 fibers per type, 2 kHz PSTH sampling
# Process all 10 batches automatically without pausing
#python run_wav_cf_batched.py \
#    --num-cf 80 \
#    --batch-size 8 \
#    --num-anf 512 512 512 \
#    --fs-target 2000 \
#    --peripheral-fs 100000 \
#    --output-dir ./models_output/high_res_sim \
#    --experiment-name high_res_80cf \
#    --auto-continue


# Same config but only process first 2 batches (16 CFs) to test
#python run_wav_cf_batched.py \
#    --num-cf 80 \
#    --batch-size 8 \
#    --num-batches 2 \
#   --num-anf 512 512 512 \
#   --fs-target 2000 \
#   --output-dir ./models_output/high_res_sim \
#   --experiment-name high_res_80cf \
#   --auto-continue