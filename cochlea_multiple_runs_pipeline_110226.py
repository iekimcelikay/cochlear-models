# Created on 11.02.2026
# Multi-run PSTH pipeline for cochlea simulations with parallel processing
import sys
import os
import copy
import time
import gc
import argparse
import multiprocessing
import psutil
import signal
import threading
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from peripheral_models.cochlea_config import CochleaConfig
from peripheral_models.cochlea_simulation import CochleaWavSimulation
from utils.timestamp_utils import generate_timestamp


class MemoryMonitor:
    """
    Monitor memory usage and terminate process if limit exceeded.

    Runs in a separate thread to continuously monitor memory consumption
    and prevent system freeze due to excessive memory usage.
    """

    def __init__(self, max_memory_gb=None, check_interval=1.0):
        """
        Initialize memory monitor.

        Parameters
        ----------
        max_memory_gb : float, optional
            Maximum memory in GB. If None, uses 80% of available memory.
        check_interval : float
            How often to check memory in seconds (default: 1.0)
        """
        self.process = psutil.Process()
        self.max_memory_bytes = self._calculate_memory_limit(max_memory_gb)
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None

        # Track memory statistics
        self.peak_memory_bytes = 0
        self.memory_samples = []

        max_gb = self.max_memory_bytes / (1024**3)
        print(f"[MemoryMonitor] Initialized with {max_gb:.2f} GB limit")

    def _calculate_memory_limit(self, max_memory_gb):
        """Calculate memory limit in bytes."""
        if max_memory_gb is not None:
            return int(max_memory_gb * 1024**3)
        else:
            # Use 80% of available system memory
            available = psutil.virtual_memory().available
            return int(available * 0.8)

    def _monitor_loop(self):
        """Continuously monitor memory usage."""
        while self.monitoring:
            try:
                # Get current memory usage
                mem_info = self.process.memory_info()
                current_memory = mem_info.rss  # Resident Set Size

                # Track peak and collect samples
                self.peak_memory_bytes = max(self.peak_memory_bytes, current_memory)
                self.memory_samples.append(current_memory)

                if current_memory > self.max_memory_bytes:
                    current_gb = current_memory / (1024**3)
                    max_gb = self.max_memory_bytes / (1024**3)
                    print(f"\n[MemoryMonitor] MEMORY LIMIT EXCEEDED!")
                    print(f"[MemoryMonitor] Current: {current_gb:.2f} GB")
                    print(f"[MemoryMonitor] Limit: {max_gb:.2f} GB")
                    print(f"[MemoryMonitor] Terminating process...")

                    # Force terminate the process
                    os.kill(os.getpid(), signal.SIGTERM)
                    time.sleep(1)  # Give SIGTERM a chance
                    os.kill(os.getpid(), signal.SIGKILL)

                time.sleep(self.check_interval)
            except Exception as e:
                print(f"[MemoryMonitor] Error in monitoring: {e}")
                break

    def get_stats(self):
        """Get memory statistics in GB."""
        if not self.memory_samples:
            return {'peak_gb': 0, 'mean_gb': 0, 'final_gb': 0}

        peak_gb = self.peak_memory_bytes / (1024**3)
        mean_gb = sum(self.memory_samples) / len(self.memory_samples) / (1024**3)
        final_gb = self.memory_samples[-1] / (1024**3)

        return {
            'peak_gb': round(peak_gb, 3),
            'mean_gb': round(mean_gb, 3),
            'final_gb': round(final_gb, 3)
        }

    def start(self):
        """Start monitoring in background thread."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print(f"[MemoryMonitor] Started monitoring")

    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            print(f"[MemoryMonitor] Stopped monitoring")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


class MultiRunPSTH:
    """
    Orchestrates multiple PSTH simulation runs with unique seeds.

    Each run generates independent spike trains using a deterministic seed,
    allowing for aggregation of stochastic neural responses.
    """

    def __init__(self, base_config, wav_files, num_runs=100, custom_parser=None,
                 max_memory_gb=None):
        """
        Initialize multi-run PSTH orchestrator.

        Parameters
        ----------
        base_config : CochleaConfig
            Base configuration (will be copied for each run)
        wav_files : list of Path
            WAV files to process
        num_runs : int
            Number of independent runs (default: 100)
        custom_parser : callable, optional
            Custom filename parser function
        max_memory_gb : float, optional
            Maximum memory per worker in GB. If None, uses 80% of available.
        """
        self.base_config = base_config
        self.wav_files = wav_files
        self.num_runs = num_runs
        self.custom_parser = custom_parser
        self.max_memory_gb = max_memory_gb

        # Create descriptive output directory based on config parameters
        timestamp = generate_timestamp()
        lsr, msr, hsr = base_config.num_ANF
        folder_name = (f"multirun_{num_runs}runs_"
                      f"{base_config.num_cf}cf_"
                      f"{hsr}-{msr}-{lsr}anf_"
                      f"{len(wav_files)}files_{timestamp}")

        self.base_output_dir = os.path.join(base_config.output_dir, folder_name)

        # Create base directory
        os.makedirs(self.base_output_dir, exist_ok=True)
        print(f"Created multi-run base directory: {self.base_output_dir}")

        # Track timing and stats
        self.start_time = None
        self.end_time = None
        self.run_stats = []  # List to store per-run statistics

    def generate_unique_seed(self, run_idx):
        """
        Generate deterministic but unique seed for each run.

        Parameters
        ----------
        run_idx : int
            Run index (0 to num_runs-1)

        Returns
        -------
        int
            Unique seed value
        """
        return (self.base_config.seed + run_idx * 12345) % (2**32)

    def run_single(self, run_idx):
        """
        Execute single run with unique seed.

        Parameters
        ----------
        run_idx : int
            Run index

        Returns
        -------
        dict
            Run statistics including runtime and memory usage
        """
        run_start_time = time.time()

        # Start memory monitoring for this worker
        monitor = MemoryMonitor(max_memory_gb=self.max_memory_gb)
        with monitor:
            # Create independent config copy
            config = copy.deepcopy(self.base_config)

            # Set unique seed
            config.seed = self.generate_unique_seed(run_idx)

            # Set run-specific output directory
            config.output_dir = os.path.join(self.base_output_dir, f"run_{run_idx:03d}")

            # Run simulation
            print(f"[Run {run_idx:03d}] Starting with seed={config.seed}")
            simulation = CochleaWavSimulation(
                config,
                self.wav_files,
                auto_parse=(self.custom_parser is not None),
                parser_func=self.custom_parser
            )
            simulation.run()

            # Clean up memory
            del simulation, config
            gc.collect()

        run_end_time = time.time()
        runtime = run_end_time - run_start_time

        # Collect statistics
        mem_stats = monitor.get_stats()
        stats = {
            'run_idx': run_idx,
            'seed': self.generate_unique_seed(run_idx),
            'runtime_seconds': round(runtime, 2),
            'peak_memory_gb': mem_stats['peak_gb'],
            'mean_memory_gb': mem_stats['mean_gb'],
            'final_memory_gb': mem_stats['final_gb'],
            'completed': True
        }

        print(f"[Run {run_idx:03d}] Completed in {runtime:.2f}s, Peak memory: {mem_stats['peak_gb']:.2f} GB")
        return stats

    def run_all_parallel(self, num_workers=None):
        """
        Execute all runs in parallel using multiprocessing.

        Parameters
        ----------
        num_workers : int, optional
            Number of parallel workers (default: cpu_count - 1)

        Returns
        -------
        list of int
            Completed run indices
        """
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)

        print(f"\n{'='*60}")
        print(f"Multi-Run PSTH Pipeline")
        print(f"{'='*60}")
        print(f"Number of runs: {self.num_runs}")
        print(f"Number of WAV files: {len(self.wav_files)}")
        print(f"Parallel workers: {num_workers}")
        print(f"Base output directory: {self.base_output_dir}")
        print(f"{'='*60}\n")

        self.start_time = time.time()

        # Create pool and execute
        pool = multiprocessing.Pool(processes=num_workers)
        try:
            self.run_stats = pool.map(self.run_single, range(self.num_runs))
            pool.close()
            pool.join()
        except Exception as e:
            pool.terminate()
            pool.join()
            raise e

        self.end_time = time.time()
        elapsed = self.end_time - self.start_time

        # Calculate aggregate statistics
        total_runtime = sum(s['runtime_seconds'] for s in self.run_stats)
        avg_runtime = total_runtime / self.num_runs
        peak_memory = max(s['peak_memory_gb'] for s in self.run_stats)
        avg_peak_memory = sum(s['peak_memory_gb'] for s in self.run_stats) / self.num_runs

        print(f"\n{'='*60}")
        print(f"All {self.num_runs} runs completed")
        print(f"Total wall time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"Average runtime per run: {avg_runtime:.2f} seconds")
        print(f"Peak memory (max across runs): {peak_memory:.2f} GB")
        print(f"Average peak memory: {avg_peak_memory:.2f} GB")
        print(f"{'='*60}\n")

        # Save comprehensive summary
        self._save_run_summary(num_workers, elapsed)

        return self.run_stats

    def _save_run_summary(self, num_workers, wall_time):
        """
        Save comprehensive summary of all runs.

        Parameters
        ----------
        num_workers : int
            Number of parallel workers used
        wall_time : float
            Total wall clock time in seconds
        """
        summary_data = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'base_output_dir': self.base_output_dir,
                'experiment_name': self.base_config.experiment_name,
            },
            'configuration': {
                'num_runs': self.num_runs,
                'num_workers': num_workers,
                'num_wav_files': len(self.wav_files),
                'wav_files': [str(f) for f in self.wav_files],
                'peripheral_fs': self.base_config.peripheral_fs,
                'fs_target': self.base_config.fs_target,
                'num_cf': self.base_config.num_cf,
                'min_cf': self.base_config.min_cf,
                'max_cf': self.base_config.max_cf,
                'num_ANF': self.base_config.num_ANF,
                'powerlaw': self.base_config.powerlaw,
                'base_seed': self.base_config.seed,
                'max_memory_gb_per_worker': self.max_memory_gb or 'auto (80% available)',
            },
            'timing': {
                'wall_time_seconds': round(wall_time, 2),
                'wall_time_minutes': round(wall_time / 60, 2),
                'total_cpu_time_seconds': round(sum(s['runtime_seconds'] for s in self.run_stats), 2),
                'average_runtime_per_run': round(sum(s['runtime_seconds'] for s in self.run_stats) / self.num_runs, 2),
                'min_runtime': round(min(s['runtime_seconds'] for s in self.run_stats), 2),
                'max_runtime': round(max(s['runtime_seconds'] for s in self.run_stats), 2),
            },
            'memory': {
                'peak_memory_gb': round(max(s['peak_memory_gb'] for s in self.run_stats), 3),
                'average_peak_memory_gb': round(sum(s['peak_memory_gb'] for s in self.run_stats) / self.num_runs, 3),
                'min_peak_memory_gb': round(min(s['peak_memory_gb'] for s in self.run_stats), 3),
                'max_peak_memory_gb': round(max(s['peak_memory_gb'] for s in self.run_stats), 3),
                'average_mean_memory_gb': round(sum(s['mean_memory_gb'] for s in self.run_stats) / self.num_runs, 3),
            },
            'per_run_stats': self.run_stats
        }

        # Save JSON file
        json_path = os.path.join(self.base_output_dir, 'run_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved detailed summary: {json_path}")

        # Save human-readable text file
        txt_path = os.path.join(self.base_output_dir, 'run_summary.txt')
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MULTI-RUN PSTH PIPELINE SUMMARY\n")
            f.write("="*70 + "\n\n")

            f.write(f"Timestamp: {summary_data['experiment_info']['timestamp']}\n")
            f.write(f"Output Directory: {summary_data['experiment_info']['base_output_dir']}\n")
            f.write(f"Experiment: {summary_data['experiment_info']['experiment_name']}\n\n")

            f.write("CONFIGURATION:\n")
            f.write("-" * 70 + "\n")
            for key, value in summary_data['configuration'].items():
                if key != 'wav_files':  # Skip file list in main section
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

            f.write("TIMING STATISTICS:\n")
            f.write("-" * 70 + "\n")
            for key, value in summary_data['timing'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            f.write("MEMORY STATISTICS:\n")
            f.write("-" * 70 + "\n")
            for key, value in summary_data['memory'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            f.write("PER-RUN DETAILS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Run':<6} {'Seed':<12} {'Runtime(s)':<12} {'Peak Mem(GB)':<15} {'Mean Mem(GB)':<15}\n")
            f.write("-" * 70 + "\n")
            for stat in self.run_stats:
                f.write(f"{stat['run_idx']:<6} {stat['seed']:<12} "
                       f"{stat['runtime_seconds']:<12.2f} {stat['peak_memory_gb']:<15.3f} "
                       f"{stat['mean_memory_gb']:<15.3f}\n")

            f.write("\n" + "="*70 + "\n")

        print(f"Saved readable summary: {txt_path}")


def main():
    """Example usage with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Multi-run PSTH pipeline for cochlea simulations"
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=100,
        help='Number of independent runs (default: 100)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: cpu_count - 1)'
    )
    parser.add_argument(
        '--max-memory',
        type=float,
        default=None,
        help='Maximum memory per worker in GB (default: 80%% of available)'
    )
    args = parser.parse_args()

    print("This is a template. Import MultiRunPSTH class for usage.")
    print(f"Configured for {args.num_runs} runs with {args.num_workers or 'auto'} workers")
    print(f"Memory limit per worker: {args.max_memory or 'auto (80% available)'} GB")


if __name__ == '__main__':
    main()
