# run_wsr_model.py
"""
Class-based runner for running WSR Model

Uses the loggers module for organized output management, logging,
and metadata saving.
"""
import numpy as np
from pathlib import Path
import time
import logging
from dataclasses import dataclass
from typing import Dict, List
import pickle

from wsrmodel import WSRModel
from stim_generator.soundgen import SoundGen
from BEZ2018_meanrate.stimulus_utils import (generate_stimuli_params,
                                             generate_tone_generator)
from loggers import FolderManager, LoggingConfigurator, MetadataSaver
from run_wsr_model_pipeline import ResultCollector

logger = logging.getLogger(__name__)


@dataclass
class WSRConfig:
    """ Configuration for WSR model simulation"""
    # Audio parameters
    sample_rate: int = 16000
    tau_ramp: float = 0.005

    # Stimulus parameters
    num_harmonics: int = 1
    tone_duration: float = 0.200
    harmonic_factor: float = 0.5
    db_range: int = 60
    freq_range: tuple = (125, 2500, 20)

    # WSR model parameters
    # [frame_length, time_constant, factor, shift]
    wsr_params: List[int] = None
    max_cf: float = 2500.0

    # Output
    results_folder: str = None

    def __post_init__(self):
        if self.wsr_params is None:
            self.wsr_params = [8, 8, -1, 0]
        if self.results_folder is None:
            # Set results folder relative to this script's location
            script_dir = Path(__file__).parent
            self.results_folder = str(script_dir / 'results')


class WSRSimulation:

    def __init__(self, config: WSRConfig):
        self.config = config
        self.sound_gen = SoundGen(config.sample_rate, config.tau_ramp)
        self.model = WSRModel(config.wsr_params)
        self.results = {}
        self.save_dir = None

    def setup_output_folder(self):
        """Create organized output directory with metadata and logging."""
        desired_dbs, desired_freqs = generate_stimuli_params(
            self.config.freq_range, self.config.db_range
        )

        # Get channel info
        mask = self.model.get_channel_mask(
            self.config.sample_rate,
            self.config.max_cf)
        num_channels = np.sum(mask)

        # Create output folder using FolderManager
        folder_manager = FolderManager(self.config.results_folder, "wsr_model")
        folder_manager.with_params(
            num_channels=int(num_channels),
            frame_length=self.config.wsr_params[0],
            time_constant=self.config.wsr_params[1],
            factor=self.config.wsr_params[2],
            shift=self.config.wsr_params[3],
            sample_rate=self.config.sample_rate,
            tone_duration=self.config.tone_duration,
            db_range=list(desired_dbs),
            freq_range=list(desired_freqs),
            max_cf=self.config.max_cf,
            num_harmonics=self.config.num_harmonics,
            harmonic_factor=self.config.harmonic_factor
        )
        self.save_dir = Path(folder_manager.create_folder())

        # Setup logging
        LoggingConfigurator.setup_basic(self.save_dir, 'wsr_simulation.log')
        logger.info(f"Results will be saved to: {self.save_dir}")
        logger.info(f"WSR parameters: {self.config.wsr_params}")
        logger.info(f"Number of channels: {num_channels}")

    def process_tone(self, freq: float, db: float, tone: np.ndarray) -> Dict:
        """Process a single tone through WSR model."""
        # Get filtered response
        channels = self.model.process_filtered(
            tone, self.config.sample_rate, max_cf=self.config.max_cf
        )

        # Calculate temporal mean
        temporal_mean = self.model.get_temporal_mean(channels)

        return {
            'freq': freq,
            'db': db,
            'temporal_mean': temporal_mean,
            'full_response': channels,
        }

    def run(self, save_full_response: bool = False):
        """Run the full experiment."""
        logger.info("Starting WSR model experiment...")
        start_time = time.time()

        # Setup output
        self.setup_output_folder()

        # Generate tone iterator
        tone_generator = generate_tone_generator(
            self.sound_gen,
            self.config.db_range,
            self.config.freq_range,
            num_harmonics=self.config.num_harmonics,
            duration=self.config.tone_duration,
            harmonic_factor=self.config.harmonic_factor
        )

        # Process all tones
        for tone_idx, (db, freq, tone) in enumerate(tone_generator):
            logger.info(f"Processing tone {tone_idx}: {freq:.1f} Hz @ {db} dB")

            result = self.process_tone(freq, db, tone)

            # Optionally drop full response to save memory
            if not save_full_response:
                result.pop('full_response')

            key = f"freq_{freq:.1f}hz_db_{db}"
            self.results[key] = result

        # Save results
        self.save_results()

        elapsed = time.time() - start_time
        logger.info(f"Completed in {elapsed:.2f} seconds")
        logger.info(f"Processed {len(self.results)} conditions")

        return self.results

    def save_results(self):
        """Save results using the ResultCollector class and MetadataSaver."""
        logger.info("Saving results...")

        # Use the existing ResultCollector
        collector = ResultCollector(self.save_dir)
        collector.results = self.results  # Assign collected results

        # Save in npz format only (as requested - no .mat files)
        collector.save(formats=['npz'])

        # Also save as pickle (additional format)
        pkl_file = self.save_dir / "wsr_results.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.results, f)
        logger.info(f"Saved pickle format: {pkl_file}")

        # Save channel CFs
        cfs = self.model.get_filterbank_CFs(self.config.sample_rate)
        mask = self.model.get_channel_mask(
            self.config.sample_rate,
            self.config.max_cf)
        filtered_cfs = cfs[mask]

        np.save(self.save_dir / "channel_cfs.npy", filtered_cfs)
        logger.info(f"Saved {len(filtered_cfs)} channel CFs")

        # Save summary statistics using MetadataSaver
        metadata_saver = MetadataSaver()
        summary = {
            'total_conditions': len(self.results),
            'num_channels': len(filtered_cfs),
            'cf_range_hz': [float(filtered_cfs.min()),
                            float(filtered_cfs.max())],
            'wsr_params': self.config.wsr_params,
            'sample_rate': self.config.sample_rate,
            'tone_duration': self.config.tone_duration,
            'db_range': self.config.db_range,
            'freq_range': self.config.freq_range
        }
        metadata_saver.save_json(self.save_dir,
                                 summary,
                                 filename="simulation_summary.json")
        logger.info("Saved simulation summary")


# ==== Usage ====
if __name__ == "__main__":
    # Configure experiment
    config = WSRConfig(
        db_range=60,
        freq_range=(125, 2500, 20),
        tone_duration=0.2,
    )

    # Run experiment
    model_simulation = WSRSimulation(config)
    results = model_simulation.run(save_full_response=True)
    # Note: save_results() is already called inside run()
