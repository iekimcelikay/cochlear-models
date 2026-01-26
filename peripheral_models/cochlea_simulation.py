#CochleaConfig dataclass

# include fields for peripheral_fs, min_cf, max_cf, num_cf, num_ANF
# stimulus parameters: sample_rate, num_harmonics, duration,
# harmonic_factor, tau_ramp
# stimulus levels and frequencies: db_range, freq_range
# sampling/downsampling: fs_target, downsample_rate
# random seed management: base_seed, run_index
# output management: output_dir,
# multiprocessing settings: num_cores -> this does not exist now/


import gc
import numpy as np
import logging
import sys
import time
import pandas as pd
import soundfile as sf
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
from cochlea.zilany2014 import run_zilany2014
from cochlea.zilany2014 import run_zilany2014_rate
from cochlea.zilany2014.util import calc_cfs # noqa: E402

# Add parent directories to path for imports to resolve modules
# - project_root: parent of models/ for loggers, stim_generator
#

_project_root = Path(__file__).parent.parent.parent

_models_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_models_dir))

# Project-level imports
from utils.folder_management import FolderCreator, FolderManager
from utils.logging_configurator import LoggingConfigurator
from utils.result_saver import ResultSaver
from utils.metadata_saver import    MetadataSaver
from stimuli.soundgen import SoundGen  # noqa: E402
from utils.stimulus_utils import (
    generate_stimuli_params,
    generate_tone_generator)
from utils.timestamp_utils import generate_timestamp


from peripheral_models.cochlea_config import CochleaConfig
from peripheral_models.tone_config import ToneConfig
from peripheral_models.cochlea_processor import CochleaProcessor, calculate_population_rate

# ================ Shared Simulation Base CLass ================

logger = logging.getLogger(__name__)
class _SimulationBase:
    """
    Shared logic for simulations using .wav files or generating stimuli.
    NOt meant to be used directly.
    """
    def __init__(self, config: CochleaConfig):
        self.config = config
        self.processor = CochleaProcessor(config)
        self.results = {}
        self.save_dir = None
        self.metadata_saver = None
        self.result_saver = None

    def _setup_logging_and_savers(self):
        """ Setup logging and saver instances. """
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        format_map = {
            'DEFAULT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'SIMPLE': '%(levelname)s: %(message)s',
            'DETAILED': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            }

        logfile_config = LoggingConfigurator(
            output_dir=self.save_dir,
            log_filename=self.config.log_filename,
            file_level=level_map.get(self.config.log_file_level, logging.INFO),
            console_level=level_map.get(self.config.log_console_level, logging.INFO),
            format_string=format_map.get(self.config.log_format, format_map['DEFAULT'])
        )
        logfile_config.setup()
        logger.info(f"Logging configured: {self.save_dir / self.config.log_filename}")

        self.metadata_saver = MetadataSaver()
        self.result_saver = ResultSaver(self.save_dir)

    def _save_metadata(self, data:dict, base_filename: str):
        """Save metadata. """
        if self.config.metadata_format == 'json':
            self.metadata_saver.save_json(self.save_dir, data, f"{base_filename}.json")
        elif self.config.metadata_format == 'yaml':
            self.metadata_saver.save_yaml(self.save_dir, data, f"{base_filename}.yaml")
        else:
            self.metadata_saver.save_text(self.save_dir, data, f"{base_filename}.txt")

    def _save_single_result(self, data: dict, filename_base: str):
        """Save individual result file (shared code)."""

        if 'npz' in self.config.save_formats:
            self.result_saver.save_npz(data, f"{filename_base}.npz")
        if 'pkl' in self.config.save_formats:
            self.result_saver.save_pickle(data, f"{filename_base}.pkl")
        if 'mat' in self.config.save_formats:
            self.result_saver.save_mat(data, f"{filename_base}.mat")

    def _save_runtime_info(self, elapsed_time: float, stim_count: int):
        """Save runtime info (shared code)."""
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        hours = int(minutes // 60)
        minutes = int(minutes % 60)
        time_formatted = f"{hours}h {minutes}m {seconds:.2f}s" if hours > 0 else f"{minutes}m {seconds:.2f}s"

        runtime_metadata = {
            'elapsed_time_seconds': round(elapsed_time, 2),
            'elapsed_time_formatted': time_formatted,
            'num_stimuli_processed': stim_count,
            'total_results': len(self.results),
        }
        self._save_metadata(runtime_metadata, 'runtime_info')

# ================= Simulation Orchestrator (Uses Utils) ================

class CochleaSimulation(_SimulationBase):
    """
    Simulation workflow for generated tone stimuli.

    Uses CochleaConfig for model parameters and ToneConfig for stimulus generation.
    """

    def __init__(
            self,
            config: CochleaConfig,
            tone_config: ToneConfig,
            ):

        super().__init__(config)
        self.tone_config = tone_config
        self.name_builder = self._custom_folder_name_builder
        self.sound_gen = SoundGen(self.tone_config.sample_rate, self.tone_config.tau_ramp)
        logger.info("Initialized CochleaSimulation")

    def _custom_folder_name_builder(self, params, timestamp):
        """Build folder name with num_cf, num_ANF, and num_tones."""
        num_cf = params.get('num_cf', 0)
        num_anf = params.get('num_ANF', (0, 0, 0))
        num_tones = params.get('num_tones', 0)

        # Format: results_10cf_32-32-32anf_5tones_20251225_123456
        lsr, msr, hsr = num_anf if isinstance(num_anf, tuple) else (num_anf, num_anf, num_anf)
        return f"results_{num_cf}cf_{lsr}-{msr}-{hsr}anf_{num_tones}tones_{timestamp}"

    def setup_output_folder(self):
        """
        Setup output folder for tone simulation.
        """
        # Use FolderManager to create organized output directory
        folder_manager = FolderManager(
            base_dir = self.config.output_dir,
            name_builder=self.name_builder,
            )

        # Add parameters from both configs for folder name
        folder_params = {
            'num_cf': self.config.num_cf,
            'num_ANF': self.config.num_ANF,
            **self.tone_config.get_folder_params()
        }
        folder_manager.with_params(**folder_params)

        # Create folder (skip stimulus_params.txt - info is in simulation_config)
        self.save_dir = Path(folder_manager.create_folder(save_text=False))
        logger.info(f"Created output directory: {self.save_dir}")

        # Use base class method for logging setup
        self._setup_logging_and_savers()


        # Save configuration metadata (both model and stimulus)
        config_metadata = {
            'processing_type': 'tone_generation',
            # Model parameters
            'peripheral_fs': self.config.peripheral_fs,
            'min_cf': self.config.min_cf,
            'max_cf': self.config.max_cf,
            'num_cf': self.config.num_cf,
            'num_ANF': self.config.num_ANF,
            'anf_types': self.config.anf_types,
            'species': self.config.species,
            'cohc': self.config.cohc,
            'cihc': self.config.cihc,
            'powerlaw': self.config.powerlaw,
            'ffGn': self.config.ffGn,
            'fs_target': self.config.fs_target,
            'seed': self.config.seed,
            # Stimulus parameters
            'sample_rate': self.tone_config.sample_rate,
            'tone_duration': self.tone_config.tone_duration,
            'num_harmonics': self.tone_config.num_harmonics,
            'db_range': self.tone_config.db_range,
            'freq_range': self.tone_config.freq_range,
            }
        self._save_metadata(config_metadata, 'simulation_config')
        logger.info("Saved cochlea simulation configuration metadata")


    def run(self) -> Dict:
        """
        Run cochlea simulation pipeline.

        Returns:
            Dictionary of lightweight result references (not full data)
        """
        logger.info("=" * 60)
        logger.info("Starting Cochlea Simulation")
        logger.info("=" * 60)

        start_time = time.time()

        # Setup output folder and logging
        if self.save_dir is None:
            self.setup_output_folder()

        # Generate stimuli using tone_generator
        logger.info("Generating stimuli...")
        tone_generator = generate_tone_generator(
            self.sound_gen,
            self.tone_config.db_range,
            self.tone_config.freq_range,
            **self.tone_config.get_stimulus_params()
            )

        # Process through cochlea-zilany2014
        logger.info("Processing through Zilany2014 model...")
        stim_count = 0

        for result in self.processor.process(tone_generator):
            stim_count += 1

            freq = result['freq']
            db = result['db']
            stim_id = f"freq_{freq:.1f}hz_db_{db}"



            stim_data = {
                'freq': freq,
                'db': db,
                'cf_list': result['cf_list'],
                'duration': result['duration'],
                'time_axis': result['time_axis'] if self.config.save_psth else None,
                'mean_rates': result['mean_rates'] if self.config.save_mean_rates else None,
                'psth': result['psth'] if self.config.save_psth else None,
            }

            # Save immediately (one file per fiber type per stimulus)
            filename = f"{self.config.experiment_name}_{stim_id}"
            self._save_single_result(stim_data, filename)

            # Store lightweight reference only (no large arrays)
            self.results[stim_id] = {
                'freq': freq,
                'db': db,
                'saved_to': filename,
                'cf_list': result['cf_list']
                }

            # CRITICAL: Delete result after all fiber types are saved
            del result, stim_data
            gc.collect()

            logger.info(f"Completed stimulus {stim_count}: {freq:.1f} Hz @ {db} dB (saved & cleared from RAM)")

        # Save runtime info using base class method
        elapsed_time = time.time() - start_time
        self._save_runtime_info(elapsed_time, stim_count)

        # Log completion
        logger.info(f"Simulation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Processed {stim_count} stimuli")
        logger.info(f"Results saved to: {self.save_dir}")
        logger.info("=" * 60)

        return self.results



# ================= CochleaWavSimulation


class CochleaWavSimulation(_SimulationBase):
    """
    Docstring for CochleaWavSimulation
    """
    @staticmethod
    def parse_wav_filename(filename:str) -> dict:
        """
        Parse filenames like (s3_animal_1_ramp10.wav)
        """

        stem = filename.replace('wav', '')
        parts = stem.split('_')

        if len(parts) >= 4:
            return {
                'sound_number': parts[0],
                'sound_type': parts[1],
                'type_number': parts[2],
                'ramp_id': parts[3]
            }
        # Return empty dict instead of failing
        logger.warning(f"Could not parse metadata from filename: {filename}")
        return {}

    def __init__(
            self,
            config: CochleaConfig,
            wav_files: list,
            metadata_dict: dict = None,
            auto_parse: bool = False,
            parser_func = None
            ):


        super().__init__(config)
        self.wav_files = wav_files

        # Handle metadata
        if metadata_dict is not None:
            # User provided explicit metadata
            self.metadata_dict = metadata_dict
            logger.info(f"Using provided metadata for {len(metadata_dict)} files")

        elif auto_parse:
            # Auto-parse filenames
            parser = parser_func if parser_func is not None else self.parse_wav_filename
            self.metadata_dict = {}

            for wav_file in wav_files:
                identifier = wav_file.stem
                self.metadata_dict[identifier] = parser(wav_file.name)

            logger.info(f"Auto-parsed metadata for {len(self.metadata_dict)} files")

        else:
            # No metadata - just juse empty dicts
            self.metadata_dict = {}
            logger.info("No metadata provided - will save empty metadata dicts")

        logger.info(f"Initialized zilany2014 WavSimulation with {len(wav_files)} files")


    def setup_output_folder(self):
        """Setup output folder - delegates logging to base class."""
        # Create simple folder for WAV files

        timestamp = generate_timestamp()
        lsr, msr, hsr = self.config.num_ANF
        folder_name = f"wav_{self.config.num_cf}cf_analyticalmeanrate_{len(self.wav_files)}files_{timestamp}"

        self.save_dir = Path(self.config.output_dir) / folder_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {self.save_dir}")

        # Use base class method for logging setup
        self._setup_logging_and_savers()

        # Save configuration metadata (specific to WAV processing)
        config_metadata = {
            'processing_type': 'wav_files',
            'num_files': len(self.wav_files),
            'files': [p.name for p in self.wav_files],
            'peripheral_fs': self.config.peripheral_fs,
            'num_cf': self.config.num_cf,
            'anf_types': self.config.anf_types,
            'species': self.config.species,
            'cohc': self.config.cohc,
            'cihc': self.config.cihc,
        }
        self._save_metadata(config_metadata, 'simulation_config')
        logger.info("Saved configuration metadata")

    def run(self) -> Dict:
        """Process all WAV files."""
        logger.info("=" * 60)
        logger.info("Starting WAV File Processing")
        logger.info("=" * 60)

        start_time = time.time()

        if self.save_dir is None:
            self.setup_output_folder()

        stim_count = 0

        for wav_path in self.wav_files:
            identifier = wav_path.stem
            logger.info(f"Processing: {wav_path.name}")

            # Load and resample
            audio, fs = sf.read(wav_path)
            if fs != self.config.peripheral_fs:
                logger.info(f"Resampling from {fs} Hz to {self.config.peripheral_fs} Hz")
                audio = wv.resample(audio, fs, self.config.peripheral_fs)

            # Get metadata
            metadata = self.metadata_dict.get(identifier, {})

            # Process using process_wavfile
            result = self.processor.process_wavfile(audio, identifier, metadata)

            # Calculate population rate
            population_rate = calculate_population_rate(result['mean_rates'])

            # Organize results
            result_data = {
                'soundfileid': identifier,
                'cf_list': result['cf_list'],
                'duration': result['duration'],
                'fiber_types': list(self.config.anf_types),
                'metadata': metadata,
                'population_rate': population_rate,
            }

            if self.config.save_mean_rates:
                result_data['mean_rates'] = result['mean_rates']

            # Note: process_wavfile uses rate model which doesn't produce PSTH
            # time_axis and psth are not available for rate-based processing

            filename = f"{self.config.experiment_name}_{identifier}"
            # Save immediately (use base class method)
            self._save_single_result(result_data, filename)

            # Store lightweight reference only (not full data)
            # This allows iteration but doesn't consume RAM
            self.results[identifier] = {
                'soundfileid': identifier,
                'saved_to': filename,
                'cf_list': result_data['cf_list'],  # Small array, OK to keep
            }

            # CRITICAL: Delete large data immediately after saving
            del result, result_data
            import gc
            gc.collect()

            stim_count += 1
            logger.info(f"Completed: {identifier} (data saved (including population rates) & cleared from RAM)")

        # Save runtime info (use base class method)
        elapsed_time = time.time() - start_time
        self._save_runtime_info(elapsed_time, stim_count)

        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Processed {stim_count} stimuli")
        logger.info(f"Results saved to: {self.save_dir}")
        logger.info("=" * 60)

        return self.results


    # ================= Testing ================
    # For the soundgen using simulation


if __name__ == '__main__':
    # Model configuration (stimulus-agnostic)
    model_config = CochleaConfig(
        num_cf=10,
        num_ANF=(12,12,12),
        powerlaw= 'approximate',
        output_dir="./models_output/cochlea_test013_approximate",
        experiment_name="test013",    # Prefix for all output files

        # Output format control
        save_formats=['npz'],         # Only save NPZ files
        metadata_format='txt',        # Save metadata as text
        save_psth=True,               # Include PSTH arrays
        save_mean_rates=True,         # Include mean rates

        # Logging configuration
        log_level='INFO',             # Overall log level (not used directly)
        log_format='DEFAULT',         # DEFAULT, SIMPLE, or DETAILED
        log_console_level='INFO',     # Console output level
        log_file_level='DEBUG',       # File output level (more detailed)
        log_filename='simulation.log' # Name of log file
        )

    # Tone stimulus configuration
    tone_config = ToneConfig(
        db_range=[60],
        freq_range=(125, 2500, 10),
        tone_duration=0.200,
        num_harmonics=1,
        harmonic_factor=0.5,
        sample_rate=100000,
        tau_ramp=0.005
    )

    # Run simulation with both configs
    simulation = CochleaSimulation(model_config, tone_config)
    results = simulation.run()

    num_tones = tone_config.freq_range[2]  # Number of frequencies from freq_range tuple
    print(f"\nProcessed {len(results)} total results (3 fiber types x {len(tone_config.db_range)} dB levels x {num_tones} tones)")


