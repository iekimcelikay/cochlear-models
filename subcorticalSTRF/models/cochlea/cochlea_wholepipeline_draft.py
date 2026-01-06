#CochleaConfig dataclass

# include fields for peripheral_fs, min_cf, max_cf, num_cf, num_ANF
# stimulus parameters: sample_rate, num_harmonics, duration,
# harmonic_factor, tau_ramp
# stimulus levels and frequencies: db_range, freq_range
# sampling/downsampling: fs_target, downsample_rate
# random seed management: base_seed, run_index
# output management: output_dir,
# multiprocessing settings: num_cores

# new cochlea is in my local folder, subcorticalSTRF/cochlea

import numpy as np
import logging
import sys
import time
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
from cochlea.zilany2014 import run_zilany2014

# Add parent directories to path for imports to resolve modules
# - project_root: parent of models/ for loggers, stim_generator
#

_project_root = Path(__file__).parent.parent.parent

_models_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_models_dir))

# Project-level imports
from utils import (FolderCreator, FolderManager, LoggingConfigurator,
                   MetadataSaver, ResultSaver, TimestampGenerator)  # noqa: E402
from stim_generator.soundgen import SoundGen  # noqa: E402
from utils.stimulus_utils import (
    generate_stimuli_params,
    generate_tone_generator)



# dataclass for cochlea configuration
@dataclass
class CochleaConfig:
    # ================ Cochlea Model Parameters ================
    peripheral_fs: float = 100000
    min_cf: float = 125
    max_cf: float = 2500
    num_cf: int = 40
    num_ANF: tuple = (128, 128, 128)
    anf_types: tuple = ('lsr', 'msr', 'hsr')
    species: str = 'human'
    cohc: float = 1.0
    cihc: float = 1.0
    powerlaw: str = 'approximate'
    ffGn: bool = True
    seed: int = 0  # Random seed for reproducibility
    fs_target: float = 100.0  # Target sampling rate for downsampled PSTH

    # ================ Stimulus Parameters ================
    sample_rate: int = 100000
    tau_ramp: float = 0.005
    num_harmonics: int = 1
    tone_duration: float = 0.200
    harmonic_factor: float = 0.5
    db_range: List[int] = field(default_factory=lambda: [60])
    freq_range: Tuple[float, float, int] = (125.0, 2500.0, 40)

    # ============ Output management ================
    output_dir: str = "./models_output"  # Output under models/output/
    experiment_name: str = "cochlea_experiment"  # Base name for output files
    save_formats: List[str] = field(default_factory=lambda: ['npz', 'pkl', 'mat'])
    metadata_format: str = 'txt'  # 'txt', 'json', or 'yaml'
    save_psth: bool = True
    save_mean_rates: bool = True

    # ============ Logging Configuration ================
    log_level: str = 'INFO'           # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    log_format: str = 'DEFAULT'       # 'DEFAULT', 'SIMPLE', 'DETAILED'
    log_console_level: str = 'INFO'   # Console output level
    log_file_level: str = 'INFO'      # File output level
    log_filename: str = 'simulation.log'  # Log filename

    def get_cochlea_kwargs(self):
        """Build keyword arguments for cochlea.run_zilany2014().

        Returns:
            Dictionary with all parameters needed by the cochlea function.
        """
        return {
            'anf_num': self.num_ANF,
            'cf': (self.min_cf, self.max_cf, self.num_cf),
            'species': self.species,
            'seed': self.seed,
            'cohc': self.cohc,
            'cihc': self.cihc,
            'powerlaw': self.powerlaw,
            'ffGn': self.ffGn
        }

    def get_stimulus_params(self):
        """ For stimulus generation."""
        return {
            'num_harmonics': self.num_harmonics,
            'duration': self.tone_duration,
            'harmonic_factor': self.harmonic_factor
            }

    def get_folder_params(self):
        """For FolderManager.with_params()."""
        # Calculate number of tones from freq_range
        num_tones = self.freq_range[2]  # freq_range = (min_freq, max_freq, num_freqs)

        return {
            'num_cf': self.num_cf,
            'num_ANF': self.num_ANF,
            'num_tones': num_tones,
            'peripheral_fs': self.peripheral_fs,
            'db_range': self.db_range,
            'freq_range': self.freq_range
            }

# Usage:
# trains = cochlea.run_zilany2014(sound, fs, **config.get_cochlea_kwargs())
# tone = sound_gen.sound_maker(freq, **config.get_stimulus_params())
# folder_mgr.with_params(**config.get_folder_params())
# use the tone_generator directly


# ==================== Cochlea Processor ====================

import thorns as th

logger = logging.getLogger(__name__)


class CochleaProcessor:
    """Process audio stimuli through Cochlea (Zilany 2014) model.

    Generates auditory nerve fiber spike trains using the cochlea package.
    Lazy evaluation - processes one tone at a time.
    """

    def __init__(self, config: CochleaConfig):
        """Initialize processor.

        Args:
            config: CochleaConfig instance with model parameters.
        """
        self.config = config
        self._time_axis = None # Cache time axis

    # ============ Pipeline Step 1: Run Model ============

    def _run_cochlea_model(self, tone: np.ndarray) -> pd.DataFrame:
        """
        Returns:
            DataFrame with columns: ['cf', 'type', 'duration', 'spikes']
        """
        trains = run_zilany2014(
            tone,
            self.config.peripheral_fs,
            **self.config.get_cochlea_kwargs()
            )
        trains.sort_values(['cf', 'type'], ascending=[True, True], inplace=True)
        return trains

    # ============ Pipeline Step 2: Convert Format ==============

    def _convert_to_array(self, trains: pd.DataFrame) -> tuple:
        """Convert spike trains DataFrame to array format."""

        spike_array = th.trains_to_array(trains, self.config.peripheral_fs).T
        cf_list = trains['cf'].unique()
        duration = trains.iloc[0]['duration']
        return spike_array, cf_list, duration

    # ============ Pipeline Step 3: Aggregate by Fiber Type ============

    def _aggregate_by_fiber_type(
            self,
            trains: pd.DataFrame,
            spike_array: np.ndarray,
            cf_list: np.ndarray,
            duration: float
            ) -> tuple:
        """ Aggregate responses by CF and fiber type.

        Returns:
            mean_rates: Dict[fiber_type, array of shape (num_cf,)]
            psth_resampled: Dict[fiber_type, array of shape (num_cf, n_bins)]
            """
        mean_rates = {}
        psth_resampled = {}

        # Calculate dimensions once
        n_cf = len(cf_list)
        bin_samples = int(round(self.config.peripheral_fs / self.config.fs_target))
        n_bins = int(np.floor(spike_array.shape[1] / bin_samples))

        for fiber_type in self.config.anf_types:
            # Filter for this fiber type
            type_mask = trains['type'].values == fiber_type
            type_spike_array = spike_array[type_mask]
            type_cfs = trains.loc[type_mask, 'cf'].values

            # Initialize outputs
            cf_mean_rates = np.zeros(n_cf)
            cf_psth = np.zeros((n_cf, n_bins))

            # Process each CF
            for i_cf, cf in enumerate(cf_list):
                cf_mask = type_cfs == cf
                cf_trains = type_spike_array[cf_mask] # (n_fibers_at_cf, n_samples)

                # Compute mean rate (Hz)
                cf_mean_rates[i_cf] = cf_trains.sum()

                # Average PSTH across fibers, then resample
                avg_psth = cf_trains.mean(axis=0)
                cf_psth[i_cf], self._time_axis = self._resample_psth(
                    avg_psth,
                    self.config.peripheral_fs,
                    self.config.fs_target
                    )

            mean_rates[fiber_type] = cf_mean_rates
            psth_resampled[fiber_type] = cf_psth

        return mean_rates, psth_resampled

    # ============ Pipeline Step 4: Resample PSTH ============

    @staticmethod
    def _resample_psth(spikes: np.ndarray, fs: float, target_fs: float) -> tuple:
        """ Resample PSTH by binning spikes at target sampling rate.

        Args:
            spikes: Spike train array (can be binary or averaged)
            fs: Original sampling freq (Hz)
            target_fs: Target sampling frequency (Hz)

        Returns:
            spike_rates: Array of spike rates at each time bin
            time_axis: Time axis (seconds)
            """
        bin_width_s = 1 / target_fs
        bin_samples = int(round(fs * bin_width_s))
        n_bins = int(np.floor(len(spikes) / bin_samples))

        # Vectorized binning
        spikes_truncated = spikes[:n_bins * bin_samples]
        spike_counts = spikes_truncated.reshape(n_bins, bin_samples).sum(axis=1)
        spike_rates = spike_counts / bin_width_s

        time_axis = np.arange(n_bins) * bin_width_s

        return spike_rates, time_axis

    # ============ Main Processing Loop ============

    def process(self, stimuli):
        """
        Docstring for process

        Args:
            stimuli: Iterator of (db, freq, tone) tuples

        Yields:
            dict: Results for each stimulus containing:
            - db, freq: Stimulus parameters
            - mean_rates: Dict[fiber_type, array(num_cf)]
            - psth: Dict[fiber_type, array(num_cf, n_bins)]
            - cf_list: Array of CFs
            - duration: Stimulus duration
            - time_axis: Time axis for PSTH
        """
        for db, freq, tone in stimuli:
            logger.info(f"Processing {freq:.1f} Hz @ {db} dB")

            try:
                # Pipeline execution
                trains = self._run_cochlea_model(tone)
                spike_array, cf_list, duration = self._convert_to_array(trains)
                mean_rates, psth = self._aggregate_by_fiber_type(
                    trains, spike_array, cf_list, duration
                    )

                yield {
                    'db': db,
                    'freq': freq,
                    'mean_rates': mean_rates,
                    'psth': psth,
                    'cf_list': cf_list,
                    'duration':duration,
                    'time_axis': self._time_axis
                    }

                #Memory cleanup
                del trains, spike_array

            except Exception as e:
                logger.error(f"Failed procesing {freq:.1f}Hz @ {db}dB: {e}")
                raise

    def process_wavfile(self, sound: np.ndarray,
                     identifier: str = "sound",
                        metadata: dict = None):
        """
        Docstring for process_wavfile

        :param self: Description
        :param sound: Description
        :type sound: np.ndarray
        :param identifier: Description
        :type identifier: str
        :param metadata: Description
        :type metadata: dict

        Returns:
            dict: Results containing:
            - mean_rates: Dict[fiber_type, array(num_cf)]
            - psth: Dict[fiber_type, array(num_cf, n_bins)]
            - cf_list: Array of CFs
            - duration: Sound duration
            - time_axis: Time axis for PSTH
            - identifier: Sound identifier
            - metadata: Any provided metadata

        """
        logger.info(f"Processing sound: {identifier}")

        try:
            trains = self._run_cochlea_model(sound)
            spike_array, cf_list, duration = self._convert_to_array(trains)
            mean_rates, psth = self._aggregate_by_fiber_type(
                trains, spike_array, cf_list, duration
                )

            result = {
                'soundfileid': identifier,
                'mean_rates': mean_rates,
                'psth': psth,
                'cf_list': cf_list,
                'duration': duration,
                'time_axis': self._time_axis,
                }

            # Include any provided metadata
            if metadata:
                result['metadata'] = metadata

            # Memory cleanup
            del trains, spike_array

            return result

        except Exception as e:
            logger.error(f"Failed processing {identifier}: {e}")
            raise


# ================= Simulation Orchestrator (Uses Utils) ================

class CochleaSimulation:
    """
    Docstring for CochleaSimulation
    """

    def __init__(
            self,
            config: CochleaConfig,
            ):

        self.config = config
        self.name_builder = self._custom_folder_name_builder
        self.sound_gen = SoundGen(config.sample_rate, config.tau_ramp)
        self.processor = CochleaProcessor(config)
        self.results = {}
        self.save_dir = None
        self.metadata_saver = None
        self.result_saver = None

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
        Docstring for setup_output_folder

        :param self: Description
        """
        # Use FolderManager to create organized output directory
        folder_manager = FolderManager(
            base_dir = self.config.output_dir,
            name_builder=self.name_builder,
            )

        # Add parameters to folder name
        folder_manager.with_params(**self.config.get_folder_params())

        # Create folder (skip stimulus_params.txt - info is in simulation_config)
        self.save_dir = Path(folder_manager.create_folder(save_text=False))
        logger.info(f"Created output directory: {self.save_dir}")

        # Setup logging to file using LoggingConfigurator with config params
        # Map string levels to logging constants
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        # Map format names to actual format strings
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
        logfile = logfile_config.setup()
        logger.info(f"Logging configured. Logs saved to: {self.save_dir / self.config.log_filename}")

        # Initialize MetadataSaver
        self.metadata_saver = MetadataSaver() # Metadatasaver takes no arguments

        # Save configuration metadata
        config_metadata = {
            'name_builder': self.name_builder,
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
            'sample_rate': self.config.sample_rate,
            'tone_duration': self.config.tone_duration,
            'num_harmonics': self.config.num_harmonics,
            'db_range': self.config.db_range,
            'freq_range': self.config.freq_range,
            'seed': self.config.seed,
            }
        self._save_metadata(config_metadata, 'simulation_config')
        logger.info("Saved cochlea simulation configuration metadata")

        #Initialize ResultSaver
        self.result_saver = ResultSaver(self.save_dir)

    def _save_metadata(self, data: dict, base_filename: str):
        """Save metadata in configured format using if/elif."""
        if self.config.metadata_format == 'json':
            self.metadata_saver.save_json(self.save_dir, data, f"{base_filename}.json")
        elif self.config.metadata_format == 'yaml':
            self.metadata_saver.save_yaml(self.save_dir, data, f"{base_filename}.yaml")
        else:  # Default to txt
            self.metadata_saver.save_text(self.save_dir, data, f"{base_filename}.txt")

    def run(self) -> Dict:
        """
        Run cochlea simulation pipeline.

        Returns:
            Dictionary of results organized by stimulus and fiber type
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
            self.config.db_range,
            self.config.freq_range,
            **self.config.get_stimulus_params()
            )

        # Process through cochlea-zilany2014
        logger.info("Processing through Zilany2014 model...")
        stim_count = 0

        for result in self.processor.process(tone_generator):
            stim_count += 1

            # Store aggregated results per fiber type
            for fiber_type in self.config.anf_types:
                key = f"freq_{result['freq']:.1f}hz_db_{result['db']}_{fiber_type}"

                self.results[key] = {
                    'freq': result['freq'],
                    'db': result['db'],
                    'fiber_type': fiber_type,
                    'cf_list': result['cf_list'],
                    'mean_rate': result['mean_rates'][fiber_type] if self.config.save_mean_rates else None,
                    'psth': result['psth'][fiber_type] if self.config.save_psth else None,
                    'duration': result['duration'],
                    'time_axis': result['time_axis'] if self.config.save_psth else None
                    }

            logger.info(f"Completed stimulus {stim_count}: {result['freq']:.1f} Hz @ {result['db']} dB")

        # Save results using ResultSaver
        logger.info("Saving results...")
        self._save_results()

        # Calculate and log runtime
        elapsed_time = time.time() - start_time
        logger.info(f"Simulation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Processed {stim_count} stimuli")
        logger.info(f"Results saved to: {self.save_dir}")
        logger.info("=" * 60)

        # Save runtime metadata with time conversion
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        hours = int(minutes // 60)
        minutes = int(minutes % 60)

        time_formatted = f"{hours}h {minutes}m {seconds:.2f}s" if hours > 0 else f"{minutes}m {seconds:.2f}s"

        runtime_metadata = {
            'elapsed_time_seconds': round(elapsed_time, 2),
            'elapsed_time_formatted': time_formatted,
            'num_stimuli_processed': stim_count,
            'num_tones': len(set(data['freq'] for data in self.results.values())),
            'num_db_levels': len(set(data['db'] for data in self.results.values())),
            'num_fiber_types': len(self.config.anf_types),
            'total_results': len(self.results),
            'save_psth': self.config.save_psth,
            'save_mean_rates': self.config.save_mean_rates,
            'save_formats': self.config.save_formats,
            'output_directory': str(self.save_dir)
            }
        self._save_metadata(runtime_metadata, 'runtime_info')

        return self.results

    def _save_results(self):
        """Save results using ResultSaver, organized by stimulus."""
        # Organize results by stimulus (freq, db)
        stim_results = {}

        # Get CF list from first result (same for all stimuli)
        first_key = list(self.results.keys())[0]
        cf_list = self.results[first_key]['cf_list']

        # Populate data organized by stimulus
        for key, data in self.results.items():
            # Extract stimulus and fiber type info
            parts = key.rsplit('_', 1)
            fiber_type = parts[1]  # e.g., "hsr"

            freq = data['freq']
            db = data['db']
            stim_id = f"freq_{freq:.1f}hz_db_{db}"

            # Initialize stimulus entry if needed
            if stim_id not in stim_results:
                stim_results[stim_id] = {
                    'freq': freq,
                    'db': db,
                    'cf_list': data['cf_list'],
                    'duration': data['duration'],
                    'time_axis': data['time_axis'] if self.config.save_psth else None,
                    'fiber_types': list(self.config.anf_types),
                    'mean_rates': {} if self.config.save_mean_rates else None,
                    'psth': {} if self.config.save_psth else None
                }

            # Add this fiber type's data for all CFs
            if self.config.save_mean_rates:
                stim_results[stim_id]['mean_rates'][fiber_type] = data['mean_rate']
            if self.config.save_psth:
                stim_results[stim_id]['psth'][fiber_type] = data['psth']

        # Save using ResultSaver - one file per stimulus
        for stim_id, stim_data in stim_results.items():
            filename = f"{self.config.experiment_name}_{stim_id}"

            # Save in configured formats only
            if 'pkl' in self.config.save_formats:
                self.result_saver.save_pickle(stim_data, f"{filename}.pkl")
            if 'mat' in self.config.save_formats:
                self.result_saver.save_mat(stim_data, f"{filename}.mat")
            if 'npz' in self.config.save_formats:
                self.result_saver.save_npz(stim_data, f"{filename}.npz")

            logger.debug(f"Saved {stim_id}")

        logger.info(f"Saved {len(stim_results)} stimulus response files")

        # Save consolidated results (mean rates only)
        self._save_consolidated_results(stim_results)

    def _save_consolidated_results(self, stim_results: dict):
        """Save consolidated mean rates across all stimuli in a single file.

        Creates arrays organized by fiber type with shape (num_freq, num_db, num_cf).
        Only includes mean rates, not PSTHs, for memory efficiency.

        Args:
            stim_results: Dictionary of results organized by stimulus
        """
        if not self.config.save_mean_rates:
            logger.info("Skipping consolidated results (save_mean_rates=False)")
            return

        # Extract metadata
        first_stim = list(stim_results.values())[0]
        cf_list = first_stim['cf_list']
        fiber_types = first_stim['fiber_types']

        # Extract unique frequencies and db levels
        frequencies = sorted(set(s['freq'] for s in stim_results.values()))
        db_levels = sorted(set(s['db'] for s in stim_results.values()))

        # Initialize arrays: (num_freq, num_db, num_cf) per fiber type
        num_freq = len(frequencies)
        num_db = len(db_levels)
        num_cf = len(cf_list)

        consolidated_data = {
            'cf_list': cf_list,
            'frequencies': np.array(frequencies),
            'db_levels': np.array(db_levels),
            'fiber_types': fiber_types
        }

        # Organize mean rates by fiber type (flatten to avoid nested dict)
        for fiber_type in fiber_types:
            mean_rate_array = np.zeros((num_freq, num_db, num_cf))

            for stim_id, stim_data in stim_results.items():
                freq = stim_data['freq']
                db = stim_data['db']

                freq_idx = frequencies.index(freq)
                db_idx = db_levels.index(db)

                mean_rate_array[freq_idx, db_idx, :] = stim_data['mean_rates'][fiber_type]

            # Store as top-level key to avoid .item() unwrapping
            consolidated_data[f'mean_rates_{fiber_type}'] = mean_rate_array

        # Save consolidated file
        filename = f"{self.config.experiment_name}_consolidated_mean_rates"

        if 'npz' in self.config.save_formats:
            self.result_saver.save_npz(consolidated_data, f"{filename}.npz")
        if 'pkl' in self.config.save_formats:
            self.result_saver.save_pickle(consolidated_data, f"{filename}.pkl")
        if 'mat' in self.config.save_formats:
            self.result_saver.save_mat(consolidated_data, f"{filename}.mat")

        logger.info(f"Saved consolidated results: {num_freq} freq × {num_db} dB × {num_cf} CF")

# ================= Testing ================

if __name__ == '__main__':
    # Config simulation
    config = CochleaConfig(
        num_cf=10,
        num_ANF=(128,128,128),
        powerlaw= 'approximate',
        db_range=[60],
        freq_range=(125,2500,10),
        output_dir="./models_output/cochlea_test012_approximate",
        experiment_name="test012",    # Prefix for all output files

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

    # Run simulation
    simulation = CochleaSimulation(config)
    results = simulation.run()

    num_tones = config.freq_range[2]  # Number of frequencies from freq_range tuple
    print(f"\nProcessed {len(results)} total results (3 fiber types x {len(config.db_range)} dB levels x {num_tones} tones)")


