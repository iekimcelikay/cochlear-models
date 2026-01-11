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

    def get_cochlea_kwargs(self, include_anf_num: bool = True):
        """Build keyword arguments for zilany2014 functions.

        Args:
            include_anf_num: If True, includes anf_num and seed for run_zilany2014().
                            If False, includes anf_types for run_zilany2014_rate().

        Returns:
            Dictionary with appropriate parameters.
        """
        kwargs = {
            'cf': (self.min_cf, self.max_cf, self.num_cf),
            'species': self.species,
            'cohc': self.cohc,
            'cihc': self.cihc,
            'powerlaw': self.powerlaw,
            'ffGn': self.ffGn
        }

        if include_anf_num:
            # For run_zilany2014 (spike trains)
            kwargs['anf_num'] = self.num_ANF
            kwargs['seed'] = self.seed
        else:
            # For run_zilany2014_rate (mean rates)
            kwargs['anf_types'] = list(self.anf_types)

        return kwargs

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
import thorns.waves as wv

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

        # Calculate cf_list then to pass to rate function
        self._cf_list = calc_cfs(
            cf=(self.config.min_cf, self.config.max_cf, self.config.num_cf),
            species=self.config.species
            )
        logger.debug(f"CF list: {len(self._cf_list)} CFs from {self._cf_list[0]:.1f} to {self._cf_list[-1]:.1f} Hz")

    # ============ Pipeline Step 1: Run Model ============

    def _run_cochlea_model(self, tone: np.ndarray) -> pd.DataFrame:
        """
        Returns:
            DataFrame with columns: ['cf', 'type', 'duration', 'spikes']
        """
        trains = run_zilany2014(
            tone,
            self.config.peripheral_fs,
            **self.config.get_cochlea_kwargs(include_anf_num=True)
            )
        trains.sort_values(['cf', 'type'], ascending=[True, True], inplace=True)
        return trains

    def _run_cochlea_rate_model(self, tone: np.ndarray) -> pd.DataFrame:
        """
        To use the analytical estimation of synaptic rate from synaptic output.

        Returns instantaneous rates at each time sample (peripheral_fs), NOT mean rates
        nor individual ANF responses.

        Args:
            tone: Audio stimulus array

        Returns:
            DataFrame with MultiIndex columns: (anf_type, cf)
            Shape: (time_samples, num_cf * num_anf_types)
            Each column represents one (anf_type, CF) combination

        """
        rates = run_zilany2014_rate(
            tone,
            self.config.peripheral_fs,
            **self.config.get_cochlea_kwargs(include_anf_num=False)
        )

        # rates is a DataFrame with MultiIndex columns: (anf_type, cf)
        # To get mean rates: rates.mean(axis=0) returns Series with MultiIndex

        return rates


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
        Process a wav file through the zilany2014 rate function (analytical estimation)

        Return mean firing rates ( averaged over time ) per CF and fiber type.

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
            - cf_list: Array of CFs
            - duration: Sound duration
            - identifier: Sound identifier
            - metadata: Any provided metadata

        """
        logger.info(f"Processing sound: {identifier}")

        try:
            # Get instantaneous rates (returns DataFrame with MultiIndex columns)
            rates_df = self._run_cochlea_rate_model(sound)
            # rates_df columns: MultiIndex[(anf_type, cf), ...]
            # rates_df.shape = (time_samples, num_cf * num_anf_types)

            # Calculate mean rate over time for each channel
            mean_rates_df = rates_df.mean(axis=0)  # Series with MultiIndex

            # Calculate duration from sound array
            duration = len(sound) / self.config.peripheral_fs

            # Organize by fiber type
            mean_rates_dict = {}
            for fiber_type in self.config.anf_types:
                # Extract all CFs for this fiber type
                # MultiIndex selection: (anf_type, cf)
                fiber_rates = mean_rates_df.xs(fiber_type, level='anf_type')
                # Sort by CF to ensure correct order
                fiber_rates = fiber_rates.sort_index()
                mean_rates_dict[fiber_type] = fiber_rates.values

            result = {
                'soundfileid': identifier,
                'mean_rates': mean_rates_dict if self.config.save_mean_rates else None,
                'cf_list': self._cf_list, # Use cached CF list from cochlea
                'duration': duration,
                }

            # Include any provided metadata
            if metadata:
                result['metadata'] = metadata

            return result

        except Exception as e:
            logger.error(f"Failed processing {identifier}: {e}")
            raise

# ================ Shared Simulation Base CLass ================


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
    Docstring for CochleaSimulation
    """

    def __init__(
            self,
            config: CochleaConfig,
            ):

        super().__init__(config)
        self.name_builder = self._custom_folder_name_builder
        self.sound_gen = SoundGen(config.sample_rate, config.tau_ramp)
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

        # Use base class method for logging setup
        self._setup_logging_and_savers()


        # Save configuration metadata
        config_metadata = {
            'processing_type': 'tone_generation',
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
            self.config.db_range,
            self.config.freq_range,
            **self.config.get_stimulus_params()
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

    def __init__(self, config: CochleaConfig, wav_files: list, metadata_dict: dict = None):
        super().__init__(config)
        self.wav_files = wav_files
        self.metadata_dict = metadata_dict or {}
        logger.info(f"Initialized WavSimulation with {len(wav_files)} files")

    def setup_output_folder(self):
        """Setup output folder - delegates logging to base class."""
        # Create simple folder for WAV files
        timestamp_gen = TimestampGenerator()
        timestamp = timestamp_gen.generate_timestamp()
        lsr, msr, hsr = self.config.num_ANF
        folder_name = f"wav_{self.config.num_cf}cf_{lsr}-{msr}-{hsr}anf_{len(self.wav_files)}files_{timestamp}"

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
            'num_ANF': self.config.num_ANF,
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

            # Organize results
            result_data = {
                'soundfileid': identifier,
                'cf_list': result['cf_list'],
                'duration': result['duration'],
                'fiber_types': list(self.config.anf_types),
                'metadata': metadata,
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
            logger.info(f"Completed: {identifier} (data saved & cleared from RAM)")

        # Save runtime info (use base class method)
        elapsed_time = time.time() - start_time
        self._save_runtime_info(elapsed_time, stim_count)

        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Processed {stim_count} stimuli")
        logger.info(f"Results saved to: {self.save_dir}")
        logger.info("=" * 60)

        return self.results


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


