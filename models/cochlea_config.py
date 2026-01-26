# CochleaConfig dataclass

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class CochleaConfig:
    """Configuration for Cochlea model parameters - stimulus-agnostic."""

    # ================ Cochlea Model Parameters ================
    peripheral_fs: float = 100000
    min_cf: float = 125
    max_cf: float = 2500
    num_cf: int = 40
    num_ANF: tuple = (128, 128, 128)  # (lsr, msr, hsr)
    anf_types: tuple = ('lsr', 'msr', 'hsr')
    species: str = 'human'
    cohc: float = 1.0
    cihc: float = 1.0
    powerlaw: str = 'approximate'
    ffGn: bool = True
    seed: int = 0  # Random seed for reproducibility
    fs_target: float = 100.0  # Target sampling rate for downsampled PSTH

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