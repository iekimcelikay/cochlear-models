"""
Components for model comparison pipeline.

This module contains:
- BEZResultsLoader: Loads pre-computed BEZ results (run-averaged)
- CFAligner: Aligns model outputs by matching characteristic frequencies
"""

import numpy as np
import logging
from pathlib import Path
from typing import Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .core import ModelResults
from WSRmodel.cf_matcher import CFMatcher
from compare_bez_cochlea_mainscript import MatlabDataLoader, PopulationAnalyzer

logger = logging.getLogger(__name__)


class BEZResultsLoader:
    """
    Loads pre-computed BEZ model results (run-averaged).
    
    Wraps MatlabDataLoader from compare_bez_cochlea_mainscript to provide
    ModelResults output compatible with comparison pipeline.
    """
    
    def __init__(self, bez_dir: Path = None):
        """
        Initialize BEZ results loader.
        
        Args:
            bez_dir: Directory containing bez_acrossruns_psths.mat
                    If None, uses default path from MatlabDataLoader
        """
        self.matlab_loader = MatlabDataLoader(bez_dir=bez_dir)
        self.population_analyzer = None
        self._loaded = False
    
    def load(
        self, 
        fiber_types: list = None,
        compute_population: bool = False,
        population_weights: dict = None
    ) -> ModelResults:
        """
        Load BEZ results using MatlabDataLoader.
        
        Args:
            fiber_types: List of fiber types to load (default: ['hsr', 'msr', 'lsr'])
            compute_population: If True, compute weighted population average
            population_weights: Custom weights for population (default: physiological ratios)
            
        Returns:
            ModelResults containing BEZ temporal means and metadata
        """
        if fiber_types is None:
            fiber_types = ['hsr', 'msr', 'lsr']
        
        # Load using MatlabDataLoader
        self.matlab_loader.load_bez_run_averaged(fiber_types=fiber_types)
        self._loaded = True
        
        # Get data
        temporal_means = self.matlab_loader.get_temporal_means()
        cfs = self.matlab_loader.get_cfs()
        frequencies = self.matlab_loader.get_frequencies()
        db_levels = self.matlab_loader.get_db_levels()
        
        # Compute population if requested
        if compute_population:
            self.population_analyzer = PopulationAnalyzer(
                weights=population_weights,
                normalize=True
            )
            population_means = self.population_analyzer.compute_population_from_temporal_means(
                temporal_means
            )
            # Add population to temporal_means
            temporal_means['population'] = population_means
            
            logger.info(
                f"Added weighted population with weights: "
                f"{self.population_analyzer.get_weights()}"
            )
            
            # Use population for comparison
            fiber_type_to_use = 'population'
        else:
            # Use first fiber type if no population
            fiber_type_to_use = fiber_types[0]
            logger.info(f"Using {fiber_type_to_use} fiber type for comparison")
        
        # Convert nested dict to array for the selected fiber type
        # Structure: [fiber_type][cf][freq][db] -> array[cf, condition]
        # Filter to only 60 dB to match WSR
        db_filter = [60.0]
        temporal_means_array = self._dict_to_array(
            temporal_means[fiber_type_to_use],
            cfs, frequencies, db_filter
        )
        
        logger.info(
            f"Loaded BEZ results: {len(cfs)} CFs, {len(frequencies)} freqs, "
            f"filtered to 60 dB only, array shape: {temporal_means_array.shape}"
        )
        
        return ModelResults(
            temporal_means=temporal_means_array,
            channel_cfs=cfs,
            frequencies=frequencies.tolist(),
            db_levels=[60.0],  # Only using 60 dB
            model_name='BEZ2018',
            metadata={
                'source': 'bez_acrossruns_psths.mat',
                'run_averaged': True,
                'fiber_types': fiber_types,
                'fiber_type_used': fiber_type_to_use,
                'has_population': compute_population,
                'population_weights': self.population_analyzer.get_weights() if compute_population else None,
                'raw_temporal_means': temporal_means,  # Keep original nested structure
                'db_filtered': 60.0
            }
        )
    
    def _dict_to_array(self, fiber_dict, cfs, frequencies, db_levels):
        """
        Convert nested dictionary to numpy array.
        
        Args:
            fiber_dict: Nested dict [cf][freq][db] = value
            cfs: Array of CFs
            frequencies: Array of frequencies  
            db_levels: Array of dB levels
            
        Returns:
            Array of shape (n_cfs, n_conditions) where n_conditions = n_freqs * n_dbs
        """
        n_cfs = len(cfs)
        n_freqs = len(frequencies)
        n_dbs = len(db_levels)
        n_conditions = n_freqs * n_dbs
        
        array = np.zeros((n_cfs, n_conditions))
        
        for cf_idx, cf in enumerate(cfs):
            cond_idx = 0
            for freq in frequencies:
                for db in db_levels:
                    array[cf_idx, cond_idx] = fiber_dict[cf][freq][db]
                    cond_idx += 1
        
        return array


class CochleaResultsLoader:
    """
    Loads Cochlea model results from MATLAB files.
    
    Wraps MatlabDataLoader to provide ModelResults output compatible
    with comparison pipeline. Supports population weighting.
    """
    
    def __init__(self, cochlea_dir: Path = None):
        """
        Initialize Cochlea results loader.
        
        Args:
            cochlea_dir: Directory containing Cochlea PSTH data files
        """
        self.matlab_loader = MatlabDataLoader(cochlea_dir=cochlea_dir)
        self.population_analyzer = None
        self._loaded = False
    
    def load(
        self,
        fiber_types: list = None,
        compute_population: bool = False,
        population_weights: dict = None
    ) -> ModelResults:
        """
        Load Cochlea results with optional population weighting.
        
        Args:
            fiber_types: List of fiber types to load (default: ['hsr', 'msr', 'lsr'])
            compute_population: If True, compute weighted population average
            population_weights: Custom weights for population averaging
            
        Returns:
            ModelResults containing Cochlea temporal means and metadata
        """
        if fiber_types is None:
            fiber_types = ['hsr', 'msr', 'lsr']
        
        # Load using MatlabDataLoader
        self.matlab_loader.load_cochlea(fiber_types=fiber_types)
        self._loaded = True
        
        # Get data
        temporal_means = self.matlab_loader.get_temporal_means()
        cfs = self.matlab_loader.get_cfs()
        frequencies = self.matlab_loader.get_frequencies()
        db_levels = self.matlab_loader.get_db_levels()
        
        # Compute population if requested
        if compute_population:
            self.population_analyzer = PopulationAnalyzer(
                weights=population_weights,
                normalize=True
            )
            population_means = self.population_analyzer.compute_population_from_temporal_means(
                temporal_means
            )
            # Add population to temporal_means
            temporal_means['population'] = population_means
            
            logger.info(
                f"Added weighted population with weights: "
                f"{self.population_analyzer.get_weights()}"
            )
            
            # Use population for comparison
            fiber_type_to_use = 'population'
        else:
            # Use first fiber type if no population
            fiber_type_to_use = fiber_types[0]
            logger.info(f"Using {fiber_type_to_use} fiber type for comparison")
        
        # Convert nested dict to array for the selected fiber type
        # Structure: [fiber_type][cf][freq][db] -> array[cf, condition]
        # Filter to only 60 dB to match WSR
        db_filter = [60.0]
        temporal_means_array = self._dict_to_array(
            temporal_means[fiber_type_to_use],
            cfs, frequencies, db_filter
        )
        
        logger.info(
            f"Loaded Cochlea results: {len(cfs)} CFs, {len(frequencies)} freqs, "
            f"filtered to 60 dB only, array shape: {temporal_means_array.shape}"
        )
        
        return ModelResults(
            temporal_means=temporal_means_array,
            channel_cfs=cfs,
            frequencies=frequencies.tolist(),
            db_levels=[60.0],  # Only using 60 dB
            model_name='Cochlea',
            metadata={
                'source': 'cochlea_psths.mat',
                'fiber_types': fiber_types,
                'fiber_type_used': fiber_type_to_use,
                'has_population': compute_population,
                'population_weights': self.population_analyzer.get_weights() if compute_population else None,
                'raw_temporal_means': temporal_means,  # Keep original nested structure
                'db_filtered': 60.0
            }
        )
    
    def _dict_to_array(self, fiber_dict, cfs, frequencies, db_levels):
        """
        Convert nested dictionary to numpy array.
        
        Args:
            fiber_dict: Nested dict [cf][freq][db] = value
            cfs: Array of CFs
            frequencies: Array of frequencies  
            db_levels: Array of dB levels
            
        Returns:
            Array of shape (n_cfs, n_conditions) where n_conditions = n_freqs * n_dbs
        """
        n_cfs = len(cfs)
        n_freqs = len(frequencies)
        n_dbs = len(db_levels)
        n_conditions = n_freqs * n_dbs
        
        array = np.zeros((n_cfs, n_conditions))
        
        for cf_idx, cf in enumerate(cfs):
            cond_idx = 0
            for freq in frequencies:
                for db in db_levels:
                    array[cf_idx, cond_idx] = fiber_dict[cf][freq][db]
                    cond_idx += 1
        
        return array


class CFAligner:
    """
    Aligns model outputs by matching characteristic frequencies.
    
    Uses CFMatcher from WSRmodel to map WSR channels to BEZ CFs.
    """
    
    def __init__(self, cf_matcher: CFMatcher):
        """
        Initialize CF aligner.
        
        Args:
            cf_matcher: CFMatcher instance for aligning CFs
        """
        self.matcher = cf_matcher
        logger.info(
            f"CFAligner initialized with {len(self.matcher.target_cfs)} target CFs"
        )
    
    def align_results(
        self,
        wsr_results: ModelResults,
        bez_results: ModelResults
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align WSR and BEZ results by matching CFs.
        
        Extracts WSR channels that match to BEZ CFs using the CFMatcher.
        BEZ results are assumed to already be in the correct CF order.
        
        Args:
            wsr_results: WSR model results
            bez_results: BEZ model results
            
        Returns:
            Tuple of (wsr_aligned, bez_aligned) with matching CF dimensions
        """
        # Get WSR channel indices that match to BEZ CFs
        wsr_indices = self.matcher.get_wsr_channel_indices()
        
        # Extract matched WSR channels
        wsr_temporal_means = wsr_results.temporal_means
        
        if wsr_temporal_means.ndim == 1:
            # Single condition: (n_channels,)
            wsr_aligned = wsr_temporal_means[wsr_indices]
        else:
            # Multiple dimensions: extract along first axis (channels)
            # Assumes channels are first dimension
            wsr_aligned = wsr_temporal_means[wsr_indices, ...]
        
        # Find which BEZ CFs match to WSR
        # matcher.target_cfs are the BEZ CFs that were matched
        # bez_results.channel_cfs are all BEZ CFs
        # We need to find indices of target_cfs in channel_cfs
        bez_cf_indices = []
        target_cfs = self.matcher.target_cfs
        
        for target_cf in target_cfs:
            # Find index in bez_results.channel_cfs
            idx = np.argmin(np.abs(bez_results.channel_cfs - target_cf))
            # Verify it's a close match (within 1 Hz)
            if np.abs(bez_results.channel_cfs[idx] - target_cf) < 1.0:
                bez_cf_indices.append(idx)
        
        bez_cf_indices = np.array(bez_cf_indices)
        
        # Extract matched BEZ CFs
        bez_temporal_means = bez_results.temporal_means
        
        if bez_temporal_means.ndim == 1:
            bez_aligned = bez_temporal_means[bez_cf_indices]
        else:
            bez_aligned = bez_temporal_means[bez_cf_indices, ...]
        
        logger.info(
            f"Aligned results: WSR shape {wsr_aligned.shape}, "
            f"BEZ shape {bez_aligned.shape}"
        )
        
        # Validate alignment
        if wsr_aligned.shape != bez_aligned.shape:
            logger.warning(
                f"Shape mismatch after alignment: WSR {wsr_aligned.shape} "
                f"vs BEZ {bez_aligned.shape}"
            )
        
        return wsr_aligned, bez_aligned
    
    def get_matched_cfs(self) -> np.ndarray:
        """Get the matched characteristic frequencies."""
        return self.matcher.target_cfs
    
    def get_wsr_indices(self) -> np.ndarray:
        """Get WSR channel indices used in alignment."""
        return self.matcher.get_wsr_channel_indices()
    
    def get_mapping_table(self) -> list:
        """Get complete CF mapping table."""
        return self.matcher.get_mapping_table()

