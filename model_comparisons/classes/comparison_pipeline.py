"""
Comparison pipeline for WSR vs BEZ model analysis.

This module orchestrates the complete workflow:
1. Load WSR results from saved simulation
2. Load BEZ results from MATLAB files
3. Optionally compute weighted population
4. Align CFs between models
5. Perform regression analysis
6. Generate plots and save results
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple
import pickle

from ..core import ModelResults, ComparisonResults, RegressionResults
from ..components import BEZResultsLoader, CochleaResultsLoader, CFAligner
from ..regression_analyzer import RegressionAnalyzer
from WSRmodel.cf_matcher import create_cf_matcher_from_bez2018
from WSRmodel.wsrmodel import WSRModel

logger = logging.getLogger(__name__)


class WSRResultsLoader:
    """
    Loads WSR model results from saved simulation files.
    
    Works with results saved by run_wsr_model_OOP.py or similar scripts.
    """
    
    def __init__(self, results_path: Path):
        """
        Initialize WSR results loader.
        
        Args:
            results_path: Path to saved WSR results (.pkl or .npz)
        """
        self.results_path = Path(results_path)
        
        if not self.results_path.exists():
            raise FileNotFoundError(f"WSR results not found: {self.results_path}")
    
    def load(self) -> ModelResults:
        """
        Load WSR results from file.
        
        Returns:
            ModelResults containing WSR temporal means and metadata
        """
        suffix = self.results_path.suffix.lower()
        
        if suffix == '.pkl':
            return self._load_from_pickle()
        elif suffix == '.npz':
            return self._load_from_npz()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _load_from_pickle(self) -> ModelResults:
        """Load from pickle file."""
        with open(self.results_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded WSR results from {self.results_path}")
        
        # Extract temporal means for each condition
        # Assuming data is dict with keys like 'freq_500.0hz_db_60'
        temporal_means_dict = {}
        frequencies = set()
        db_levels = set()
        
        for key, result in data.items():
            if key.startswith('freq_'):
                # Extract frequency and dB from key
                freq = result.get('freq')
                db = result.get('db')
                temporal_mean = result.get('temporal_mean')
                
                if temporal_mean is not None:
                    temporal_means_dict[key] = temporal_mean
                    frequencies.add(freq)
                    db_levels.add(db)
        
        # Get channel CFs (assuming same for all conditions)
        first_result = next(iter(data.values()))
        channel_cfs = first_result.get('channel_cfs', np.array([]))
        
        # Stack temporal means into array
        # Shape: (n_channels, n_conditions)
        temporal_means_list = [temporal_means_dict[k] for k in sorted(temporal_means_dict.keys())]
        temporal_means = np.column_stack(temporal_means_list) if temporal_means_list else np.array([])
        
        return ModelResults(
            temporal_means=temporal_means,
            channel_cfs=channel_cfs,
            frequencies=sorted(frequencies),
            db_levels=sorted(db_levels),
            model_name='WSR',
            metadata={
                'source': str(self.results_path),
                'n_conditions': len(temporal_means_dict)
            }
        )
    
    def _load_from_npz(self) -> ModelResults:
        """Load from numpy .npz file."""
        data = np.load(self.results_path, allow_pickle=True)
        
        logger.info(f"Loaded WSR results from {self.results_path}")
        
        return ModelResults(
            temporal_means=data['temporal_means'],
            channel_cfs=data['channel_cfs'],
            frequencies=data['frequencies'].tolist(),
            db_levels=data['db_levels'].tolist(),
            model_name='WSR',
            metadata=data.get('metadata', {}).item() if 'metadata' in data else {}
        )


class ComparisonPipeline:
    """
    Orchestrates WSR vs auditory model comparison workflow.
    
    Combines loading, alignment, regression, and visualization into a
    single coherent pipeline with optional population analysis.
    
    Supports comparison with BEZ2018 or Cochlea models.
    """
    
    def __init__(
        self,
        wsr_results_path: Path,
        model_dir: Optional[Path] = None,
        model_type: str = 'bez',
        output_dir: Optional[Path] = None,
        min_cf: float = 180.0,
        fs: int = 16000,
        compute_population: bool = False,
        population_weights: Optional[dict] = None
    ):
        """
        Initialize comparison pipeline.
        
        Args:
            wsr_results_path: Path to WSR results file
            model_dir: Directory containing model results (uses default if None)
            model_type: Type of model to compare ('bez' or 'cochlea')
            output_dir: Directory for saving comparison results
            min_cf: Minimum CF to include (Hz)
            fs: Sampling frequency for WSR model (Hz)
            compute_population: If True, compute weighted population average
            population_weights: Custom weights for population (default: physiological ratios)
        """
        self.wsr_results_path = Path(wsr_results_path)
        self.model_type = model_type.lower()
        self.output_dir = Path(output_dir) if output_dir else self.wsr_results_path.parent
        self.min_cf = min_cf
        self.fs = fs
        self.compute_population = compute_population
        self.population_weights = population_weights
        
        # Validate model type
        if self.model_type not in ['bez', 'cochlea']:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'bez' or 'cochlea'")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.wsr_loader = WSRResultsLoader(self.wsr_results_path)
        
        # Select appropriate model loader based on model_type
        if self.model_type == 'bez':
            self.model_loader = BEZResultsLoader(bez_dir=model_dir)
            self.model_name = 'BEZ2018'
        elif self.model_type == 'cochlea':
            self.model_loader = CochleaResultsLoader(cochlea_dir=model_dir)
            self.model_name = 'Cochlea'
        
        self.regression_analyzer = RegressionAnalyzer()
        
        logger.info(f"ComparisonPipeline initialized")
        logger.info(f"  Model type: {self.model_name}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Population analysis: {self.compute_population}")
    
    def run(
        self,
        fiber_types: Optional[list] = None,
        bez_freq_range: Tuple[float, float, int] = (125, 2500, 20),
        wsr_params: list = [8, 8, -1, 0]
    ) -> ComparisonResults:
        """
        Run complete comparison pipeline.
        
        Args:
            fiber_types: BEZ fiber types to load (default: ['hsr', 'msr', 'lsr'])
            bez_freq_range: BEZ CF range (min_cf, max_cf, num_cf)
            wsr_params: WSR model parameters for CF matching
            
        Returns:
            ComparisonResults with all analysis outputs
        """
        logger.info("=" * 70)
        logger.info(f"Starting WSR vs {self.model_name} comparison pipeline")
        logger.info("=" * 70)
        
        # 1. Load WSR results
        logger.info("Step 1: Loading WSR results...")
        wsr_results = self.wsr_loader.load()
        logger.info(f"  ✓ WSR: {wsr_results.temporal_means.shape}, CFs: {len(wsr_results.channel_cfs)}")
        
        # 2. Load model results (with optional population)
        logger.info(f"Step 2: Loading {self.model_name} results...")
        model_results = self.model_loader.load(
            fiber_types=fiber_types,
            compute_population=self.compute_population,
            population_weights=self.population_weights
        )
        logger.info(f"  ✓ {self.model_name}: CFs: {len(model_results.channel_cfs)}")
        if self.compute_population:
            logger.info(f"  ✓ Population computed with weights: {model_results.metadata['population_weights']}")
        
        # 3. Create CF matcher and aligner
        logger.info("Step 3: Creating CF matcher...")
        wsr_model = WSRModel(wsr_params)
        cf_matcher = create_cf_matcher_from_bez2018(
            wsr_model,
            freq_range=bez_freq_range,
            fs=self.fs,
            min_cf=self.min_cf
        )
        aligner = CFAligner(cf_matcher)
        logger.info(f"  ✓ Matched {len(cf_matcher.target_cfs)} CFs")
        
        # 4. Align results
        logger.info("Step 4: Aligning WSR and model data by CFs...")
        wsr_aligned, model_aligned = aligner.align_results(wsr_results, model_results)
        logger.info(f"  ✓ Aligned shapes: WSR {wsr_aligned.shape}, {self.model_name} {model_aligned.shape}")
        
        # 5. Perform regression
        logger.info("Step 5: Performing regression analysis...")
        regression_results = self.regression_analyzer.fit(
            wsr_aligned,
            model_aligned,
            label_x="WSR",
            label_y=self.model_name,
            normalize=True
        )
        logger.info(
            f"  ✓ R² = {regression_results.r_squared:.4f}, "
            f"p = {regression_results.p_value:.2e}"
        )
        
        # 6. Create comparison results
        comparison_results = ComparisonResults(
            regression_results=regression_results,
            wsr_results=wsr_results,
            bez_results=model_results,
            aligned_data=(wsr_aligned, model_aligned),
            cf_mapping=aligner.get_mapping_table(),
            metadata={
                'model_type': self.model_type,
                'model_name': self.model_name,
                'wsr_params': wsr_params,
                'bez_freq_range': bez_freq_range,
                'fiber_types': fiber_types or ['hsr', 'msr', 'lsr'],
                'min_cf': self.min_cf,
                'fs': self.fs,
                'compute_population': self.compute_population,
                'population_weights': model_results.metadata.get('population_weights')
            }
        )
        
        logger.info("=" * 70)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 70)
        
        return comparison_results
    
    def save_results(
        self,
        comparison_results: ComparisonResults,
        prefix: str = None
    ):
        """
        Save comparison results to output directory.
        
        Args:
            comparison_results: ComparisonResults from run()
            prefix: Filename prefix for saved files (default: "wsr_vs_{model_name}")
        """
        if prefix is None:
            prefix = f"wsr_vs_{self.model_name.lower()}"
            
        logger.info("Saving comparison results...")
        
        # Save CF mapping table
        cf_mapping_file = self.output_dir / f"{prefix}_cf_mapping.csv"
        import csv
        with open(cf_mapping_file, 'w', newline='') as f:
            if comparison_results.cf_mapping:
                writer = csv.DictWriter(f, fieldnames=comparison_results.cf_mapping[0].keys())
                writer.writeheader()
                writer.writerows(comparison_results.cf_mapping)
        logger.info(f"  ✓ CF mapping: {cf_mapping_file}")
        
        # Save regression results
        regression_file = self.output_dir / f"{prefix}_regression.npz"
        np.savez(
            regression_file,
            slope=comparison_results.regression_results.slope,
            intercept=comparison_results.regression_results.intercept,
            r_squared=comparison_results.regression_results.r_squared,
            p_value=comparison_results.regression_results.p_value,
            x_data=comparison_results.regression_results.x_data,
            y_data=comparison_results.regression_results.y_data,
            predictions=comparison_results.regression_results.predictions,
            residuals=comparison_results.regression_results.residuals,
            metadata=comparison_results.regression_results.metadata
        )
        logger.info(f"  ✓ Regression data: {regression_file}")
        
        # Save aligned data
        aligned_file = self.output_dir / f"{prefix}_aligned_data.npz"
        wsr_aligned, model_aligned = comparison_results.aligned_data
        np.savez(
            aligned_file,
            wsr_aligned=wsr_aligned,
            model_aligned=model_aligned,
            model_name=comparison_results.metadata['model_name'],
            metadata=comparison_results.metadata
        )
        logger.info(f"  ✓ Aligned data: {aligned_file}")
        
        # Save complete results as pickle
        results_file = self.output_dir / f"{prefix}_complete_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(comparison_results, f)
        logger.info(f"  ✓ Complete results: {results_file}")
    
    def plot_results(
        self,
        comparison_results: ComparisonResults,
        prefix: str = None,
        show_plots: bool = False
    ):
        """
        Generate and save comparison plots.
        
        Args:
            comparison_results: ComparisonResults from run()
            prefix: Filename prefix for plot files (default: "wsr_vs_{model_name}")
            show_plots: If True, display plots interactively
        """
        import matplotlib.pyplot as plt
        
        if prefix is None:
            prefix = f"wsr_vs_{self.model_name.lower()}"
        
        logger.info("Generating comparison plots...")
        
        # Extract CF and frequency arrays for color coding
        cf_list, freq_list = self._extract_cf_freq_arrays(comparison_results)
        
        # Basic comparison plot with CF-brightness encoding (single plot)
        basic_plot_file = self.output_dir / f"{prefix}_comparison.png"
        fig1 = self.regression_analyzer.plot_comparison(
            comparison_results.regression_results,
            save_path=basic_plot_file,
            show_unity_line=True,
            cf_list=cf_list,
            freq_list=freq_list,
            figsize=(10, 8)  # Larger single plot
        )
        logger.info(f"  ✓ Comparison plot: {basic_plot_file}")
        
        if show_plots:
            plt.show()
        else:
            plt.close('all')
    
    def _extract_cf_freq_arrays(self, comparison_results: ComparisonResults):
        """
        Extract CF and frequency arrays for each data point.
        
        Returns:
            Tuple of (cf_array, freq_array) where each element corresponds to a data point
        """
        # Get matched CFs from the CF mapping table
        cf_mapping = comparison_results.cf_mapping
        matched_cfs = np.array([row['target_cf'] for row in cf_mapping])
        
        # Get frequencies
        frequencies = np.array(comparison_results.bez_results.frequencies)
        
        # Build arrays: each (CF, freq) combination becomes one point
        cf_list = []
        freq_list = []
        
        for cf in matched_cfs:
            for freq in frequencies:
                cf_list.append(cf)
                freq_list.append(freq)
        
        return np.array(cf_list), np.array(freq_list)
