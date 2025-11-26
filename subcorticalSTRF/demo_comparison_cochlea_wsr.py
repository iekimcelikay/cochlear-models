"""
Demo script for WSR vs Cochlea population comparison.

Compares WSR model against Cochlea model weighted population average
using default physiological ratios.
"""

import logging
from pathlib import Path
from model_comparisons.classes.comparison_pipeline import ComparisonPipeline
from loggers import FolderManager, LoggingConfigurator, MetadataSaver

logger = logging.getLogger(__name__)


def main():
    """Run WSR vs Cochlea population comparison."""

    # Base directory
    base_dir = Path(__file__).parent

    # WSR results from run_wsr_model_OOP.py
    wsr_results_path = base_dir / 'WSRmodel/results/'
    'wsr_model_psth_batch_92chans_8ms_8tc_-1factor_0shift_20251124_154938/'
    'wsr_results.pkl'

    # Cochlea data directory
    cochlea_dir = base_dir / 'cochlea_meanrate/out/condition_psths'

    # Create organized output directory using FolderManager
    folder_manager = FolderManager(
        base_dir=base_dir / 'model_comparisons/results',
        name_builder='model_comparison'
    ).with_params(
        model1='wsr',
        model2='cochlea',
        comparison_type='population',
        wsr_results_path=str(wsr_results_path),
        min_cf=180.0,
        fs=16000
    )

    output_dir = Path(folder_manager.create_folder())

    # Setup logging to file and console using LoggingConfigurator
    log_file = LoggingConfigurator.setup_basic(output_dir,
                                               'comparison_log.txt')
    logger.info(f"Log file: {log_file}")

    # Create and run pipeline with population analysis for Cochlea model
    pipeline = ComparisonPipeline(
        wsr_results_path=wsr_results_path,
        model_dir=cochlea_dir,  # Specify Cochlea data directory
        output_dir=output_dir,
        min_cf=180.0,
        fs=16000,
        compute_population=True,  # Uses default weights:
        # HSR 65%, MSR 23%, LSR 12%
        model_type='cochlea'  # Use Cochlea model instead of BEZ
    )

    results = pipeline.run(
        fiber_types=['hsr', 'msr', 'lsr'],
        bez_freq_range=(125, 2500, 20),
        wsr_params=[8, 8, -1, 0]
    )

    # Save and plot
    pipeline.save_results(results)  # Uses default prefix "wsr_vs_cochlea"
    pipeline.plot_results(results, show_plots=True)

    # Save summary statistics using MetadataSaver
    metadata_saver = MetadataSaver()
    summary_stats = {
        'r_squared': float(results.regression_results.r_squared),
        'p_value': float(results.regression_results.p_value),
        'slope': float(results.regression_results.slope),
        'intercept': float(results.regression_results.intercept),
        'n_data_points': len(results.regression_results.x_data),
        'model_comparison': 'wsr vs cochlea',
        'fiber_types': results.metadata['fiber_types'],
        'population_weights': results.metadata.get('population_weights')
    }
    metadata_saver.save_json(output_dir, summary_stats,
                             filename="summary_statistics.json")

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"R² = {results.regression_results.r_squared:.4f}")
    logger.info(f"p-value = {results.regression_results.p_value:.2e}")


if __name__ == '__main__':
    main()
