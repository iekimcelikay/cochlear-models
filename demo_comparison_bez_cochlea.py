"""
Demo script for BEZ vs Cochlea population comparison.

Compares BEZ model against Cochlea model weighted population averages using default physiological ratios.
"""

import logging
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from compare_bez_cochlea_mainscript import (
    MatlabDataLoader,
    PopulationAnalyzer,
    average_across_runs
)

# Configure logging to both file and console
def setup_logging(output_dir):
    """Configure logging to write to both file and console."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'comparison_log.txt'
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    
    return log_file

logger = logging.getLogger(__name__)


def main():
    """Run BEZ vs Cochlea population comparison."""
    
    # Base directory
    base_dir = Path(__file__).parent
    
    # Output directory
    output_dir = base_dir / 'model_comparisons/results/bez_vs_cochlea_population'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file and console
    log_file = setup_logging(output_dir)
    
    # Fiber types to analyze
    fiber_types = ['hsr', 'msr', 'lsr']
    
    # Initialize loaders
    logger.info("=" * 80)
    logger.info("BEZ vs Cochlea Population Comparison")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")
    
    # Load BEZ data with population
    logger.info("\n1. Loading BEZ model data...")
    cochlea_dir = base_dir / 'cochlea_meanrate/out/condition_psths'
    bez_loader = MatlabDataLoader(cochlea_dir=cochlea_dir)
    bez_loader.load_bez_run_averaged(fiber_types=fiber_types)
    
    # Compute BEZ population
    logger.info("   Computing BEZ weighted population...")
    bez_population_analyzer = PopulationAnalyzer(normalize=True)
    bez_temporal_means = bez_loader.get_temporal_means()
    bez_population = bez_population_analyzer.compute_population_from_temporal_means(
        bez_temporal_means
    )
    logger.info(f"   Weights: {bez_population_analyzer.get_weights()}")
    
    # Load Cochlea data with population
    logger.info("\n2. Loading Cochlea model data...")
    cochlea_loader = MatlabDataLoader(cochlea_dir=cochlea_dir)
    cochlea_loader.load_cochlea(fiber_types=fiber_types)
    
    # Compute Cochlea population
    logger.info("   Computing Cochlea weighted population...")
    cochlea_population_analyzer = PopulationAnalyzer(normalize=True)
    cochlea_temporal_means = cochlea_loader.get_temporal_means()
    cochlea_population = cochlea_population_analyzer.compute_population_from_temporal_means(
        cochlea_temporal_means
    )
    logger.info(f"   Weights: {cochlea_population_analyzer.get_weights()}")
    
    # Get parameters
    bez_cfs = bez_loader.get_cfs()
    bez_freqs = bez_loader.get_frequencies()
    bez_dbs = bez_loader.get_db_levels()
    
    cochlea_cfs = cochlea_loader.get_cfs()
    cochlea_freqs = cochlea_loader.get_frequencies()
    cochlea_dbs = cochlea_loader.get_db_levels()
    
    logger.info(f"\n3. Data dimensions:")
    logger.info(f"   BEZ: CFs={len(bez_cfs)}, Freqs={len(bez_freqs)}, dBs={len(bez_dbs)}")
    logger.info(f"   Cochlea: CFs={len(cochlea_cfs)}, Freqs={len(cochlea_freqs)}, dBs={len(cochlea_dbs)}")
    
    # Convert to arrays (filter to 60 dB only)
    logger.info("\n4. Converting to arrays (60 dB only)...")
    db_filter = [60.0]
    
    def dict_to_array(fiber_dict, cfs, frequencies, db_levels):
        """Convert nested dict to 2D array."""
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
    
    bez_array = dict_to_array(bez_population, bez_cfs, bez_freqs, db_filter)
    cochlea_array = dict_to_array(cochlea_population, cochlea_cfs, cochlea_freqs, db_filter)
    
    logger.info(f"   BEZ array shape: {bez_array.shape}")
    logger.info(f"   Cochlea array shape: {cochlea_array.shape}")
    logger.info(f"   Expected: ({len(bez_cfs)}, {len(bez_freqs) * len(db_filter)})")
    
    # Align CFs (should be identical, but use tolerance for floating-point comparison)
    logger.info("\n5. Aligning characteristic frequencies...")
    
    # Check if CFs are identical
    logger.info(f"   BEZ CFs: {bez_cfs[:5]}... (showing first 5)")
    logger.info(f"   Cochlea CFs: {cochlea_cfs[:5]}... (showing first 5)")
    
    # Find common CFs with tolerance for floating-point comparison
    tolerance = 0.01  # 0.01 Hz tolerance
    common_cfs = []
    bez_cf_indices = []
    cochlea_cf_indices = []
    
    for i, bez_cf in enumerate(bez_cfs):
        for j, cochlea_cf in enumerate(cochlea_cfs):
            if abs(bez_cf - cochlea_cf) < tolerance:
                common_cfs.append(bez_cf)
                bez_cf_indices.append(i)
                cochlea_cf_indices.append(j)
                break
    
    common_cfs = np.array(common_cfs)
    logger.info(f"   Common CFs: {len(common_cfs)} of {len(bez_cfs)} BEZ CFs")
    
    # Extract aligned data
    bez_aligned = bez_array[bez_cf_indices, :]
    cochlea_aligned = cochlea_array[cochlea_cf_indices, :]
    
    logger.info(f"   Aligned shapes: BEZ {bez_aligned.shape}, Cochlea {cochlea_aligned.shape}")
    logger.info(f"   This represents {bez_aligned.shape[0]} CFs × {bez_aligned.shape[1]} frequencies = {bez_aligned.shape[0] * bez_aligned.shape[1]} total points")
    
    # Normalize to [0, 1]
    logger.info("\n6. Normalizing data to [0, 1] range...")
    
    def normalize_to_unit_range(data):
        """Min-max normalization to [0, 1]."""
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min < 1e-10:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    
    bez_norm = normalize_to_unit_range(bez_aligned)
    cochlea_norm = normalize_to_unit_range(cochlea_aligned)
    
    logger.info(f"   BEZ range: [{bez_norm.min():.3f}, {bez_norm.max():.3f}]")
    logger.info(f"   Cochlea range: [{cochlea_norm.min():.3f}, {cochlea_norm.max():.3f}]")
    
    # Perform regression
    logger.info("\n7. Performing linear regression...")
    from scipy.stats import linregress
    
    bez_flat = bez_norm.flatten()
    cochlea_flat = cochlea_norm.flatten()
    
    slope, intercept, r_value, p_value, std_err = linregress(bez_flat, cochlea_flat)
    r_squared = r_value ** 2
    
    logger.info(f"   R² = {r_squared:.4f}")
    logger.info(f"   Slope = {slope:.4f}")
    logger.info(f"   Intercept = {intercept:.4f}")
    logger.info(f"   p-value = {p_value:.2e}")
    
    # Create comparison plot
    logger.info("\n8. Generating comparison plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by CF
    cf_colors = plt.cm.Spectral(np.linspace(0, 1, len(common_cfs)))
    
    # Flatten data for all CFs and frequencies
    for cf_idx, cf in enumerate(common_cfs):
        # Extract data for this CF (all frequencies)
        bez_cf_data = bez_norm[cf_idx, :]
        cochlea_cf_data = cochlea_norm[cf_idx, :]
        
        # Plot each frequency point with alpha based on |freq - CF|
        # When freq == CF: alpha = 1.0 (brightest)
        # As |freq - CF| increases: alpha decreases (dimmer)
        for freq_idx, freq in enumerate(bez_freqs):
            freq_diff_normalized = abs(freq - cf) / cf
            # Invert: 0 difference = alpha 1.0, large difference = alpha 0.2
            alpha = 1.0 - 0.8 * min(freq_diff_normalized, 1.0)
            
            ax.scatter(
                bez_cf_data[freq_idx],
                cochlea_cf_data[freq_idx],
                color=cf_colors[cf_idx],
                alpha=alpha,
                s=50,
                edgecolors='none'
            )
    
    logger.info(f"   Plotted {len(common_cfs) * len(bez_freqs)} data points")
    
    # Add unity line and regression line
    unity_line = ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Unity')[0]
    
    x_line = np.array([0, 1])
    y_line = slope * x_line + intercept
    fit_line = ax.plot(x_line, y_line, 'r-', linewidth=2, 
                       label=f'Fit: y={slope:.3f}x+{intercept:.3f}\nR²={r_squared:.4f}, p={p_value:.2e}')[0]
    
    # Create CF legend with all CFs
    cf_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=cf_colors[i], markersize=8, 
                             label=f'{cf:.1f} Hz')
                  for i, cf in enumerate(common_cfs)]
    
    legend_cf = ax.legend(handles=cf_handles, title='CF', loc='upper left', 
                          bbox_to_anchor=(1.02, 1), fontsize=8, ncol=1)
    ax.add_artist(legend_cf)
    
    # Add fit legend (use separate handles to avoid overwriting CF legend)
    legend_fit = ax.legend(handles=[unity_line, fit_line], loc='lower right', fontsize=10)
    ax.add_artist(legend_fit)
    
    ax.set_xlabel('BEZ2018 Population (normalized)', fontsize=12)
    ax.set_ylabel('Cochlea Population (normalized)', fontsize=12)
    ax.set_title('BEZ vs Cochlea Population Comparison (60 dB)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Adjust layout to make room for legends
    plt.subplots_adjust(right=0.75)
    
    # Save plot - bbox_extra_artists ensures legends are included
    plot_file = output_dir / 'bez_vs_cochlea_comparison.png'
    fig.savefig(plot_file, dpi=300, bbox_inches='tight', 
                bbox_extra_artists=[legend_cf, legend_fit])
    logger.info(f"   Plot saved: {plot_file}")
    
    # Save results
    logger.info("\n9. Saving results...")
    
    # Save aligned data
    data_file = output_dir / 'bez_cochlea_aligned_data.npz'
    np.savez(
        data_file,
        bez_aligned=bez_aligned,
        cochlea_aligned=cochlea_aligned,
        bez_normalized=bez_norm,
        cochlea_normalized=cochlea_norm,
        common_cfs=common_cfs,
        frequencies=bez_freqs,
        db_level=60.0
    )
    logger.info(f"   Data saved: {data_file}")
    
    # Save regression results
    reg_file = output_dir / 'bez_cochlea_regression.npz'
    np.savez(
        reg_file,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        p_value=p_value,
        std_err=std_err
    )
    logger.info(f"   Regression saved: {reg_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"R² = {r_squared:.4f}, p-value = {p_value:.2e}")
    
    plt.show()


if __name__ == '__main__':
    main()
