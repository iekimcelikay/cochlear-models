"""
Regression analysis for model comparison.

This module contains:
- RegressionAnalyzer: Performs linear regression between model outputs
- RegressionResults dataclass: Stores regression statistics
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional
from scipy import stats
import matplotlib.pyplot as plt

from .core import RegressionResults

logger = logging.getLogger(__name__)


class RegressionAnalyzer:
    """
    Performs linear regression analysis between model outputs.
    
    Fits linear models, computes statistics, and generates diagnostic plots.
    """
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize regression analyzer.
        
        Args:
            logger_instance: Optional logger instance (uses module logger if None)
        """
        self.logger = logger_instance or logger
    
    def normalize_to_unit_range(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to [0, 1] range using min-max normalization.
        
        Args:
            data: Input array
            
        Returns:
            Normalized array in [0, 1] range
        """
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        
        if data_max - data_min == 0:
            return np.zeros_like(data)
        
        return (data - data_min) / (data_max - data_min)
    
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        label_x: str = "WSR",
        label_y: str = "BEZ",
        normalize: bool = True
    ) -> RegressionResults:
        """
        Fit linear regression model: y = slope * x + intercept
        
        Args:
            x: Input data (e.g., WSR temporal means)
            y: Target data (e.g., BEZ temporal means)
            label_x: Label for x data (for plotting/reporting)
            label_y: Label for y data (for plotting/reporting)
            normalize: If True, normalize both x and y to [0, 1] range before regression
            
        Returns:
            RegressionResults with slope, intercept, R², p-value, etc.
        """
        # Flatten arrays for regression
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        # Normalize if requested
        if normalize:
            x_flat = self.normalize_to_unit_range(x_flat)
            y_flat = self.normalize_to_unit_range(y_flat)
            self.logger.info("Data normalized to [0, 1] range before regression")
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x_flat) | np.isnan(y_flat))
        x_clean = x_flat[valid_mask]
        y_clean = y_flat[valid_mask]
        
        if len(x_clean) < 2:
            self.logger.warning(
                f"Insufficient data for regression: only {len(x_clean)} valid points"
            )
            return self._create_nan_results(x_flat, y_flat, label_x, label_y)
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        
        # Calculate predictions and residuals
        predictions = slope * x_clean + intercept
        residuals = y_clean - predictions
        
        r_squared = r_value ** 2
        
        self.logger.info(
            f"Regression: {label_y} = {slope:.4f} * {label_x} + {intercept:.4f}, "
            f"R² = {r_squared:.4f}, p = {p_value:.2e}, n = {len(x_clean)}"
        )
        
        return RegressionResults(
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            p_value=p_value,
            predictions=predictions,
            residuals=residuals,
            x_data=x_clean,
            y_data=y_clean,
            metadata={
                'label_x': label_x,
                'label_y': label_y,
                'std_err': std_err,
                'n_points': len(x_clean),
                'n_removed': len(x_flat) - len(x_clean)
            }
        )
    
    def _create_nan_results(
        self, 
        x_flat: np.ndarray, 
        y_flat: np.ndarray,
        label_x: str,
        label_y: str
    ) -> RegressionResults:
        """Create RegressionResults with NaN values for invalid data."""
        return RegressionResults(
            slope=np.nan,
            intercept=np.nan,
            r_squared=np.nan,
            p_value=np.nan,
            predictions=np.array([]),
            residuals=np.array([]),
            x_data=x_flat,
            y_data=y_flat,
            metadata={
                'label_x': label_x,
                'label_y': label_y,
                'std_err': np.nan,
                'n_points': 0,
                'n_removed': len(x_flat)
            }
        )
    
    def plot_comparison(
        self,
        results: RegressionResults,
        save_path: Optional[Path] = None,
        show_unity_line: bool = True,
        figsize: tuple = (10, 8),
        cf_list: Optional[np.ndarray] = None,
        freq_list: Optional[np.ndarray] = None
    ) -> plt.Figure:
        """
        Plot regression results with scatter plot.
        
        Args:
            results: RegressionResults from fit()
            save_path: Optional path to save figure
            show_unity_line: If True, show y=x line for reference
            figsize: Figure size (width, height)
            cf_list: Optional array of CFs for each data point (for color coding)
            freq_list: Optional array of frequencies for each data point (for brightness)
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        label_x = results.metadata.get('label_x', 'X')
        label_y = results.metadata.get('label_y', 'Y')
        
        # Scatter plot with CF-brightness encoding if CF and freq provided
        if cf_list is not None and freq_list is not None:
            # Create colormap for CFs
            unique_cfs = np.unique(cf_list)
            n_cfs = len(unique_cfs)
            cmap = plt.cm.Spectral
            cf_colors = {cf: cmap(i / max(1, n_cfs - 1)) for i, cf in enumerate(unique_cfs)}
            
            # Compute alpha based on |freq - CF|
            alphas = []
            for cf, freq in zip(cf_list, freq_list):
                freq_diff = abs(freq - cf)
                max_diff = max(abs(freq_list - cf_list)) if len(freq_list) > 0 else 1.0
                if max_diff > 0:
                    normalized_diff = freq_diff / max_diff
                    alpha = 0.2 + 0.8 * (1 - normalized_diff) ** 2
                else:
                    alpha = 1.0
                alphas.append(alpha)
            
            # Plot each point with CF color and frequency-based alpha
            for x, y, cf, alpha in zip(results.x_data, results.y_data, cf_list, alphas):
                ax.scatter(x, y, c=[cf_colors[cf]], alpha=alpha, s=60, edgecolors='none')
        else:
            # Simple scatter plot
            ax.scatter(results.x_data, results.y_data, alpha=0.5, s=20, label='Data')
        
        # Regression line
        x_range = np.array([results.x_data.min(), results.x_data.max()])
        y_pred = results.slope * x_range + results.intercept
        ax.plot(
            x_range, y_pred, 'r-', linewidth=2.5,
            label=f'Fit: y = {results.slope:.3f}x + {results.intercept:.3f}', zorder=100
        )
        
        # Unity line
        if show_unity_line:
            ax.plot(x_range, x_range, 'k--', alpha=0.3, linewidth=1.5, label='y = x', zorder=99)
        
        ax.set_xlabel(f'{label_x} Temporal Mean (normalized)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{label_y} Temporal Mean (normalized)', fontsize=12, fontweight='bold')
        
        title = f'Model Comparison (R² = {results.r_squared:.4f}, p = {results.p_value:.2e})'
        if cf_list is not None:
            title += '\nColor = CF, Brightness = |freq - CF|'
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        # Create CF legend if CF data provided
        if cf_list is not None:
            from matplotlib.patches import Patch
            legend_handles = []
            for cf in sorted(unique_cfs):
                patch = Patch(facecolor=cf_colors[cf], edgecolor='black', linewidth=0.5, label=f'CF = {cf:.0f} Hz')
                legend_handles.append(patch)
            
            # Add regression and unity line to legend
            from matplotlib.lines import Line2D
            legend_handles.append(Line2D([0], [0], color='r', linewidth=2.5, label=f'Fit: y = {results.slope:.3f}x + {results.intercept:.3f}'))
            if show_unity_line:
                legend_handles.append(Line2D([0], [0], color='k', linestyle='--', linewidth=1.5, label='y = x'))
            
            # Place legend outside plot area
            ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                     fontsize=9, framealpha=0.95, title='Characteristic Frequencies', title_fontsize=10)
        else:
            ax.legend(loc='lower right')
        
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        textstr = f'Slope: {results.slope:.4f}\\n'
        textstr += f'Intercept: {results.intercept:.4f}\\n'
        textstr += f'R²: {results.r_squared:.4f}\\n'
        textstr += f'p-value: {results.p_value:.2e}\\n'
        textstr += f'N: {results.metadata.get("n_points", len(results.x_data))}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, family='monospace')
        
        # Adjust layout to make room for legend
        if cf_list is not None:
            fig.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on right for legend
        else:
            fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved comparison plot to {save_path}")
        
        return fig
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved regression plot to {save_path}")
        
        return fig
    
    def plot_detailed_diagnostics(
        self,
        results: RegressionResults,
        save_path: Optional[Path] = None,
        figsize: tuple = (15, 10)
    ) -> plt.Figure:
        """
        Plot comprehensive regression diagnostics.
        
        Creates a 2x2 grid with:
        - Scatter plot with regression line
        - Residuals vs fitted values
        - Q-Q plot of residuals
        - Histogram of residuals
        
        Args:
            results: RegressionResults from fit()
            save_path: Optional path to save figure
            figsize: Figure size (width, height)
            
        Returns:
            matplotlib Figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        label_x = results.metadata.get('label_x', 'X')
        label_y = results.metadata.get('label_y', 'Y')
        
        # 1. Scatter plot with regression line
        ax1.scatter(results.x_data, results.y_data, alpha=0.5, s=20)
        x_range = np.array([results.x_data.min(), results.x_data.max()])
        y_pred = results.slope * x_range + results.intercept
        ax1.plot(x_range, y_pred, 'r-', linewidth=2)
        ax1.plot(x_range, x_range, 'k--', alpha=0.3, linewidth=1)
        ax1.set_xlabel(f'{label_x} (spikes/s)')
        ax1.set_ylabel(f'{label_y} (spikes/s)')
        ax1.set_title(f'Regression Fit (R² = {results.r_squared:.4f})')
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals vs fitted
        ax2.scatter(results.predictions, results.residuals, alpha=0.5, s=20)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Fitted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals vs Fitted')
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        stats.probplot(results.residuals, dist="norm", plot=ax3)
        ax3.set_title('Normal Q-Q Plot')
        ax3.grid(True, alpha=0.3)
        
        # 4. Histogram of residuals
        ax4.hist(results.residuals, bins=30, alpha=0.7, edgecolor='black')
        ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax4.set_xlabel('Residuals')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Residuals')
        ax4.grid(True, alpha=0.3)
        
        fig.suptitle(
            f'{label_y} vs {label_x}: Diagnostic Plots\n'
            f'n = {results.metadata.get("n_points", "N/A")}, '
            f'p = {results.p_value:.2e}',
            fontsize=14
        )
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved diagnostic plots to {save_path}")
        
        return fig
    
    def compute_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = 'pearson'
    ) -> dict:
        """
        Compute correlation coefficient between two arrays.
        
        Args:
            x: First array
            y: Second array
            method: 'pearson' or 'spearman'
            
        Returns:
            Dictionary with 'correlation' and 'p_value'
        """
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x_flat) | np.isnan(y_flat))
        x_clean = x_flat[valid_mask]
        y_clean = y_flat[valid_mask]
        
        if len(x_clean) < 2:
            return {'correlation': np.nan, 'p_value': np.nan, 'n': 0}
        
        if method == 'pearson':
            corr, p_val = stats.pearsonr(x_clean, y_clean)
        elif method == 'spearman':
            corr, p_val = stats.spearmanr(x_clean, y_clean)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        return {
            'correlation': corr,
            'p_value': p_val,
            'n': len(x_clean),
            'method': method
        }
    
