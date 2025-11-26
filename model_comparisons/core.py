"""
Core data structures for model comparison
This module contains dataclasses that represent model results and comparison configurations.

"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

@dataclass
class ModelResults:
    """Container for model outputs"""
    temporal_means: np.ndarray  # Shape: (n_channels, n_conditions)
    channel_cfs: np.ndarray
    frequencies: List[float]
    db_levels: List[float]
    model_name: str
    metadata: Dict = field(default_factory=dict)

@dataclass
class ComparisonConfig:
    """Configuration for comparing two models"""
    bez_results_path: str
    wsr_results_path: str
    output_dir: str
    align_cfs: bool = True
    min_cf: float = 180.0


@dataclass
class RegressionResults:
    """Results from linear regression analysis."""
    slope: float
    intercept: float 
    r_squared: float
    p_value: float 
    predictions: np.ndarray  # Predicted values from the regression
    residuals: np.ndarray  # Residuals (y - y_pred)
    x_data: np.ndarray  # Independent variable data (cleaned)
    y_data: np.ndarray  # Dependent variable data (cleaned)
    metadata: Dict = field(default_factory=dict)  # Additional info (std_err, n_points, etc.)

@dataclass
class ComparisonResults:
    """Container for comparison results between two models"""
    regression_results: RegressionResults
    wsr_results: ModelResults
    bez_results: ModelResults
    aligned_data: Tuple[np.ndarray, np.ndarray]  # (wsr_aligned, bez_aligned)
    cf_mapping: List[Dict]
    metadata: Dict = field(default_factory=dict)