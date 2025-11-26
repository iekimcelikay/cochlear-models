"""
subcorticalSTRF: Auditory nerve fiber model comparison and analysis framework.

This package provides tools and utilities for working with three major auditory
nerve fiber models:
1. BEZ2018 - Mechanistic model (Bruce et al. 2018)
2. Cochlea - Phenomenological model (Python cochlea package)
3. WSR - Wavelet Spectrogram Representation (custom implementation)

Core Components:
    - loggers: Centralized logging, folder management, and metadata saving
    - stim_generator: Audio stimulus generation utilities
    - models: Model implementations (BEZ2018, WSR, comparison tools)
    - analyses: Data analysis and comparison workflows

Quick Start:
    from subcorticalSTRF.loggers import FolderManager, LoggingConfigurator
    from subcorticalSTRF.stim_generator import SoundGen

    # Create output folder with automatic timestamping
    manager = FolderManager("./results", "bez2018").create_folder()

    # Setup logging
    LoggingConfigurator.setup_basic(manager.get_results_folder())

Created: 2025-01-16 08:00:00
"""

# Expose primary utilities for convenient importing
from stim_generator import SoundGen

# Expose logging utilities
from loggers import (
    FolderManager,
    ExperimentFolderManager,  # Backward compatibility alias
    LoggingConfigurator,
    MetadataSaver,
    model_builders,
)

# Public API for this package
__all__ = [
    "SoundGen",
    "FolderManager",
    "ExperimentFolderManager",
    "LoggingConfigurator",
    "MetadataSaver",
    "model_builders",
]
