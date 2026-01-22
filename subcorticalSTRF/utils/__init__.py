"""
Folder management, logging, metadata, and result saving utilities.

This package provides tools for organizing experiment results, analysis outputs,
and any workflow that requires structured folders with automatic timestamping,
logging, metadata saving, and multi-format result persistence.

Main Components:
    - FolderManager: Orchestrates folder creation with automatic naming and metadata
    - FolderCreator: Low-level filesystem operations for directory creation
    - LoggingConfigurator: Sets up file and console logging with proper formatting
    - MetadataSaver: Saves experiment parameters to JSON, text, or YAML formats
    - ResultSaver: Saves simulation results in multiple formats (pickle, MATLAB, NPZ)
    - model_builders: Registry of naming functions for different model types

Example - Experiment workflow:
    from utils import FolderManager, LoggingConfigurator, ResultSaver

    # Create organized output folder
    manager = FolderManager("./results", "cochlea_zilany2014")
    manager.with_params(num_runs=10, num_cf=20, num_ANF=(128, 128, 128))
    output_dir = manager.create_folder()

    # Setup logging
    LoggingConfigurator.setup_basic(output_dir, 'experiment.log')

    # Run experiment
    results = run_simulation()

    # Save results in multiple formats
    saver = ResultSaver(output_dir)
    saver.save_all(results, base_filename='results', formats=['pickle', 'mat'])

Example - Analysis workflow:
    from utils import FolderManager, LoggingConfigurator, MetadataSaver

    # Create analysis output folder
    manager = FolderManager("./results", "model_comparison")
    output_dir = manager.with_params(
        model1='cochlea',
        model2='wsr',
        comparison_type='temporal_mean'
    ).create_folder()

    # Setup logging and save metadata
    LoggingConfigurator.setup_basic(output_dir)
    MetadataSaver().save_json(output_dir, analysis_params, 'metadata.json')

Example - Quick result saving:
    from utils import save_results

    # Convenience function for quick saves
    paths = save_results(
        data=my_results,
        save_dir='./output',
        filename='experiment_001',
        formats=['pickle', 'mat', 'npz']
    )
"""
# Import the main user-facing classes
from .folder_management import FolderCreator, FolderManager, ExperimentFolderManager
# ExperimentFolderManager is alias for backward compatibility
from .logging_configurator import LoggingConfigurator

# Import the registry of the model builders
from .model_builders import model_builders

# Import other components for potential direct use

from .metadata_saver import MetadataSaver
from .result_saver import ResultSaver, save_results
from .timestamp_utils import generate_timestamp, TimestampFormats

# Define the public API
__all__ = [
    # Main classes
    "FolderManager",
    "ExperimentFolderManager",  # Backward compatibility
    "LoggingConfigurator",

    # Registry for model builders
    "model_builders",

    # Other components
    "FolderCreator",
    "MetadataSaver",
    "ResultSaver",
    "save_results",
    "generate_timestamp",
    "TimestampFormats"
]
