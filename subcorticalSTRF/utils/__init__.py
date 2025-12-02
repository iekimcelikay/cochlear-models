"""
Stimulus and simulation parameter management, folder management,
logging configuration, and metadata saving utilities.

This package provides tools for creating stimulus parameters,
organizing experiment results, analysis outputs,
and any workflow that requires structured folders with automatic
timestamping, logging, and metadata saving.

# TODO: update this docstring
Main Components:
    - FolderManager: Orchestrates folder creation with automatic naming and
    metadata
    - LoggingConfigurator: Sets up file and console logging
    - MetadataSaver: Saves parameters/results to JSON, text, or YAML
    - model_builders: Registry of naming functions for different use cases

Example - Experiment workflow:
    from utils import FolderManager, LoggingConfigurator

    # Create organized output folder
    manager = (FolderManager("./results", "bez2018")
               .with_params(num_runs=10, num_cf=20, num_ANF=(4,4,4))
               .create_folder())

    # Setup logging
    LoggingConfigurator.setup_basic(manager.get_results_folder(),
                                                'experiment.log')

Example - Analysis workflow:
    from utils import FolderManager, LoggingConfigurator, MetadataSaver

    # Custom folder naming
    def analysis_namer(params, timestamp):
        return f"analysis_{params['name']}_{timestamp}"

    manager = FolderManager("./results", analysis_namer)
    output_dir = manager.with_params(name="comparison").create_folder()

    # Setup logging and save results
    LoggingConfigurator.setup_basic(output_dir)
    MetadataSaver().save_json(output_dir, results_dict)
"""
# Import the main user-facing classes
from .folder_manager import FolderManager, ExperimentFolderManager
# ExperimentFolderManager is alias for backward compatibility
from .logging_configurator import LoggingConfigurator

# Import the registry of the model builders
from .model_builders import model_builders

# Import other components for potential direct use
from .folder_creator import FolderCreator
from .metadata_saver import MetadataSaver
from .result_saver import ResultSaver, save_results
from .timestamp_generator import TimestampGenerator, TimestampFormats

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
    "TimestampGenerator",
    "TimestampFormats"
]
