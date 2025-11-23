""" 
Folder management and experiment logging
This package provides tools for organizing experiment results into 
structured folders with automatic timestamping and metadata saving. 

Example:
from subcorticalSTRF.loggers import ExperimentFolderManager

manager = (ExperimentFolderManager("./results", "bez2018")
           .with_params(num_runs=10, num_cf=20, num_ANF=(4,4,4))
           .create_batch_folder())
"""
# Import the main user-facing class
from .folder_manager import ExperimentFolderManager

# Import the registry of the model builders
from .model_builders import model_builders

# Import other components for potential direct use
from .folder_creator import FolderCreator
from .metadata_saver import MetadataSaver
from .timestamp_generator import TimestampGenerator, TimestampFormats

# Define the public API
__all__ = [
    # Main class
    "ExperimentFolderManager",
    # Registry for model builders
    "model_builders",

    # Other components
    "FolderCreator",
    "MetadataSaver",
    "TimestampGenerator",
    "TimestampFormats"
]