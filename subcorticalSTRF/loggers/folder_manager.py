""" Orchestrate the folder management workflow for any use case. """

from .folder_creator import FolderCreator
from .metadata_saver import MetadataSaver
from .timestamp_generator import TimestampGenerator
from .model_builders import model_builders  # Registry of name builders


class FolderManager:
    """
    Generic folder manager for organizing outputs with automatic timestamping
    and metadata saving.

    Can be used for experiments, model comparisons, analyses, or any workflow
    that needs organized output directories with metadata.

    Args:
        base_dir: Base directory where folders will be created
        name_builder: Either a string (registered in model_builders) or a
        callable that takes (params, timestamp) and returns a folder name
    """

    def __init__(self, base_dir, name_builder=None):

        self.params = {}
        self.name_builder = name_builder
        self.results_folder = None

        # Composition: inject dependencies
        self.folder_creator = FolderCreator(base_dir)
        self.metadata_saver = MetadataSaver()
        self.timestamp_generator = TimestampGenerator()

    def with_params(self, **kwargs):
        """Set experiment parameters."""
        self.params.update(kwargs)
        return self

    def with_timestamp_format(self, format_string):
        """Set custom timestamp format."""
        self.timestamp_generator = TimestampGenerator(format_string)
        return self

    def create_folder(self, folder_name=None, save_json=True, save_text=True):
        """
        Orchestrate the complete folder creation workflow.

        Args:
            folder_name: Optional explicit folder name. If None, uses
            name_builder with params and timestamp.
            save_json: Save params as JSON metadata file
            save_text: Save params as text metadata file

        Steps:
            1. Generate timestamp
            2. Build folder name using name_builder (if folder_name not
            provided)
            3. Create folder
            4. Save metadata files
        """

        # 1. Generate timestamp
        timestamp = self.timestamp_generator.generate_timestamp()

        # 2. Build folder name
        if folder_name is None:
            if self.name_builder is None:
                # Default: use timestamp only
                folder_name = f"results_{timestamp}"
            elif isinstance(self.name_builder, str):
                # Get builder from registry
                builder_func = model_builders.get(self.name_builder)
                folder_name = builder_func(self.params, timestamp)
            else:
                # Use custom callable
                folder_name = self.name_builder(self.params, timestamp)
        # 3. Create folder
        self.results_folder = self.folder_creator.create_folder(folder_name)

        # 4. Save metadata files
        if save_json:
            self.metadata_saver.save_json(self.results_folder, self.params)
        if save_text:
            self.metadata_saver.save_text(self.results_folder, self.params)

        return self.results_folder

    def create_subfolder(self, subfolder_name):
        """ Create a subfolder within the results folder."""

        if not self.results_folder:
            raise RuntimeError("Results folder not created yet.",
                               "Call create_folder() first.")
        return self.folder_creator.create_subfolder(self.results_folder,
                                                    subfolder_name)

    def get_results_folder(self):
        """ Get the path to the results folder."""
        return self.results_folder


# Backward compatibility alias
ExperimentFolderManager = FolderManager


# Usage examples:

# Using model name from registry:
# manager = (FolderManager("./results", "bez2018")
#           .with_params(num_runs=10, num_cf=20, min_cf=125, max_cf=8000,
#                       num_ANF=(4, 4, 4))
#           .create_folder())

# Using cochlea model:
# manager = (FolderManager("./results", "cochlea_zilany2014")
#           .with_params(num_runs=5, num_cf=20, min_cf=125, max_cf=8000)
#           .create_folder())

# Using WSR model:
# manager = (FolderManager("./results", "wsr_model")
#           .with_params(num_channels=128, frame_length=16,
#                       time_constant=8, factor=2, shift=4)
#           .create_folder())

# Custom timestamp format:
# manager = (FolderManager("./results", "bez2018")
#           .with_timestamp_format("%Y-%m-%d_%H-%M-%S")
#           .with_params(num_runs=10, num_cf=20, min_cf=125, max_cf=8000,
#                       num_ANF=(4, 4, 4))
#           .create_folder())

# Custom name builder function:
# def custom_namer(params, timestamp):
#     return f"analysis_{params['name']}_{timestamp}"
# manager = FolderManager("./results", custom_namer).with_params(name="test")
#                                                      .create_folder()

# Simple folder with explicit name:
# manager = FolderManager("./results").create_folder("my_analysis_20250101")

# Simple folder with just timestamp:
# manager = FolderManager("./results").create_folder()

# Create subfolders:

# manager.create_subfolder("plots")
# manager.create_subfolder("raw_data")
