""" Orchestrate the folder management workflow. """

from .folder_creator import FolderCreator
from .metadata_saver import MetadataSaver
from .timestamp_generator import TimestampGenerator
from .model_builders import model_builders  # Registry of name builders

class ExperimentFolderManager:
    def __init__(self, base_dir, model_builder):

        self.params = {}
        self.model_builder = model_builder
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
    def create_batch_folder(self, save_json=True, save_text=True):
        """
        Orchestrate the complete folder creation workflow.

        Steps:
            1. Generate timestamp
            2. Build folder name using model builder
            3. Create folder
            4. Save metadata files
        """

        # 1. Generate timestamp
        timestamp = self.timestamp_generator.generate_timestamp()

        # 2. Build folder name
        # If model_builder is a string, get it from registry:
        if isinstance(self.model_builder, str):
            builder_func = model_builders.get(self.model_builder)
            folder_name = builder_func(self.params, timestamp)
        else:
            folder_name = self.model_builder(self.params, timestamp)    
        
        # 3. Create folder
        self.results_folder = self.folder_creator.create_folder(folder_name)

        # 4. Save metadata files
        if save_json:
            self.metadata_saver.save_metadata_json(self.results_folder, self.params)
        if save_text:
            self.metadata_saver.save_metadata_text(self.results_folder, self.params)
        
        return self.results_folder
    
    def create_subfolder(self, subfolder_name):
        """ Create a subfolder within the results folder."""

        if not self.results_folder:
            raise RuntimeError("Results folder not created yet. Call create_batch_folder() first.")
        return self.folder_creator.create_subfolder(self.results_folder, subfolder_name)
    def get_results_folder(self):
        """ Get the path to the results folder."""
        return self.results_folder
    
    
## Usage examples:

    
    # Using model name from registry:
# manager = (ExperimentFolderManager("./results", "bez2018")
#           .with_params(num_runs=10, num_cf=20, min_cf=125, max_cf=8000, 
#                       num_ANF=(4, 4, 4))
#           .create_batch_folder())

# Using cochlea model:
#manager = (ExperimentFolderManager("./results", "cochlea_zilany2014")
#           .with_params(num_runs=5, num_cf=20, min_cf=125, max_cf=8000)
#           .create_batch_folder())

# Using WSR model:
#manager = (ExperimentFolderManager("./results", "wsr_model")
#           .with_params(num_channels=128, frame_length=16, 
#                       time_constant=8, factor=2, shift=4)
#           .create_batch_folder())

# Custom timestamp format:
#manager = (ExperimentFolderManager("./results", "bez2018")
#           .with_timestamp_format("%Y-%m-%d_%H-%M-%S")
#           .with_params(num_runs=10, num_cf=20, min_cf=125, max_cf=8000, 
#                       num_ANF=(4, 4, 4))
#           .create_batch_folder())

# Create subfolders:

#manager.create_subfolder("plots")
#manager.create_subfolder("raw_data")