"""
Create directories on the filesystem for analysis results. 
Created on: 2025-11-23
Ekim Celikay (With CoPilot assistance)
"""
import os

class FolderCreator:

    def __init__(self, base_dir):
        """
        Args: 
            base_dir (str): Base directory where folders will be created.
        """
        self.base_dir = base_dir
    def create_folder(self, folder_name):
        """
        Create folder within base_dir.

        args:
            folder_name (str): Name of the folder to create.

        returns:
            str: Full path to the created folder.
        """
        folder_path = os.path.join(self.base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path  
    def create_subfolder(self, parent_folder, subfolder_name):
        """
        Create subfolder within a parent folder.

        args:
            parent_folder (str): Path to the parent folder.
            subfolder_name (str): Name of the subfolder to create.

        returns:
            str: Full path to the created subfolder.
        """
        subfolder_path = os.path.join(parent_folder, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)
        return subfolder_path