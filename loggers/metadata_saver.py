"""
Save metadata about experiment parameters to several file formats.
"""

import json
import os

class MetadataSaver:
    def save_json(self, folder_path, data, filename="params.json"):
        """
        Save metadata dictionary to a JSON file.

        Parameters:
        - folder_path: Directory where the file will be saved
        - data: Dictionary containing metadata
        - filename: Name of the JSON file (default: "params.json")
        """
        
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[INFO] Saved metadata to {filepath}")
        return filepath
        
    def save_text(self, folder_path, data, filename="stimulus_params.txt"):
        """
        Save metadata dictionary to a text file.

        Parameters:
        - folder_path: Directory where the file will be saved
        - data: Dictionary containing metadata
        - filename: Name of the text file (default: "stimulus_params.txt")
        """
        
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "w") as f:
            for key, value in data.items():
                f.write(f"{key}={value}\n")
        print(f"[INFO] Saved stimulus params to {filepath}")
        return filepath 
    
    def save_yaml(self, folder_path, data, filename="params.yaml"):
        """
        Save metadata dictionary to a YAML file.

        Parameters:
        - folder_path: Directory where the file will be saved
        - data: Dictionary containing metadata
        - filename: Name of the YAML file (default: "params.yaml")
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to save YAML files. Install it via 'pip install pyyaml'.")

        filepath = os.path.join(folder_path, filename)
        with open(filepath, "w") as f:
            yaml.dump(data, f)
        print(f"[INFO] Saved metadata to {filepath}")
        return filepath