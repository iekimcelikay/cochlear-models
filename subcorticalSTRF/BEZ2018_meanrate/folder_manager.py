import os
import json
import time
from datetime import datetime

class FolderManager:
    def __init__(self, base_dir, num_repeats, num_cf, num_ANF, stimulus_params):
        self.base_dir = base_dir
        self.num_repeats = num_repeats
        self.num_cfs = num_cf
        self.num_ANF = num_ANF # the order in my module is (lsr, msr,hsr). this is different in cochlea, for example.
        self.stimulus_params = stimulus_params
        self.results_folder = None
        self.last_run_time_stamp = None


    def create_batch_folder(self):
        lsr, msr, hsr = self.num_ANF
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"psth_batch_{self.num_repeats}runs_{self.num_cfs}cfs_{lsr}-{msr}-{hsr}fibers_{self.timestamp}"
        self.results_folder = os.path.join(self.base_dir, folder_name)
        os.makedirs(self.results_folder, exist_ok=True)
        print(f"[INFO] Output directory: {self.results_folder}")
        self.save_metadata()
        self.save_stimulus_params_text()


    def save_metadata(self, metadata_dict=None, filename="params.json"):
        if not self.results_folder:
            raise RuntimeError("Results folder not created yet!")
        if metadata_dict is None:
            metadata_dict = self.stimulus_params
        filepath = os.path.join(self.results_folder, filename)
        with open(filepath, "w") as f:
            json.dump(metadata_dict, f, indent=4)
        print(f"[INFO] Saved metadata to {filepath}")

    def save_stimulus_params_text(self, filename="stimulus_params.txt"):
        if not self.results_folder:
            raise RuntimeError("Results folder not created yet!")

        filepath = os.path.join(self.results_folder, filename)
        with open(filepath, "w") as f:
            for key, value in self.stimulus_params.items():
                f.write(f"{key}={value}\n")
        print(f"[INFO] Saved stimulus params to {filepath}")

