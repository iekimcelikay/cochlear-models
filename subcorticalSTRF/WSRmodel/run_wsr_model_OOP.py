#run_wsr_model.py
"""
Class-based runner for running WSR Model

"""
import numpy as np
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
import scipy.io as sio
import pickle 

from WSRmodel.wsrmodel import WSRModel
from stim_generator.soundgen import SoundGen
from BEZ2018_meanrate.stimulus_utils import generate_tone_generator, generate_stimuli_params
from loggers import ExperimentFolderManager

@dataclass
class WSRConfig:
    """ Configuration for WSR model simulation"""
    # Audio parameters
    sample_rate: int = 16000
    tau_ramp: float = 0.005

    # Stimulus parameters
    num_harmonics: int = 1
    tone_duration: float = 0.200
    harmonic_factor: float = 0.5
    db_range: int = 60
    freq_range: tuple = (125, 2500, 20)

    # WSR model parameters
    wsr_params: List[int] = None
    max_cf: float = 2500.0

    # Output
    results_folder: str = './results'

    def __post_init__(self):
        if self.wsr_params is None:
            self.wsr_params = [8, 8, -1, 0]

class WSRExperiment:

    def __init__(self, config: WSRConfig):
        self.config = config
        self.sound_gen = SoundGen
        self.model = WSRModel(config.wsr_params)
        self.results = {} 
        self.save_dir = None

    def setup_output_folder(self):
        desired_dbs, desired_freqs = generate_stimuli_params(
            self.config.freq_range, self.config.db_range
        )

        # Get channel info
        mask = self.model.get_channel_mask(self.config.sample_rate, self.config.max_cf)
        num_channels = np.sum(mask)

        folder_manager = (
            ExperimentFolderManager(self.config.results_folder, "wsr_model")
            .with_params(
                num_channels=int(num_channels),
                frame_length=self.config.wsr_params[0],
                time_constant=self.config.wsr_params[1],
                factor=self.config.wsr_params[2],
                shift=self.config.wsr_params[3],
                sample_rate=self.config.sample_rate,
                tone_duration=self.config.tone_duration,
                db_range=list(desired_dbs),
                freq_range=list(desired_freqs),
            )
            .create_batch_folder()
        )
        
        self.save_dir = folder_manager.get_results_path()
        print(f"[INFO] Results will be saved to: {self.save_dir}")
        
    def process_tone(self, freq: float, db: float, tone: np.ndarray) -> Dict:
        """Process a single tone through WSR model."""
        # Get filtered response
        channels = self.model.process_filtered(
            tone, self.config.sample_rate, max_cf=self.config.max_cf
        )
        
        # Calculate temporal mean
        temporal_mean = self.model.get_temporal_mean(channels)
        
        return {
            'freq': freq,
            'db': db,
            'temporal_mean': temporal_mean,
            'full_response': channels,
        }
    
    def run(self, save_full_response: bool = False):
        """Run the full experiment."""
        print("[INFO] Starting WSR model experiment...")
        start_time = time.time()
        
        # Setup output
        self.setup_output_folder()
        
        # Generate tone iterator
        tone_generator = generate_tone_generator(
            self.sound_gen,
            self.config.db_range,
            self.config.freq_range,
            num_harmonics=self.config.num_harmonics,
            duration=self.config.tone_duration,
            harmonic_factor=self.config.harmonic_factor
        )
        
        # Process all tones
        for tone_idx, (db, freq, tone) in enumerate(tone_generator):
            print(f"[INFO] Processing tone {tone_idx}: {freq:.1f} Hz @ {db} dB")
            
            result = self.process_tone(freq, db, tone)
            
            # Optionally drop full response to save memory
            if not save_full_response:
                result.pop('full_response')
            
            key = f"freq_{freq:.1f}hz_db_{db}"
            self.results[key] = result
        
        # Save results
        self.save_results()
        
        elapsed = time.time() - start_time
        print(f"[INFO] Completed in {elapsed:.2f} seconds")
        print(f"[INFO] Processed {len(self.results)} conditions")
        
        return self.results
    
    def save_results(self):
        """Save results in multiple formats."""
        # Save as .mat file
        mat_file = Path(self.save_dir) / "wsr_responses.mat"
        sio.savemat(str(mat_file), self.results)
        print(f"[INFO] Saved MATLAB format: {mat_file}")
        
        # Save as .npz file
        npz_file = Path(self.save_dir) / "wsr_responses.npz"
        np.savez(str(npz_file), **self.results)
        print(f"[INFO] Saved NumPy format: {npz_file}")

        # Save as pickle file 
        wsr_results_path = Path(self.save_dir) / "wsr_results.pkl" 
        with open(wsr_results_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save channel CFs
        cfs = self.model.get_filterbank_CFs(self.config.sample_rate)
        mask = self.model.get_channel_mask(self.config.sample_rate, self.config.max_cf)
        filtered_cfs = cfs[mask]
        
        np.save(Path(self.save_dir) / "channel_cfs.npy", filtered_cfs)
        print(f"[INFO] Saved {len(filtered_cfs)} channel CFs")


# ==== Usage ====
if __name__ == "__main__":
    # Configure experiment
    config = WSRConfig(
        db_range=60,
        freq_range=(125, 2500, 20),
        tone_duration=0.2,
    )
    
    # Run experiment
    experiment = WSRExperiment(config)
    results = experiment.run(save_full_response=False)

    