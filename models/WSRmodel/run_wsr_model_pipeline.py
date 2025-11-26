# run_wsr_model_pipeline.py

from typing import Iterator, Dict, Tuple
from pathlib import Path
import numpy as np

from BEZ2018_meanrate.stimulus_utils import generate_tone_generator
from WSRmodel.wsrmodel import WSRModel
from stim_generator.soundgen import SoundGen
from loggers import FolderManager, LoggingConfigurator


class StimulusGenerator:
    """Generate stimuli for the experiment."""

    def __init__(self, sound_gen, db_range, freq_range, **tone_params):
        self.sound_gen = sound_gen
        self.db_range = db_range
        self.freq_range = freq_range
        self.tone_params = tone_params

    def generate(self) -> Iterator[Tuple[float, float, np.ndarray]]:
        """Yield (db, freq, tone) tuples."""
        return generate_tone_generator(
            self.sound_gen,
            self.db_range,
            self.freq_range,
            **self.tone_params
        )


class WSRProcessor:
    """Process stimuli through WSR model."""

    def __init__(self, model, fs, max_cf=2500.0):
        self.model = model
        self.fs = fs
        self.max_cf = max_cf

    def process(self, stimuli: Iterator) -> Iterator[Dict]:
        """Process stimuli and yield results."""
        for db, freq, tone in stimuli:
            channels = self.model.process_filtered(tone,
                                                   self.fs,
                                                   max_cf=self.max_cf)
            temporal_mean = self.model.get_temporal_mean(channels)

            yield {
                'db': db,
                'freq': freq,
                'temporal_mean': temporal_mean,
                'channels': channels,
            }


class ResultCollector:
    """Collect and save results."""

    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.results = {}

    def collect(self, results: Iterator[Dict]):
        """Collect results from iterator."""
        for idx, result in enumerate(results):
            freq = result['freq']
            db = result['db']
            key = f"freq_{freq:.1f}hz_db_{db}"
            self.results[key] = result
            print(f"[INFO] Collected result {idx}: {freq:.1f} Hz @ {db} dB")

        return self.results

    def save(self, formats=['mat', 'npz']):
        """Save in multiple formats."""
        if 'mat' in formats:
            import scipy.io as sio
            sio.savemat(self.save_dir / "wsr_responses.mat", self.results)

        if 'npz' in formats:
            np.savez(self.save_dir / "wsr_responses.npz", **self.results)

        print(f"[INFO] Saved {len(self.results)} results to {self.save_dir}")


# ==== Usage: Build a pipeline ====
if __name__ == "__main__":

    # Setup
    sound_gen = SoundGen(16000, 0.005)
    model = WSRModel([8, 8, -1, 0])

    # Create output folder using FolderManager
    folder_mgr = (FolderManager("./results", "wsr_model")
                  .with_params(
                      num_channels=128,
                      frame_length=8,
                      time_constant=8,
                      factor=-1,
                      shift=0,
                      sample_rate=16000
                  )
                  .create_folder())

    # Setup logging
    LoggingConfigurator.setup_basic(folder_mgr, 'pipeline.log')

    # Build pipeline
    stimuli = StimulusGenerator(
        sound_gen,
        db_range=(50, 90, 10),
        freq_range=(125, 2500, 20),
        num_harmonics=1,
        duration=0.2,
        harmonic_factor=0.5
    ).generate()

    processor = WSRProcessor(model, fs=16000, max_cf=2500.0)

    collector = ResultCollector(folder_mgr)

    # Run pipeline
    results = collector.collect(processor.process(stimuli))
    collector.save(formats=['npz'])

    print(f"Pipeline complete. Results saved to: {folder_mgr}")
