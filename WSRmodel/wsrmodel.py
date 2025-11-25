import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

# Add parent directory and pyNSL to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))
sys.path.insert(0, str(current_dir / 'pyNSL-main'))

import pyNSL
from stim_generator.soundgen import SoundGen
from BEZ2018_meanrate  import generate_tone_generator, generate_stimuli_params
soundgen = SoundGen(16000, tau=0.005)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WSRModel:
    def __init__(self, params):
        self.params = params
        # Initialize model parameters here
    def process(self, audio, fs):
        # Process the audio signal using the WSR model
        channels = pyNSL.wav2aud(audio, fs, self.params, fs)
        return channels
    def get_filterbank_CFs(self, fs):
        filterbank = pyNSL.pyNSL.get_filterbank()
        CFs = filterbank[f'CF_{fs//1000}kHz'].squeeze()
        return CFs
    
    def get_channel_mask(self, fs, max_cf = 2500.0):
        CFs = self.get_filterbank_CFs(fs)
        mask = CFs <= max_cf
        num_channels = np.sum(mask)
        logger.info(f'Using {num_channels} channels with CF <= {max_cf} Hz (out of {len(CFs)} total channels)'
                    )
        return mask
    
    def process_filtered(self, audio, fs, max_cf=2500.0):
        mask = self.get_channel_mask(fs, max_cf)
        mask = mask[1:129]
        channels = self.process(audio, fs)
        filtered_channels = channels[:, mask]
        return filtered_channels

    def get_temporal_mean(self, channels):
        """
        Calculate temporal mean across time samples for each channel.
        
        Args:
            channels: Array of shape (n_time, n_channels)
            
        Returns:
            Array of shape (n_channels,) with temporal mean for each channel
        """
        temporal_mean = np.mean(channels, axis=0)
        logger.info(
            f'Calculated temporal mean across {channels.shape[0]} '
            f'time samples for {channels.shape[1]} channels.'
        )
        return temporal_mean
    
    def process_at_target_cfs(
        self, 
        audio: np.ndarray, 
        fs: int, 
        target_cfs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process audio and extract channels matching target CFs.
        
        This method is useful for comparing WSR output with cochlea/BEZ models
        that use specific characteristic frequencies.
        
        Args:
            audio: Audio signal array
            fs: Sampling frequency (Hz)
            target_cfs: Array of target characteristic frequencies (Hz)
            
        Returns:
            Tuple of (extracted_channels, corresponding_cfs):
                - extracted_channels: Array (n_time, n_target_cfs)
                - corresponding_cfs: Array of target CFs used
                
        Example:
            >>> from scipy.io import loadmat
            >>> # Load CFs from cochlea data
            >>> data = loadmat('cochlea_psths.mat')
            >>> target_cfs = data['cfs']
            >>> 
            >>> # Process audio at those specific CFs
            >>> model = WSRModel([8, 8, -1, 0])
            >>> from stim_generator.soundgen import SoundGen
            >>> soundgen = SoundGen(16000, tau=0.005)
            >>> tone = soundgen.sound_maker(freq=500, tone_duration=0.2)
            >>> channels, cfs = model.process_at_target_cfs(tone, 16000, target_cfs)
            >>> 
            >>> # Now channels match the structure of cochlea/BEZ output
            >>> temporal_mean = np.mean(channels, axis=0)  # (n_target_cfs,)
        """
        # Import here to avoid circular dependency
        from WSRmodel.cf_matcher import CFMatcher
        
        # Get WSR filterbank CFs
        wsr_cfs = self.get_filterbank_CFs(fs)
        
        # Create matcher
        matcher = CFMatcher(wsr_cfs, target_cfs)
        
        # Process audio with full filterbank
        full_response = self.process(audio, fs)
        
        # Extract channels matching target CFs
        extracted_channels, corresponding_cfs = matcher.extract_channels(full_response)
        
        logger.info(
            f'Processed audio and extracted {len(corresponding_cfs)} channels '
            f'matching target CFs'
        )
        
        return extracted_channels, corresponding_cfs
    


## Example usage:
params = [8, 8, -1, 0]  
model = WSRModel(params)
tone = soundgen.sound_maker(freq=500, num_harmonics=1, tone_duration=0.2, harmonic_factor= 0.5)
cfs = model.get_filterbank_CFs(16000)

channels = model.process(tone, 16000)
filtered_channels = model.process_filtered(tone, 16000, max_cf=2500.0)
logger.info(f"Filtered channels shape: {filtered_channels.shape}")

model.plot_auditory_spectrogram(tone, 16000)
# channels.shape = (num_samples, num_channels) (so here, for a 200 ms tone, it's (3200, 128)). Each column is the output of one auditory channel over time.


db_range = 60
tone_freq_range = (125, 2500, 20)
# To run it with multiple sounds
desired_dbs, desired_freqs = generate_stimuli_params(tone_freq_range, db_range)

# generate_tone_generator returns a generator - iterate over it directly
tone_generator = generate_tone_generator(soundgen, db_range, tone_freq_range, num_harmonics=1, duration=0.2, harmonic_factor=0.5)   
for db, freq, tone in tone_generator:  # ← Unpack the yielded tuple
    channels = model.process(tone, 16000)
    temporal_mean = model.get_temporal_mean(channels)
    logger.info(f"Tone {freq} Hz at {db} dB - Temporal mean shape: {temporal_mean.shape}")
        # temporal_mean.shape = (num_channels,) (here, (128,)) - the average response of each channel over time for this tone

        
