"""
WSR (Wavelet Spectrogram Representation) Model Implementation.

This module provides the WSRModel class for processing audio signals using
wavelet-based spectral representation. It supports feature extraction,
filtering by characteristic frequency, and comparison with other auditory
models (cochlea, BEZ2018).

Created: 2025-01-16 08:00:00
"""

import logging
from typing import Tuple

import numpy as np

# Conditional import for pyNSL (lazy loading to handle optional dependency)
try:
    import pyNSL
except ImportError:
    pyNSL = None

from stim_generator.soundgen import SoundGen
from BEZ2018_meanrate import generate_tone_generator

# Configure module logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global sound generator
soundgen = SoundGen(16000, tau=0.005)


class WSRModel:
    """Wavelet Spectrogram Representation (WSR) auditory model."""

    def __init__(self, params):
        """Initialize WSR model with parameters.

        Args:
            params: List of model parameters for pyNSL.wav2aud().
        """
        self.params = params

    def process(self, audio, fs):
        """Process audio signal using WSR model.

        Args:
            audio: Audio signal array.
            fs: Sampling frequency (Hz).

        Returns:
            Array of processed channels (n_time, n_channels).
        """
        if pyNSL is None:
            raise ImportError(
                "pyNSL is required for WSRModel but is not installed. "
                "install it manually or check the pyNSL-main/ directory."
            )
        channels = pyNSL.wav2aud(audio, fs, self.params, fs)
        return channels

    def get_filterbank_CFs(self, fs):
        """Get characteristic frequencies (CFs) of WSR filterbank.

        Args:
            fs: Sampling frequency (Hz).

        Returns:
            Array of characteristic frequencies.
        """
        if pyNSL is None:
            raise ImportError(
                "pyNSL is required for WSRModel but is not installed."
            )
        filterbank = pyNSL.pyNSL.get_filterbank()
        cfs = filterbank[f'CF_{fs//1000}kHz'].squeeze()
        return cfs

    def get_channel_mask(self, fs, max_cf=2500.0):
        """Generate mask for channels with CF <= max_cf.

        Args:
            fs: Sampling frequency (Hz).
            max_cf: Maximum characteristic frequency (Hz). Default 2500.0.

        Returns:
            Boolean mask array of shape (n_channels,).
        """
        cfs = self.get_filterbank_CFs(fs)
        mask = cfs <= max_cf
        num_channels = np.sum(mask)
        logger.info(
            f"Using {num_channels} channels with CF <= {max_cf} Hz "
            f"(out of {len(cfs)} total channels)"
        )
        return mask

    def process_filtered(self, audio, fs, max_cf=2500.0):
        """Process audio and return channels filtered by CF.

        Args:
            audio: Audio signal array.
            fs: Sampling frequency (Hz).
            max_cf: Maximum characteristic frequency (Hz). Default 2500.0.

        Returns:
            Array of filtered channels (n_time, n_filtered_channels).
        """
        mask = self.get_channel_mask(fs, max_cf)
        mask = mask[1:129]
        channels = self.process(audio, fs)
        filtered_channels = channels[:, mask]
        return filtered_channels

    def get_temporal_mean(self, channels):
        """Calculate temporal mean across time samples for each channel.

        Args:
            channels: Array of shape (n_time, n_channels).

        Returns:
            Array of shape (n_channels,) with temporal mean for each channel.
        """
        temporal_mean = np.mean(channels, axis=0)
        logger.info(
            f"Calculated temporal mean across {channels.shape[0]} "
            f"time samples for {channels.shape[1]} channels."
        )
        return temporal_mean

    def process_at_target_cfs(
        self,
        audio: np.ndarray,
        fs: int,
        target_cfs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process audio and extract channels matching target CFs.

        This method is useful for comparing WSR output with cochlea/BEZ
        models that use specific characteristic frequencies.

        Args:
            audio: Audio signal array.
            fs: Sampling frequency (Hz).
            target_cfs: Array of target characteristic frequencies (Hz).

        Returns:
            Tuple of (extracted_channels, corresponding_cfs):
                - extracted_channels: Array (n_time, n_target_cfs).
                - corresponding_cfs: Array of target CFs used.

        Example:
            >>> from scipy.io import loadmat
            >>> # Load CFs from cochlea data
            >>> data = loadmat('cochlea_psths.mat')
            >>> target_cfs = data['cfs']
            >>>
            >>> # Process audio at those specific CFs
            >>> model = WSRModel([8, 8, -1, 0])
            >>> from stim_generator.soundgen import SoundGen
            >>> soundgen_inst = SoundGen(16000, tau=0.005)
            >>> tone = soundgen_inst.sound_maker(
            ...     freq=500, tone_duration=0.2
            ... )
            >>> channels, cfs = model.process_at_target_cfs(
            ...     tone, 16000, target_cfs
            ... )
            >>>
            >>> # Now channels match cochlea/BEZ output structure
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
        extracted_channels, corresponding_cfs = matcher.extract_channels(
            full_response
        )

        logger.info(
            f"Processed audio and extracted {len(corresponding_cfs)} "
            f"channels matching target CFs"
        )

        return extracted_channels, corresponding_cfs


# Example usage and test scenarios
if __name__ == "__main__":
    """Example: WSR model processing pipeline."""
    params = [8, 8, -1, 0]
    model = WSRModel(params)

    # Test with simple tone
    tone = soundgen.sound_maker(
        freq=500, num_harmonics=1, tone_duration=0.2, harmonic_factor=0.5
    )
    cfs = model.get_filterbank_CFs(16000)

    channels = model.process(tone, 16000)
    filtered_channels = model.process_filtered(tone, 16000, max_cf=2500.0)
    logger.info(f"Filtered channels shape: {filtered_channels.shape}")

    # channels.shape = (num_samples, num_channels) (e.g., (3200, 128))
    # Each column is the output of one auditory channel over time.

    # Test with multiple sounds at different dB levels and frequencies
    db_range = 60
    tone_freq_range = (125, 2500, 20)

    # generate_tone_generator returns a generator - iterate directly
    tone_generator = generate_tone_generator(
        soundgen, db_range, tone_freq_range,
        num_harmonics=1, duration=0.2, harmonic_factor=0.5
    )
    for db, freq, tone in tone_generator:  # Unpack yielded tuple
        channels = model.process(tone, 16000)
        temporal_mean = model.get_temporal_mean(channels)
        logger.info(
            f"Tone {freq} Hz at {db} dB - "
            f"Temporal mean shape: {temporal_mean.shape}"
        )
        # temporal_mean.shape = (num_channels,) e.g., (128,)
        # Average response of each channel over time for this tone
