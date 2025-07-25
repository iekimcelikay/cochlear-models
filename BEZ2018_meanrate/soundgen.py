# This will be the functional script adapted from sound_generator.py

import numpy as np
import math
import thorns
import time
from thorns import waves as wv
import scipy.signal as signal
from scipy.signal import firwin, filtfilt

# Type annotations are unsupported in Python 2.  So will remove them.

# This script will define the functions, and then another module script will call these functions.

class SoundGen:

    def __init__(self, sample_rate, tau):
        """
        Initialize the CreateSound instance.
        :param sample_rate: Sample rate of sounds ( per second).
        :param tau: The ramping window in seconds.
        """
        self.sample_rate = sample_rate
        self.tau = tau


    def sound_maker(self, freq, num_harmonics, tone_duration, harmonic_factor, dbspl):
        """
        :param freq: Base frequency in Hz.
        :param num_harmonics: Number of harmonic tones.
        :param tone_duration: Duration of each tone in seconds.
        :param harmonic_factor: Harmonic amplitude decay factor for each tone.
        :param dbspl: Desired dbspl (loudness) level.
        :return: sound: np.ndarray: array of audio samples representing the harmonic complex tone.
        """
        # Create the time array
        t = np.linspace(0, tone_duration, int(self.sample_rate * tone_duration), endpoint=False)
        # Initialize the sound array
        sound = np.zeros_like(t)
        # Generate the harmonics
        for k in range(1, num_harmonics + 1):
            omega = 2 * np.pi * freq * k
            harmonic = np.sin(omega * t)
            amplitude = (harmonic_factor ** (k - 1)) / num_harmonics
            sound = sound + amplitude * harmonic

        sound = np.clip(sound, -1.0, 1.0)
        # Normalize the sound
        normalized_sound = thorns.waves.set_dbspl(sound, dbspl)
        return normalized_sound

    def noise_maker(self, tone_duration, seed=None):
        if seed is not None:
            np.random.seed(seed)
        sound = np.random.normal(0, 1, tone_duration * self.sample_rate)
        return sound

    def bandpass_filter_FIR(self, tone_duration, dbspl, lowcut, highcut, numtaps, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Generate the white noise using noise_maker method
        noise = self.noise_maker(tone_duration)
        # design FIR bandpass filter
        fir_coeffs = firwin(numtaps, [lowcut, highcut], pass_zero=True, fs = self.sample_rate)

        # apply zero-phase FIR filter (no delay)
        filtered_noise = filtfilt(fir_coeffs, [1.0], noise)

        # set to desired dbspl using thorns
        normalized_noise = thorns.waves.set_dbspl(filtered_noise, dbspl)
        return normalized_noise

    def generate_multiple_band_limited_noises(self, n_trials, tone_duration, lowcut, highcut, numtaps,dbspl,base_seed=0):
        """
        Generate multiple reproducible band-limited white noise stimuli for the Zilany model.

        Parameters:
            n_trials (int): Number of stimuli to generate
            tone_duration (float): Duration in seconds
            lowcut (float): Lower cutoff frequency of bandpass filter (Hz)
            highcut (float): Upper cutoff frequency of bandpass filter (Hz)
            numtaps (int): FIR filter length
            dbspl (float): Desired dB SPL
            base_seed (int): Seed offset to allow reproducible variation

        Returns:
            list of numpy.ndarray: List of filtered, level-adjusted noise signals
        """
        fir_coeffs = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=self.sample_rate)
        n_samples = int(tone_duration * self.sample_rate)
        stimuli = []

        for i in range(n_trials):
            seed = base_seed + i
            np.random.seed(seed)
            white_noise = np.random.normal(0, 1, n_samples)
            filtered = filtfilt(fir_coeffs, [1.0], white_noise)
            scaled = thorns.waves.set_dbspl(filtered, dbspl)
            stimuli.append(scaled)

        return stimuli

    def ramp_in_out(self, sound):
        sound = sound.copy()
        L = int (self.tau * self.sample_rate)
        hw = np.hamming(2*L)
        sound[:L] *= hw[:L]
        sound[-L:] *= hw[L:]
        return sound

    def sine_ramp(self, sound):
        L = int(self.tau * self.sample_rate)
        t = np.linspace(0, L / self.sample_rate, L)
        sine_window = np.sin(np.pi * t / (2 * self.tau)) ** 2  # Sine fade-in

        sound = sound.copy()
        sound[:L] *= sine_window  # Apply fade-in
        sound[-L:] *= sine_window[::-1]  # Apply fade-out

        return sound

    def generate_sequence(self, freq, num_harmonics, tone_duration, harmonic_factor, dbspl, total_duration, isi):

        # Generate the tone using the sound_maker method
        sound = self.sound_maker(freq,num_harmonics, tone_duration, harmonic_factor, dbspl)
        # Sine ramp
        ramped_sound = self.sine_ramp(sound)
        # Calculate the number of tones that can fit into the total duration
        total_samples = int(total_duration * self.sample_rate)
        isi_samples = int(isi * self.sample_rate)
        num_tones = int((total_samples + isi_samples) // (len(ramped_sound) + isi_samples))

        # Generate the sequence with ISI gaps between each tone
        sequence = np.array([])

        for _ in range(num_tones):
            sequence = np.concatenate((sequence, ramped_sound))

            # Add ISI (silent gap) between tones
            sequence = np.concatenate((sequence, np.zeros(isi_samples)))

        # Ensure the sequence doesn't exceed the total duration
        if len(sequence) > total_samples:
            sequence = sequence[:total_samples]

        return sequence




####
