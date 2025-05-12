# This will be the functional script adapted from sound_generator.py

import numpy as np
import sounddevice as sd
import math


# This script will define the functions, and then another module script will call these functions.

# def normalize_power(sound):
#     rms = np.sqrt(np.mean(sound ** 2)) # rms: root mean square
#     normalized_sound =  sound / rms
#     return normalized_sound


class SoundGen:

    def __init__(self, sample_rate: int, tau: float):
        """
        Initialize the CreateSound instance.
        :param sample_rate: Sample rate of sounds ( per second).
        :param tau: The ramping window in seconds.
        """
        self.sample_rate = sample_rate
        self.tau = tau


    def sound_maker(self, freq, num_harmonics, tone_duration, harmonic_factor):
        """
        :param freq: Base frequency in Hz.
        :param num_harmonics: Number of harmonic tones.
        :param tone_duration: Duration of each tone in seconds.
        :param harmonic_factor: Harmonic amplitude decay factor for each tone.
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
        return sound


    def ramp_in_out(self, sound):
        sound = sound.copy()
        L = int (self.tau * self.sample_rate)
        hw = np.hamming(2*L)
        sound[:L] *= hw[:L]
        sound[-L:] *= hw[L:]
        return sound


    def normalize_sound(self, sound) :
        rms = np.sqrt(np.mean(sound ** 2))  # rms: root mean square
        normalized_sound = 0.99 * sound / rms
        return normalized_sound

    def generate_sequence(self, freq, num_harmonics, tone_duration, harmonic_factor, total_duration, isi):

        # Generate the tone using the sound_maker method
        sound = self.sound_maker(freq,num_harmonics, tone_duration, harmonic_factor)

        # Normalize the sound
        normalized_sound = self.normalize_sound(sound)

        # Apply ramping to the normalized sound
        ramped_sound = self.ramp_in_out(normalized_sound)

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
