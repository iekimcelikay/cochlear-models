# This will be the functional script adapted from sound_generator.py

import numpy as np
import sounddevice as sd
import math
import thorns


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

        sound = np.clip(sound, -1.0, 1.0)
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
        normalized_sound = np.clip(normalized_sound, -1.0, 1.0)  # Clipping to [-1.0, 1.0]
        return normalized_sound

    def exponential_ramp(self, sound):
        L = int(self.tau * self.sample_rate)
        t = np.linspace(0, L / self.sample_rate, L)
        fade_in = 1 - np.exp(-t / self.tau)  # Fade-in (exponential)
        fade_out = np.exp(-t / self.tau)  # Fade-out (exponential)

        sound = sound.copy()
        sound[:L] *= fade_in  # Apply fade-in
        sound[-L:] *= fade_out  # Apply fade-out

        return sound

    def sine_ramp(self, sound):
        L = int(self.tau * self.sample_rate)
        t = np.linspace(0, L / self.sample_rate, L)
        sine_window = np.sin(np.pi * t / (2 * self.tau)) ** 2  # Sine fade-in

        sound = sound.copy()
        sound[:L] *= sine_window  # Apply fade-in
        sound[-L:] *= sine_window[::-1]  # Apply fade-out

        return sound

    def gaussian_ramp(self, sound):
        L = int(self.tau * self.sample_rate)
        t = np.linspace(-L / self.sample_rate, L / self.sample_rate, 2 * L)
        gaussian = np.exp(-t ** 2 / (2 * (self.tau / 2.0) ** 2))  # Gaussian window
        sound = sound.copy()
        sound[:L] *= gaussian[:L]  # Apply fade-in
        sound[-L:] *= gaussian[L:]  # Apply fade-out

        return sound

    def generate_sequence(self, freq, num_harmonics, tone_duration, harmonic_factor, total_duration, isi):

        # Generate the tone using the sound_maker method
        sound = self.sound_maker(freq,num_harmonics, tone_duration, harmonic_factor)

        # Normalize the sound
        #normalized_sound = self.normalize_sound(sound)

        normalized_sound = thorns.waves.set_dbspl(sound, 70)

        # Apply ramping to the normalized sound
        #ramped_sound = self.ramp_in_out(normalized_sound)

        # Exponential ramp
        #ramped_sound = self.exponential_ramp(normalized_sound)

        # Sine ramp
        ramped_sound = self.sine_ramp(normalized_sound)

        # Gaussian ramp
        #ramped_sound = self.gaussian_ramp(normalized_sound)

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
