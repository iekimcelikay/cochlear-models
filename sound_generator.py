"""
sound_generator.py

This script contains two main classes:
- CreateSound: for generating harmonic tones and chords
- SoundSequence: for assembling sequences of tones with silence intervals

Usage:
    Create a CreateSound instance with frequency/frequencies and tone settings,
    then pass it into a SoundSequence to build and play a tone sequence.

Dependencies:
    - numpy
    - sounddevice

Author: [Ekim Celikay]
Date: [2025-04-18]
"""

import numpy as np
import sounddevice as sd
import math
import thorns



class CreateSound:
    """
    Generates harmonic complex tones based on a base frequency and configurable harmonic structure.

    This class creates individual tones composed of multiple harmonics, applies amplitude decay,
    adds smooth onset/offset ramps to reduce clicks, and provides normalization to prevent clipping.
    adds smooth onset/offset ramps to reduce clicks, and provides normalization to prevent clipping.

    Attributes:
        freq (float): Base (fundamental) frequency in Hz.
        num_harmonics (int): Number of harmonic overtones to include.
        tone_duration (float): Duration of each generated tone in seconds.
        sample_rate (int): Number of samples per second (Hz).
        harmonic_factor (float): Amplitude decay factor for successive harmonics.

    Methods:
        create_hct(): Generates a harmonic complex tone (HCT) as a NumPy array.
        ramp_in_out(sound, tau=0.0025): Applies a smooth ramp to the beginning and end of a tone.
        normalize(sound): Normalizes the audio to prevent clipping (values exceeding [-1, 1]).
    """

    def __init__(self, freq: float, num_harmonics: int, tone_duration: float, sample_rate: int, harmonic_factor: float):
        """
        Initialize the CreateSound instance.

        :param freq: Base frequency in Hz.
        :param num_harmonics: Number of harmonic tones.
        :param tone_duration: Duration of each tone in seconds.
        :param sample_rate: Sample rate of sounds ( per second).
        :param harmonic_factor: Harmonic amplitude decay factor for each tone.
        """
        self.freq= freq # Hz
        self.num_harmonics = num_harmonics # Number of harmonic factors, i.e., 4
        self.tone_duration = tone_duration # Seconds
        self.sample_rate = sample_rate # How many sample points? i.e., 44100
        self.harmonic_factor = harmonic_factor # Amplitude decay factor per harmonic

    def create_hct(self) -> np.ndarray:
        """
        Generate a harmonic complex tone (HCT) composed of multiple harmonics.

        Each harmonic is a sine wave at a multiple of the base frequency,
        with its amplitude scaled by a harmonic decay factor.

        Returns:
            np.ndarray: Array of audio samples representing the harmonic complex tone.
        """
        freq = self.freq
        # Create the time array
        t = np.linspace(0, self.tone_duration, int(self.sample_rate * self.tone_duration), endpoint=False)

        # Initialize the sound array
        sound = np.zeros_like(t) #or np.zeros(int(self.duration * self.sample_rate))

        # Generate harmonics
        for k in range(1, self.num_harmonics + 1):
            omega = 2 * np.pi * freq * k # (k)-th harmonic
            harmonic = np.sin(omega * t)
            amplitude = (self.harmonic_factor ** (k - 1)) / self.num_harmonics # Decay the amplitude
            sound = sound + amplitude * harmonic

        return sound

    def ramp_in_out(self, sound, tau=0.003) :
        """
        Apply a smooth ramp to a **copy** of the signal.

        Apply a smooth ramp (fade in and out) to the beginning and end of a tone.

        This helps prevent clicking artifacts caused by abrupt amplitude changes.

        Args:
            sound (np.ndarray): Audio signal to apply the ramp to.
            tau (float): Duration of the ramp (in seconds). Default is 0.0025.

        Returns:
            np.ndarray: Audio signal with ramps applied.
        """
        sound = sound.copy()  # 🛠️ Make a copy to avoid modifying in-place
        L = int(tau * self.sample_rate) # Length of ramping window.
        hw= np.hamming(2 * L)
        sound[:L] *= hw[:L]
        sound[-L:] *= hw[L:]

        return sound

    def normalize(self, sound): ## This was suggested by chatgpt, I haven't verified if it does what I need (it's functioning tho)
        """
        Normalize the amplitude of the audio signal to avoid clipping.

        If the absolute maximum value exceeds 1, the entire signal is scaled down
        to ensure it fits within the range [-1, 1].

        Args:
            sound (np.ndarray): Audio signal to normalize.

        Returns:
            np.ndarray: Normalized audio signal.
        """
        ### do the normalizing as the mean of the square , normalixe by power. this should be smaller than 1, if it stoo quiet you would neet to be multiply by 1.

        max_val = np.max(np.abs(sound))
        if max_val > 1:
            sound /= max_val  # Normalize to the range [-1, 1]
        return sound

#    def play_sound(self, sound):
#        sd.play(sound, self.sample_rate)
#        sd.wait()

class SoundSequence:
    """
    Assembles and generates a sequence of harmonic tones with inter-stimulus intervals (ISI).

    Attributes:
        sound_object (CreateSound): An instance of CreateSound used to generate individual tones.
        total_duration (float): Total duration of the entire sequence in seconds.
        isi (float): Inter-stimulus interval (duration of silence) between tones in seconds.
        num_tones (int): Number of tones that will fit within the total duration, accounting for ISI.

    Methods:
        calculate_num_tones(): Calculates how many tones can fit into the total duration.
        generate_sequence(): Generates the full sequence of tones with ISIs as silence between tones.
    """
    def __init__(self, sound_object, total_duration: float, isi: float):
        """
        Initializes a SoundSequence instance for constructing a sequence of tones with silence intervals.

        Args:
            sound_object (CreateSound): Instance of CreateSound used to generate tones.
            total_duration (float): Total duration of the tone sequence in seconds.
            isi (float): Inter-stimulus interval (silence) between tones, in seconds.
        """
        self.sound_object = sound_object
        self.total_duration = total_duration
        self.isi = isi
        self.num_tones = self.calculate_num_tones() # Calculate the number of tones in the sequence

    def calculate_num_tones(self):
        """
        Calculate how many tones can fit into the total duration,
        including the inter-stimulus intervals (ISI).

        Returns:
            int: Number of tones that can be played within the total duration.
        """
        duration_of_one_tone = self.sound_object.tone_duration
        num_tones = int(self.total_duration // (duration_of_one_tone + self.isi))
        return num_tones

    def generate_sequence(self):
        """
        Generate a complete audio sequence by alternating tones and silent intervals.

        For each tone, it generates the harmonic sound, applies a ramp,
        appends it to the sequence, and adds a silent interval (ISI) after it.

        Returns:
            np.ndarray: Audio sequence ready for playback.
        """
        sequence = np.array([])

        # Generate each tone and concatenate them with ISI gaps.

        for _ in range(self.num_tones):
            stim = self.sound_object.create_hct() # Generate a single tone
            tone = self.sound_object.ramp_in_out(stim)
            sequence = np.concatenate((sequence, tone)) # Add the tone to the sequence

            # Add ISI (silent gap) between tones
            isi_samples = int(self.isi * self.sound_object.sample_rate) # We need to
            sequence = np.concatenate((sequence, np.zeros(isi_samples))) # Add ISI gap


        total_samples = int(self.total_duration * self.sound_object.sample_rate)
        if len(sequence) > total_samples:
            sequence = sequence[:total_samples]

        return sequence
#----------------------------------------------------------------------------------------------

# Create the sound object

## Condition 1:
sound1 = CreateSound(
    freq=100,               # Base/Fundamental frequency in Hz
    num_harmonics=4,        # Number of harmonics (1st to 4th)
    tone_duration=0.133,      # Duration in seconds
    sample_rate=44100,      # Sampling rate (Hz)
    harmonic_factor= 1     # Harmonic decay factor
)

# Create a sound sequence with ISI of 0.2 seconds and a total duration of 5 seconds
cond1 = SoundSequence(sound_object=sound1, total_duration=5.0, isi= 0.033)

# Generate the sound sequence
tone_sequence = cond1.generate_sequence()

# Normalize the sequence to avoid clipping ## I don't think this changes anything.
tone_sequence = sound1.normalize(tone_sequence)

# Play the generated sound sequence
sd.play(tone_sequence, sound1.sample_rate)
sd.wait()  # Wait until the sound has finished playing

## CONDITION 2: on:133 ms, isi:200 ms
sound2 = CreateSound(
    freq=100,               # Base/Fundamental frequency in Hz
    num_harmonics=4,        # Number of harmonics (1st to 4th)
    tone_duration=0.133,      # Duration in seconds
    sample_rate=44100,      # Sampling rate (Hz)
    harmonic_factor= 1     # Harmonic decay factor
)

cond2 = SoundSequence(sound_object=sound2, total_duration=5.0, isi= 0.200)
tone_sequence2 = cond2.generate_sequence()
#sd.play(tone_sequence2, sound2.sample_rate)
#sd.wait()

## CONDITION 3: on:133 ms, isi: 867 ms
sound3 = CreateSound(
    freq=100,               # Base/Fundamental frequency in Hz
    num_harmonics=4,        # Number of harmonics (1st to 4th)
    tone_duration=0.133,      # Duration in seconds
    sample_rate=44100,      # Sampling rate (Hz)
    harmonic_factor= 1     # Harmonic decay factor
)

cond3 = SoundSequence(sound_object=sound3, total_duration=5.0, isi= 0.867)
tone_sequence3 = cond3.generate_sequence()
#sd.play(tone_sequence3, sound3.sample_rate)
#sd.wait()

## CONDITION 4: 33 ms, isi: 133 ms
sound4 = CreateSound(
    freq=100,               # Base/Fundamental frequency in Hz
    num_harmonics=4,        # Number of harmonics (1st to 4th)
    tone_duration=0.033,      # Duration in seconds
    sample_rate=44100,      # Sampling rate (Hz)
    harmonic_factor= 1     # Harmonic decay factor
)

cond4 = SoundSequence(sound_object=sound4, total_duration=5.0, isi= 0.133)
tone_sequence4 = cond4.generate_sequence()
#sd.play(tone_sequence4, sound4.sample_rate)
#sd.wait()

## CONDITION 5: 200 ms, isi: 133 ms
sound5 = CreateSound(
    freq=100,               # Base/Fundamental frequency in Hz
    num_harmonics=4,        # Number of harmonics (1st to 4th)
    tone_duration=0.200,      # Duration in seconds
    sample_rate=44100,      # Sampling rate (Hz)
    harmonic_factor= 1     # Harmonic decay factor
)

cond5 = SoundSequence(sound_object=sound5, total_duration=5.0, isi= 0.133)
tone_sequence5 = cond5.generate_sequence()
#sd.play(tone_sequence5, sound5.sample_rate)
#sd.wait()

## CONDITION 6: On: 867 ms, isi: 133 ms
sound6 = CreateSound(
    freq=100,               # Base/Fundamental frequency in Hz
    num_harmonics=4,        # Number of harmonics (1st to 4th)
    tone_duration=0.867,      # Duration in seconds
    sample_rate=44100,      # Sampling rate (Hz)
    harmonic_factor= 1     # Harmonic decay factor
)

cond6 = SoundSequence(sound_object=sound6, total_duration=5.0, isi= 0.133)
tone_sequence6 = cond6.generate_sequence()
#sd.play(tone_sequence6, sound6.sample_rate)
#sd.wait()

## CONDITION 7: on: 267 ms, isi: 67 ms
sound7 = CreateSound(
    freq=100,               # Base/Fundamental frequency in Hz
    num_harmonics=4,        # Number of harmonics (1st to 4th)
    tone_duration=0.267,      # Duration in seconds
    sample_rate=44100,      # Sampling rate (Hz)
    harmonic_factor= 1     # Harmonic decay factor
)

cond7 = SoundSequence(sound_object=sound7, total_duration=5.0, isi= 0.067)
tone_sequence7 = cond7.generate_sequence()
#sd.play(tone_sequence7, sound7.sample_rate)
#sd.wait()

## CONDITION 8: on: 800 ms, isi: 200 ms
sound8 = CreateSound(
    freq=100,               # Base/Fundamental frequency in Hz
    num_harmonics=4,        # Number of harmonics (1st to 4th)
    tone_duration=0.800,      # Duration in seconds
    sample_rate=44100,      # Sampling rate (Hz)
    harmonic_factor= 0.5     # Harmonic decay factor
)

cond8 = SoundSequence(sound_object=sound8, total_duration=5.0, isi= 0.200)
tone_sequence8 = cond8.generate_sequence()
# sd.play(tone_sequence8, sound8.sample_rate)
# sd.wait()

# CONDITION 9: On: 5000 ms, isi:0 ms
sound9 = CreateSound(
    freq=100,               # Base/Fundamental frequency in Hz
    num_harmonics=4,        # Number of harmonics (1st to 4th)
    tone_duration=5.0,      # Duration in seconds
    sample_rate=44100,      # Sampling rate (Hz)
    harmonic_factor= 1     # Harmonic decay factor
)

cond9 = SoundSequence(sound_object=sound9, total_duration=5.0, isi= 0)
tone_sequence9 = cond9.generate_sequence()
#sd.play(tone_sequence9, sound9.sample_rate)
#sd.wait()


# KIm2024 sequences:
# condition 1: 133 ms, isi: 33 ms
# condition 2: 133 ms, isi: 200 ms
# condition 3: 133 ms, isi: 867 ms
# condition 4: 33 ms, isi: 133 ms
# condition 5: 200 ms, isi: 133 ms
# condition 6: 867 ms, isi: 133 ms
# condition 7: 267 ms, isi: 67 ms
# condition 8: 800 ms, isi: 200 ms
# condition 9: 5000 ms, isi: 0 ms