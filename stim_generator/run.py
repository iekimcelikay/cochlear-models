import thorns.waves

from soundgenDEP import SoundGen
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write

# Initialize the generator
sample_rate = 44100
tau = 0.01
sound_gen = SoundGen(sample_rate, tau)

# Create a single sound first (to check the ramping)
tone = sound_gen.sound_maker(100, 4, 0.150, 0.5)
normalized_tone = thorns.waves.set_dbspl(tone, 50)
ramped_tone = sound_gen.ramp_in_out(normalized_tone)

write("generated_tone.wav", sample_rate, ramped_tone.astype(np.float32))  # Save as 32-bit floating point

# Time vector in milliseconds
t_ms = np.linspace(0, len(normalized_tone) / sample_rate * 1000, len(normalized_tone))

# Plot both waveforms
plt.figure(figsize=(10, 4))
plt.plot(t_ms, normalized_tone, label="Original (Normalized)", linestyle='--', alpha=0.6, color='gray')
plt.plot(t_ms, ramped_tone, label="Ramped Tone", color='blue')

plt.title("Original vs. Ramped Tone Waveform")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

sd.play(ramped_tone, samplerate=sample_rate )
sd.wait()

#
# # List of tuples: (freq, num_harmonics, tone_duration, harmonic_factor, total_duration, isi)
# sequence_params = [
#     (100, 4, 0.133, 0.5, 5, 0.033), # Condition 1
#     (100, 4, 0.133, 0.5, 5, 0.200), # Condition 2
#     (100, 4, 0.133, 0.5, 5, 0.867), # Condition 3
#     (100, 4, 0.033, 0.5, 5, 0.133), # Condition 4 -- this almost always clicks
#     (100, 4, 0.200, 0.5, 5, 0.133), # Condition 5
#     (100, 4, 0.867, 0.5, 5, 0.133), # Condition 6
#     (100, 4, 0.267, 0.5, 5, 0.067), # Condition 7
#     (100, 4, 0.800, 0.5, 5, 0.200), # Condition 8
#     (100, 4, 5.000, 0.1, 5, 0.000), # Condition 9
# ]
#
# # Generate all sequences and store them in a list
# sequences = []
#
# for params in sequence_params:
#     sequence = sound_gen.generate_sequence(*params)
#     sequences.append(sequence)
#
# # Sequences is a list of 9 numpy arrays
# sd.play(sequences[0], samplerate=sample_rate )
# sd.wait()

# Parameters from Condition 1
params =  (100, 4, 0.200, 0.5, 5, 0.133)  # (freq, num_harmonics, tone_duration, harmonic_factor, total_duration, isi)
sequence = sound_gen.generate_sequence(*params)

# Create time axis in milliseconds
t_ms_seq = np.linspace(0, len(sequence) / sample_rate * 1000, len(sequence))

# Plot the full sequence
plt.figure(figsize=(12, 4))
plt.plot(t_ms_seq, sequence, color='darkgreen')
plt.title("Full 5-Second Harmonic Tone Sequence with Ramping and ISI")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()


sd.play(sequence, samplerate=sample_rate )
sd.wait()

