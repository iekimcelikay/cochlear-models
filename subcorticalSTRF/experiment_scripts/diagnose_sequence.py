import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from subcorticalSTRF.stim_generator.soundgen import SoundGen

# Create sound generator
sound_maker = SoundGen(sample_rate=44100, tau=0.005)

# Generate sequences with same parameters as your original script
print("Generating sequence with 1 harmonic...")
sequence_1h, frequencies = sound_maker.generate_sequence_gaussian_freq(
    freq_mean=440, freq_std=50, num_harmonics=1,
    tone_duration=0.2, harmonic_factor=0.7, 
    total_duration=5.0, isi=0.133, 
    freq_min=400, freq_max=600, seed=43, stereo=True
)

print("Generating sequence with 7 harmonics...")
sequence_7h, frequencies2 = sound_maker.generate_sequence_gaussian_freq(
    freq_mean=440, freq_std=50, num_harmonics=7,
    tone_duration=0.2, harmonic_factor=0.7, 
    total_duration=5.0, isi=0.133, 
    freq_min=400, freq_max=600, seed=43, stereo=True
)

# Extract just the first tone from each sequence to compare
tone_samples = int(0.2 * 44100)  # 200ms
first_tone_1h = sequence_1h[:tone_samples, 0]  # First channel, first tone
first_tone_7h = sequence_7h[:tone_samples, 0]  # First channel, first tone

# Analyze with FFT
fft_1h = np.abs(np.fft.rfft(first_tone_1h))
fft_7h = np.abs(np.fft.rfft(first_tone_7h))
freqs = np.fft.rfftfreq(len(first_tone_1h), 1/44100)

# Find peaks (harmonics)
threshold = np.max(fft_1h) * 0.1
peaks_1h = freqs[fft_1h > threshold]
peaks_7h = freqs[fft_7h > threshold]

print(f"\nFirst tone frequency: {frequencies[0]:.1f} Hz")
print(f"\nSequence with 1 harmonic:")
print(f"  RMS amplitude: {np.sqrt(np.mean(first_tone_1h**2)):.4f}")
print(f"  Max amplitude: {np.max(np.abs(first_tone_1h)):.4f}")
print(f"  Detected peaks: {len(peaks_1h[peaks_1h < 3000])}")

print(f"\nSequence with 7 harmonics:")
print(f"  RMS amplitude: {np.sqrt(np.mean(first_tone_7h**2)):.4f}")
print(f"  Max amplitude: {np.max(np.abs(first_tone_7h)):.4f}")
print(f"  Detected peaks: {len(peaks_7h[peaks_7h < 3000])}")

if len(peaks_7h) > len(peaks_1h):
    print("\n✓ HARMONICS ARE WORKING - 7-harmonic version has more frequency peaks")
else:
    print("\n✗ PROBLEM - Both have same number of peaks!")
