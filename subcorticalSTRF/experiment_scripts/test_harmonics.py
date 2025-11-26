import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import matplotlib.pyplot as plt
from subcorticalSTRF.stim_generator.soundgen import SoundGen

# Create sound generator
sound_maker = SoundGen(sample_rate=44100, tau=0.005)

# Generate two sounds: 1 harmonic vs 7 harmonics
tone_1_harmonic = sound_maker.sound_maker(freq=440, num_harmonics=1, 
                                          tone_duration=0.2, harmonic_factor=0.7)
tone_7_harmonics = sound_maker.sound_maker(freq=440, num_harmonics=7, 
                                           tone_duration=0.2, harmonic_factor=0.7)

# Plot the first 1000 samples to visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Time domain
time = np.linspace(0, 0.01, 441)  # First 10ms
axes[0, 0].plot(time, tone_1_harmonic[:441])
axes[0, 0].set_title('1 Harmonic - Time Domain')
axes[0, 0].set_xlabel('Time (s)')

axes[0, 1].plot(time, tone_7_harmonics[:441])
axes[0, 1].set_title('7 Harmonics - Time Domain')
axes[0, 1].set_xlabel('Time (s)')

# Frequency domain (FFT)
fft_1 = np.abs(np.fft.rfft(tone_1_harmonic))
fft_7 = np.abs(np.fft.rfft(tone_7_harmonics))
freqs = np.fft.rfftfreq(len(tone_1_harmonic), 1/44100)

axes[1, 0].plot(freqs[:2000], fft_1[:2000])
axes[1, 0].set_title('1 Harmonic - Frequency Domain')
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].axvline(440, color='r', linestyle='--', label='Fundamental')
axes[1, 0].legend()

axes[1, 1].plot(freqs[:2000], fft_7[:2000])
axes[1, 1].set_title('7 Harmonics - Frequency Domain')
axes[1, 1].set_xlabel('Frequency (Hz)')
for k in range(1, 8):
    axes[1, 1].axvline(440*k, color='r', linestyle='--', alpha=0.5)
axes[1, 1].legend(['Signal', 'Harmonics'])

plt.tight_layout()
plt.savefig('/home/ekim/PycharmProjects/phd_firstyear/subcorticalSTRF/experiment_scripts/harmonic_comparison.png')
print("Plot saved to harmonic_comparison.png")

# Print some stats
print(f"\n1 Harmonic - RMS: {np.sqrt(np.mean(tone_1_harmonic**2)):.4f}")
print(f"7 Harmonics - RMS: {np.sqrt(np.mean(tone_7_harmonics**2)):.4f}")
print(f"\nMax amplitude 1 harmonic: {np.max(np.abs(tone_1_harmonic)):.4f}")
print(f"Max amplitude 7 harmonics: {np.max(np.abs(tone_7_harmonics)):.4f}")
