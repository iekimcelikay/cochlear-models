# sound sampling rate: 44100 (Hz) or 44.1 kHZ
import numpy as np
import matplotlib.pyplot as plt

# Parameters
a1 = 1.0       # Amplitude of first sine wave

a2 = 0.75       # Amplitude of second sine wave
w1 = 2 * np.pi * 1  # Frequency of first sine wave (1 Hz)
w2 = 2 * np.pi * 2  # Frequency of second sine wave (5 Hz)

# Time array
t = np.linspace(0, 2, 1000)  # 2 seconds, 1000 points

# Signal
s = a1 * np.sin(w1 * t) + a2 * np.sin(w2 * t)

# Plotting
plt.figure(figsize=(10, 4))
plt.plot(t, s, label='s(t) = a1·sin(w1·t) + a2·sin(w2·t)')
plt.xlabel('Time (t)')
plt.ylabel('s(t)')
plt.title('Sum of Two Sine Waves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
