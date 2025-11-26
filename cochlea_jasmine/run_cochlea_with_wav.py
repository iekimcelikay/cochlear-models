import numpy as np
import thorns as th
from thorns import waves as wv
import cochlea
import soundfile as sf
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
from cochlea.zilany2014 import calc_cfs

def cochlea_mean_rate(peripheral_fs, sound, min_cf, max_cf, num_cf, randnoise=True):
    """

    :param peripheral_fs:
    :param sound:
    :param num_cf:
    :param randnoise: set this to False if you don't want to apply a gaussian noise.
    :return:
    """
    auditory_nerve_rate_ts = cochlea.run_zilany2014_rate(sound, peripheral_fs,
                                                         anf_types=('lsr', 'msr', 'hsr'),
                                                         cf=(min_cf, max_cf, num_cf),
                                                         species='human',
                                                         cohc=1,
                                                         cihc=1,
                                                         powerlaw='actual',
                                                         ffGn=randnoise)
    return auditory_nerve_rate_ts

# MODEL PARAMETERS
peripheral_fs = 100e3
min_cf = 125
max_cf = 2500
num_cf = 30

# SOUND FILE PARAMETERS
root_dir = '//subcorticalSTRF/cochlea_jasmine/'
# You can loop this.
sound_number = 3
sound_type = 'animal'
type_number = 1
type_id = 10
sound_name = "s{}_{}_{}_ramp{}".format(sound_number, sound_type, type_number, type_id)
sound_filetype = ".wav"
sound_file_path = root_dir + sound_name + sound_filetype


# Check if the sound_file_path is correct
try:
    os.path.isfile(sound_file_path)==True
except:
    print("File does not exist")
    exit()


# Load the sound file as ndarray
# fs is sample size. also length(audio) will be = fs. But sf.read requires fs to be an output
audio, fs = sf.read(sound_file_path)


# WE NEED TO CONVERT THE SAMPLE SIZE TO 100e3 (minimum sample rate cochlea accepts)
sound = wv.resample(audio, fs, peripheral_fs)

# Get the rate level functions
rates = cochlea_mean_rate(peripheral_fs, sound, min_cf, max_cf, num_cf
                          , randnoise=True)

rf = np.zeros(num_cf)  # Shape: (channel, dB, freq)

rates_f = pd.DataFrame(rates)
rates_dfname = "rates_{}.csv".format(sound_name)
rates_f.to_csv(rates_dfname, index=False)
# Compute time axis
num_time_points = rates.shape[1]
duration_sec = num_time_points / peripheral_fs
time_sec = np.linspace(0, duration_sec, num_time_points)

### Plot rates
fig, ax = plt.subplots()
img = ax.imshow(
    rates.T,
    aspect='auto',
    origin='lower',
    extent=[0, duration_sec, 0, num_cf]
 )

# Define cfs (for t ticks)
cfs = calc_cfs((125,2500,30), 'human')

# Customize axis ticks
num_time_ticks = 6
time_tick_positions = np.linspace(0, duration_sec, num_time_ticks)
ax.set_xticks(time_tick_positions)
ax.set_xlabel("Time (s)")


num_cf_ticks = 30
cf_tick_indices = np.linspace(0, num_cf -1, num_cf_ticks).astype(int)
ax.set_yticks(cf_tick_indices)
ax.set_yticklabels(["{:.0f} Hz".format(cfs[i]) for i in cf_tick_indices])
ax.set_ylabel("Characteristic Frequency (Hz)")

subtitle = "soundfile: {}".format(sound_name)
fig.suptitle(subtitle)

plt.colorbar(img, ax=ax, label="Firing Rate (spikes/s)")
plt.show()


rates_fname = "rates_{}.png".format(sound_name)
fig.savefig(rates_fname)

mean_rates = rates.mean(axis=0)  # shape: (num_cf,)
for channel in range(num_cf):
    rf[channel] = mean_rates[channel]

# Debug point
print('stop')
