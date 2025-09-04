# Will take a sound file (np.array) as an input.
import pandas as pd
import thorns
from matplotlib import pyplot as plt
from thorns import spikes
# So I need to call the `soundgen` module.
import numpy as np
import subcorticalSTRF.stim_generator.soundgen as soundgen
import thorns.waves as wv

sample_rate = 44100
tau = 0.01
sound_gen = soundgen.SoundGen(sample_rate, tau)

sequence_params = [
    (400, 4, 0.050, 0.5, 70, 2, 0.033), # Condition 1
    (400, 4, 0.133, 0.5, 70, 2, 0.200), # Condition 2
    (400, 4, 0.133, 0.5, 70, 2, 0.867), # Condition 3
    (400, 4, 0.033, 0.5, 70, 2, 0.133), # Condition 4 -- this almost always clicks
    (400, 4, 0.200, 0.5, 70, 2, 0.133), # Condition 5
    (400, 4, 0.867, 0.5, 70, 2, 0.133), # Condition 6
    (400, 4, 0.267, 0.5, 70, 2, 0.067), # Condition 7
    (400, 4, 0.800, 0.5, 70, 2, 0.200), # Condition 8
    (400, 4, 2.000, 0.5, 70, 2, 0.000), # Condition 9

]

peripheral_fs = 100e3
num_cf = 100

sequence = sound_gen.generate_sequence(*sequence_params[8])
trains = sound_gen.peripheral_simulator(sequence, peripheral_fs, num_cf)

# trains_f = pd.DataFrame(trains)
# trains_f.to_csv('cond9spiketrains.csv', index=False)
thorns.plot_raster(thorns.accumulate(trains))


# Accumulate the spike trains
anf_acc = thorns.accumulate(trains, keep=['cf', 'duration'])
anf_acc.sort_values('cf', ascending=False, inplace=True)

# Plot the auditory nerve response

fig, ax = plt.subplots(2,1)
thorns.plot_signal(
    signal = sequence,
    fs= sample_rate,
    ax=ax[0]
)

### Plot manually, using the same logic of thorns.plot_neurogram
neurogram = spikes.trains_to_array(
    anf_acc,
    fs=peripheral_fs,
)

extent = (
    0,  # left
    neurogram.shape[0] / peripheral_fs,  # right
    0,  # bottom
    neurogram.shape[1]  # top
)
ax[1].imshow(neurogram.T, aspect='auto', extent=extent, vmin=0, vmax=1)
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Channel number")

#thorns.plot_neurogram(anf_acc, peripheral_fs, ax=ax[1])
plt.show()

## Now the time to simulate the RECEPTIVE FIELDS.
# obviously that requires first ->
# creating different stimuli with different loudness & different frequencies.

# Let's say 30 frequencies between 125 - 2500, in equal steps.
#
desired_frequencies = np.linspace(125, 2500, 30) # Create an array (start, end, total_number)
desired_dbs = np.arange(50, 90, 5) # Create an array (start, end, in_steps)

import numpy as np
print(desired_frequencies)
print(desired_dbs)

all_params_list = []

for i_dbs in range(len(desired_dbs)):
    for i_freq in range(len(desired_frequencies)):
        all_params_list.append((desired_dbs[i_dbs], desired_frequencies[i_freq]))
        freq = desired_frequencies[i_freq]
        db = desired_dbs[i_dbs]
        my_tone = sound_gen.sound_maker(freq, 4, 0.200, 0.5, db)


print(all_params_list)


