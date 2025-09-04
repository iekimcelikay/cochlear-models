# Will take a sound file (np.array) as an input.

import thorns
from matplotlib import pyplot as plt
from thorns import spikes

import numpy as np
import subcorticalSTRF.stim_generator.soundgen as soundgen
from subcorticalSTRF.bruce2018_python.helperfunctions.simulator_utils import greenwood_human

sample_rate = 100e3
tau = 0.01
sound_gen = soundgen.SoundGen(sample_rate, tau)

## Now the time to simulate the RECEPTIVE FIELDS.
# obviously that requires first ->
# creating different stimuli with different loudness & different frequencies.

# Let's say 30 frequencies between 125 - 2500, in equal steps.
# TODO: Change the frequency creation method, it's not through geomspace.

desired_dbs = np.arange(0, 110, 5) # Create an array (start, end, in_steps)
peripheral_fs = 100e3
num_cf = 30
desired_frequencies = greenwood_human((125, 2500, num_cf)) # Create an array (start, end, total_number)
subcortical_fs = 100e3
num_harmonics = 1
tone_duration = 0.200
harmonic_factor = 0.5


step = 'spike trains'
#step = 'rates'

all_params_list = []
all_tones_list = []


rf = np.zeros((num_cf, len(desired_dbs), len(desired_frequencies)))  # Shape: (channel, dB, freq)


for i_dbs in range(len(desired_dbs)):
    for i_freq in range(len(desired_frequencies)):
        all_params_list.append((desired_dbs[i_dbs], desired_frequencies[i_freq]))
        freq = desired_frequencies[i_freq]
        db = desired_dbs[i_dbs]
        my_tone = sound_gen.sound_maker(freq, num_harmonics, tone_duration, harmonic_factor, db)
        all_tones_list.append(my_tone)
        if step == 'spike trains':
            trains = sound_gen.peripheral_simulator(my_tone, peripheral_fs, num_cf)
            # Accumulate the spike trains
            anf_acc = thorns.accumulate(trains, keep=['cf', 'duration'])
            anf_acc.sort_values('cf', ascending=False, inplace=True)
            # Plot the auditory nerve response
            fig, ax = plt.subplots(2, 1) # ax = matplotlib.pyplot
            subtitle = "freq= %.3f , loudness = %d" % (freq, db)
            fig.suptitle(subtitle)
            thorns.plot_signal(
                signal=my_tone,
                fs=sample_rate,
                ax=ax[0]
            )
            # Plot manually, using the same logic of thorns.plot_neurogram
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

            # thorns.plot_neurogram(anf_acc, peripheral_fs, ax=ax[1]) The upper code replaces this so to allow manual calibration of parameters such as vmin and vmax.
            plt.show()
            if num_harmonics == 1:
                fname = "puretone_spike_freq=%.3f_loudness=%d.png" % (freq, db)
            else:
                fname = "complextone_spike_freq=%.3f_loudness=%d.png" % (freq, db)
            fig.savefig(fname)
        elif step == 'rates':
            rates = sound_gen.peripheral_sim_rate(my_tone, peripheral_fs, num_cf, subcortical_fs)
            #rates_f = pd.DataFrame(rates)
            #rates_dfname = "rates_freq=%.3f_loudness=%d" % (freq, db)
            #rates_f.to_csv(rates_dfname, index=False)

            ### Plot rates
            #fig, ax = plt.subplots()
            #img = ax.imshow(
            #    rates.T,
            #    aspect='auto'
            #)
            #subtitle = "freq= %.3f , loudness = %d" % (freq, db)
            #fig.suptitle(subtitle)
            #plt.colorbar(img)
            #plt.show()
            #rates_fname = "rates_freq=%.3f_loudness=%d.png" % (freq, db)
            #fig.savefig(rates_fname)

            mean_rates = rates.mean(axis=0)  # shape: (num_cf,)
            for channel in range(num_cf):
                rf[channel, i_dbs, i_freq] = mean_rates[channel]


#
# if num_harmonics == 1:
#     np.savez('puretone_rf_data.npz', rf=rf, desired_frequencies=desired_frequencies, desired_dbs=desired_dbs,
#     cf_list=cf_list)
#     print("Pure tones RF data saved to rf_data.npz")
# else:
#     np.savez('hcomplextone_rf_data.npz', rf=rf, desired_frequencies=desired_frequencies, desired_dbs=desired_dbs,
#      cf_list=cf_list)
#     print("Harmonic complex tones RF data saved to rf_data.npz")
#
#
#

