# deprecated file path: cochlea_meanrate/cochlea_meanrate_function.py
# deprecated file
# TODO: remove this file in future releases
# TODO: 1. cochlea config dataclass
# TODO: 2. Main simulation class
import thorns as th
import numpy as np
import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import math
from cochlea.zilany2014 import calc_cfs
import sys
import time
import gc
from stim_generator.soundgen import SoundGen
from scipy.io import savemat
# from scipy.signal import decimate


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ----------- Parameter Setup -----------

peripheral_fs = 100e3
min_cf = 125
max_cf = 2500
num_cf = 10
fs_target = 1000
num_ANF = (128, 128, 128)
num_runs = 1
downsample_rate = int(round(peripheral_fs / fs_target))

num_harmonics = 1
duration = 0.200
harmonic_factor = 0.5
sample_rate = 100e3
tau_ramp = 0.01
num_tones = 20
freq_range = (125, 2500, num_tones)
db_range = (50, 90, 10)


def generate_stimuli_params(freq_range, num_cf, db_range):
    desired_dbs = np.arange(*db_range)
    desired_freqs = calc_cfs(freq_range, species='human')
    return desired_dbs, desired_freqs


def generate_ramped_tone(sound_gen, freq, num_harmonics, duration, harmonic_factor, db):
    tone = sound_gen.sound_maker(freq, num_harmonics, duration, harmonic_factor, db)
    return sound_gen.sine_ramp(tone)

# ----------- One Run Simulation Function -----------


def simulate_one_run(run):
    print("Running simulation for run {}".format(run))
    # ----------- Generate a fully unique seed per run -----------
    unique_seed = (int(time.time() * 1e6) + os.getpid() + run) % (2**32)
    np.random.seed(unique_seed)
    print("Unique seed is {}".format(unique_seed))

    all_meta_rows = []
    all_spikes_list = []

    sound_gen = SoundGen(sample_rate, tau_ramp)
    desired_dbs, desired_freqs = generate_stimuli_params(freq_range, num_cf, db_range)

    for db in desired_dbs:
        for freq in desired_freqs:
            print("Generating tone at {:.1f} Hz and {:d} dB SPL".format(freq, int(db)))

            my_tone = generate_ramped_tone(sound_gen, freq, num_harmonics, duration, harmonic_factor, db)
            trains_acc = sound_gen.cochlea_rate_manual(my_tone, peripheral_fs, min_cf, max_cf,  num_cf, num_ANF, seed=unique_seed)
            trains_acc.sort_values(['cf', 'type'], ascending=[True, True], inplace=True)

            signal_array = th.trains_to_array(trains_acc, peripheral_fs).T
            n_fibers = len(trains_acc)

            for i_fiber in range(n_fibers):
                signal = signal_array[i_fiber]
                signal = signal.astype(np.float32)

                meta = {
                    'cf': trains_acc.iloc[i_fiber]['cf'],
                    'duration': trains_acc.iloc[i_fiber]['duration'],
                    'fiber_type': trains_acc.iloc[i_fiber]['type'],
                    'tone_freq': freq,
                    'db': db,
                    'run': run
                }
                all_meta_rows.append(meta)
                all_spikes_list.append(signal)

            del trains_acc, signal_array, signal, my_tone
            gc.collect()

    # ----------- Save Output -----------
    n = len(all_meta_rows)

    cf_array = np.array([meta['cf'] for meta in all_meta_rows])
    duration_array = np.array([meta['duration'] for meta in all_meta_rows])
    anf_array = np.array([meta['fiber_type'] for meta in all_meta_rows])
    tone_freq_array = np.array([meta['tone_freq'] for meta in all_meta_rows])
    db_array = np.array([meta['db'] for meta in all_meta_rows])
    spikes_array = np.empty(n, dtype=object)
    run_array = np.array([meta['run'] for meta in all_meta_rows])

    for i in range(n):
        spikes_array[i] = all_spikes_list[i]

    mat_data = {
        'cf': cf_array,
        'duration': duration_array,
        'anf_type': anf_array,
        'tone_freq': tone_freq_array,
        'db': db_array,
        'spikes': spikes_array,
        'run': run_array,
        'params': {
            'num_cf': num_cf,
            'num_tones': num_tones,
            'num_ANF': num_ANF,
            'peripheral_fs': peripheral_fs,
            'stim_duration': duration
        }
    }

    num_hsr, num_msr, num_lsr = num_ANF
    filename = "cochlea_spike_rates_numcf_{:d}_numtonefreq_{:d}_hml_{:d}_{:d}_{:d}_run{:03d}_nonacc.mat".format(
        num_cf, num_tones, num_hsr, num_msr, num_lsr, run)
    savemat(filename, mat_data, oned_as='column')
    print("Saved full run to: {}".format(filename))

    del all_meta_rows, all_spikes_list, mat_data, cf_array, duration_array, anf_array
    gc.collect()


# ----------- Main Multiprocessing Entrypoint -----------

if __name__ == '__main__':
    import multiprocessing
    start = time.time()

    num_cores = multiprocessing.cpu_count() - 1

    pool = multiprocessing.Pool(processes=num_cores)
    pool.map(simulate_one_run, range(num_runs))
    pool.close()
    pool.join()
    print("Finished all runs in {:.2f} seconds".format(time.time() - start))
