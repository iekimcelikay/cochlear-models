import sys, os
print("DEBUG: __file__ =", __file__)
print("DEBUG: sys.argv =", sys.argv)
print("DEBUG: __name__ =", __name__)

#from __future__ import print_function, division
import thorns as th
import numpy as np
import cochlea
import soundfile as sf

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from cochlea.zilany2014 import calc_cfs

import time
import gc
import argparse
from scipy.io import savemat
from scipy.signal import decimate
import numpy.random as rnd
from soundgen import SoundGen



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ----------- Parameter Setup (defaults) -----------
peripheral_fs = 100e3
min_cf = 125
max_cf = 2500
num_cf = 20
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

# ----------- Stimulus Helpers -----------

def generate_stimuli_params(freq_range_tuple, num_cf_unused, db_range_tuple):
    desired_dbs = np.arange(*db_range_tuple)
    desired_freqs = calc_cfs(freq_range_tuple, species='human')
    return desired_dbs, desired_freqs

def generate_ramped_tone(sound_gen, freq, num_harmonics_local, duration_local, harmonic_factor_local, db):
    tone = sound_gen.sound_maker(freq, num_harmonics_local, duration_local, harmonic_factor_local, db)
    return sound_gen.sine_ramp(tone)

def get_cf_list(min_cf_local, max_cf_local, num_cf_local):
    return calc_cfs((min_cf_local, max_cf_local, num_cf_local), species='human')

def split_cf_indices(num_cf_local, batch_size):
    start = 0
    while start < num_cf_local:
        end = min(num_cf_local, start + batch_size)
        yield (start, end)
        start = end

def prepare_stimuli_pairs():
    desired_dbs, desired_freqs = generate_stimuli_params(freq_range, num_cf, db_range)
    pairs = []
    for db in desired_dbs:
        for freq in desired_freqs:
            pairs.append((float(freq), float(db)))
    return pairs

# ----------- Full Legacy Multi-Run (all freqs * levels * CFs) -----------

def simulate_one_run(run):
    print("Running simulation for run {}".format(run))
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
            trains_acc = sound_gen.cochlea_rate_manual(
                my_tone,
                peripheral_fs,
                min_cf,
                max_cf,
                num_cf,
                num_ANF,
                seed=unique_seed
            )
            trains_acc.sort_values(['cf', 'type'], ascending=[True, True], inplace=True)

            signal_array = th.trains_to_array(trains_acc, peripheral_fs).T
            n_fibers = len(trains_acc)

            for i_fiber in xrange(n_fibers):
                signal = signal_array[i_fiber].astype(np.float32)
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

            del trains_acc, signal_array, my_tone
            gc.collect()

    n = len(all_meta_rows)
    cf_array = np.array([m['cf'] for m in all_meta_rows])
    duration_array = np.array([m['duration'] for m in all_meta_rows])
    anf_array = np.array([m['fiber_type'] for m in all_meta_rows])
    tone_freq_array = np.array([m['tone_freq'] for m in all_meta_rows])
    db_array = np.array([m['db'] for m in all_meta_rows])
    run_array = np.array([m['run'] for m in all_meta_rows])
    spikes_array = np.empty(n, dtype=object)
    for i in xrange(n):
        spikes_array[i] = all_spikes_list[i]

    num_hsr, num_msr, num_lsr = num_ANF
    filename = "cochlea_spike_rates_numcf_{:d}_numtonefreq_{:d}_hml_{:d}_{:d}_{:d}_run{:03d}_nonacc.mat".format(
        num_cf, num_tones, num_hsr, num_msr, num_lsr, run)

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
    savemat(filename, mat_data, oned_as='column')
    print("Saved full run to: {}".format(filename))

    del all_meta_rows, all_spikes_list, mat_data
    gc.collect()

# ----------- Batch Processing For Single (freq, db) Across CF Subset -----------

def process_freq_db_cf_batch(freq,
                             db,
                             cf_start,
                             cf_end,
                             run,
                             seed,
                             output_dir,
                             verbose):
    if seed is None:
        seed = (int(time.time() * 1e6) + os.getpid() + run + int(freq) + int(db) + cf_start) % (2**32)
    np.random.seed(seed)
    if verbose:
        print("[Batch] freq={:.2f} Hz db={:.1f} run={} cf_slice=({}:{}) seed={}".format(
            freq, db, run, cf_start, cf_end, seed))

    sound_gen = SoundGen(sample_rate, tau_ramp)
    my_tone = generate_ramped_tone(sound_gen, freq, num_harmonics, duration, harmonic_factor, db)

    # Compute full set then subset (keeps ordering consistency)
    trains_acc = sound_gen.cochlea_rate_manual(
        my_tone,
        peripheral_fs,
        min_cf,
        max_cf,
        num_cf,
        num_ANF,
        seed=seed
    )
    trains_acc.sort_values(['cf', 'type'], ascending=[True, True], inplace=True)

    cf_list_full = get_cf_list(min_cf, max_cf, num_cf)
    cf_subset_values = cf_list_full[cf_start:cf_end]

    subset_df = trains_acc[trains_acc['cf'].isin(cf_subset_values)].copy()
    subset_df.sort_values(['cf', 'type'], ascending=[True, True], inplace=True)

    signal_array = th.trains_to_array(subset_df, peripheral_fs).T
    n_fibers = len(subset_df)

    all_meta_rows = []
    all_spikes_list = []

    for i_fiber in xrange(n_fibers):
        signal = signal_array[i_fiber].astype(np.float32)
        meta = {
            'cf': subset_df.iloc[i_fiber]['cf'],
            'duration': subset_df.iloc[i_fiber]['duration'],
            'fiber_type': subset_df.iloc[i_fiber]['type'],
            'tone_freq': freq,
            'db': db,
            'run': run,
            'cf_batch_start': cf_start,
            'cf_batch_end': cf_end
        }
        all_meta_rows.append(meta)
        all_spikes_list.append(signal)

    n = len(all_meta_rows)
    cf_array = np.array([m['cf'] for m in all_meta_rows])
    duration_array = np.array([m['duration'] for m in all_meta_rows])
    anf_array = np.array([m['fiber_type'] for m in all_meta_rows])
    tone_freq_array = np.array([m['tone_freq'] for m in all_meta_rows])
    db_array = np.array([m['db'] for m in all_meta_rows])
    run_array = np.array([m['run'] for m in all_meta_rows])
    batch_start_array = np.array([m['cf_batch_start'] for m in all_meta_rows])
    batch_end_array = np.array([m['cf_batch_end'] for m in all_meta_rows])

    spikes_array = np.empty(n, dtype=object)
    for i in xrange(n):
        spikes_array[i] = all_spikes_list[i]

    num_hsr, num_msr, num_lsr = num_ANF
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError:
            pass

    filename = os.path.join(
        output_dir,
        "cochlea_spike_rates_freq{:.2f}_db{:d}_cfbatch_{:03d}_{:03d}_hml_{:d}_{:d}_{:d}_run{:03d}.mat".format(
            freq, int(db), cf_start, cf_end, num_hsr, num_msr, num_lsr, run
        )
    )

    mat_data = {
        'cf': cf_array,
        'duration': duration_array,
        'anf_type': anf_array,
        'tone_freq': tone_freq_array,
        'db': db_array,
        'spikes': spikes_array,
        'run': run_array,
        'cf_batch_start': batch_start_array,
        'cf_batch_end': batch_end_array,
        'params': {
            'num_cf_total': num_cf,
            'cf_start': cf_start,
            'cf_end': cf_end,
            'num_ANF': num_ANF,
            'peripheral_fs': peripheral_fs,
            'stim_duration': duration
        }
    }
    savemat(filename, mat_data, oned_as='column')
    if verbose:
        print("[Batch] Saved {} fibers to {}".format(n, filename))

    del (trains_acc, subset_df, signal_array, my_tone,
         all_meta_rows, all_spikes_list, mat_data)
    gc.collect()

# ----------- CLI Entrypoint -----------

def main():
    print("DEBUG: main() entered")
    parser = argparse.ArgumentParser(description="Cochlea mean-rate batch / legacy processor (Python 2.7 compatible)")
    parser.add_argument('--list-pairs', action='store_true', help='List all (freq, dB) pairs and exit')
    parser.add_argument('--freq', type=float, help='Tone frequency (Hz) for batch mode')
    parser.add_argument('--db', type=float, help='Tone level (dB SPL) for batch mode')
    parser.add_argument('--cf-batch-size', type=int, help='Number of CFs per batch (batch mode)')
    parser.add_argument('--batch-index', type=int, help='CF batch index (0-based) (batch mode)')
    parser.add_argument('--all-cf', action='store_true', help='Process all CFs (ignore batching)')
    parser.add_argument('--run', type=int, default=0, help='Run ID')
    parser.add_argument('--seed', type=int, default=None, help='Optional seed override')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--legacy-full-run', action='store_true', help='Execute original full grid over (freq, dB)')
    parser.add_argument('--num-runs', type=int, default=num_runs, help='Number of legacy runs')
    parser.add_argument('--num-cf', type=int, default=num_cf, help='Override number of CFs (global)')
    parser.add_argument('--min-cf', type=float, default=min_cf, help='Override min CF')
    parser.add_argument('--max-cf', type=float, default=max_cf, help='Override max CF')
    args = parser.parse_args()

    global num_cf, min_cf, max_cf
    num_cf = args.num_cf
    min_cf = args.min_cf
    max_cf = args.max_cf

    pairs = prepare_stimuli_pairs()

    if args.list_pairs:
        print("Index\tFreq(Hz)\tLevel(dB)")
        for i, (f, d) in enumerate(pairs):
            print("{}\t{:.2f}\t{:.1f}".format(i, f, d))
        return

    if args.legacy_full_run:
        print("Starting legacy full run mode with {} runs".format(args.num_runs))
        import multiprocessing
        start = time.time()
        # Use range for Python 2 list
        runs_list = range(args.num_runs)
        cpu = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=max(1, cpu - 1))
        pool.map(simulate_one_run, runs_list)
        pool.close()
        pool.join()
        print("Finished all runs in {:.2f} seconds".format(time.time() - start))
        return
    # Batch / single-tone logic
    if not args.legacy_full_run:
        if args.freq is None or args.db is None:
            print("Error: need --freq and --db (or --list-pairs / --legacy-full-run)")
            sys.exit(1)

        cf_list_full = get_cf_list(min_cf, max_cf, num_cf)
        if args.all_cf:
            cf_start, cf_end = 0, len(cf_list_full)
        else:
            if args.cf_batch_size is None or args.batch_index is None:
                print("Error: need --cf-batch-size and --batch-index (or use --all-cf)")
                sys.exit(1)
            batches = list(split_cf_indices(len(cf_list_full), args.cf_batch_size))
            if args.batch_index < 0 or args.batch_index >= len(batches):
                print("Error: batch-index out of range (0 to {}).".format(len(batches) - 1))
                sys.exit(1)
            cf_start, cf_end = batches[args.batch_index]

        process_freq_db_cf_batch(
            freq=args.freq,
            db=args.db,
            cf_start=cf_start,
            cf_end=cf_end,
            run=args.run,
            seed=args.seed,
            output_dir=args.output_dir,
            verbose=True
        )

    # Batch mode
    if args.freq is None or args.db is None or args.cf_batch_size is None or args.batch_index is None:
        print("Error: In batch mode you must supply --freq --db --cf-batch-size --batch-index (or use --list-pairs / --legacy-full-run)")
        sys.exit(1)

    # Validate pair exists (allow small tolerance)
    found = False
    for (f, d) in pairs:
        if abs(f - args.freq) < 1e-6 and abs(d - args.db) < 1e-6:
            found = True
            break
    if not found:
        print("Warning: (freq, db) pair not in generated grid; continuing anyway.")

    cf_list_full = get_cf_list(min_cf, max_cf, num_cf)
    batches = list(split_cf_indices(len(cf_list_full), args.cf_batch_size))
    if args.batch_index < 0 or args.batch_index >= len(batches):
        print("Error: batch-index out of range (0 to {}).".format(len(batches) - 1))
        sys.exit(1)

    cf_start, cf_end = batches[args.batch_index]
    process_freq_db_cf_batch(
        freq=args.freq,
        db=args.db,
        cf_start=cf_start,
        cf_end=cf_end,
        run=args.run,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=True
    )
    return
    
if __name__ == '__main__':
    main()
