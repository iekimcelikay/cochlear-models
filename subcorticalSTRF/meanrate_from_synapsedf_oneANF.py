import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import gc
import matplotlib.pyplot as plt
import math

def compute_firing_rate_exclude_onset(psth_array, fs, stim_duration_sec, tau_ramp_sec=0.01):
    if not isinstance(psth_array, np.ndarray):
        psth_array = np.array(psth_array).squeeze()
    onset_sample = int(tau_ramp_sec * fs)
    offset_sample = int(stim_duration_sec * fs)
    stim_window = psth_array[onset_sample:offset_sample]
    spike_count = stim_window.sum()
    stim_window_duration = stim_duration_sec - tau_ramp_sec
    return spike_count / stim_window_duration

# Settings
anf_types = ['lsr', 'msr', 'hsr']
anf_num = 8
peripheral_fs = 100e3
stim_duration_sec = 0.2
tau_ramp_sec = 0.01
root = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF"
output_dir = f"{root}/rate_level_outputs"

# Process one ANF type at a time
for anf_type in anf_types:
    print(f"\nProcessing {anf_type.upper()}...")

    fname = f"{root}/synapse_df_{anf_type}_{anf_num}.pkl"
    with open(fname, 'rb') as f:
        synapse_df = pickle.load(f)

    # Compute firing rate column
    synapse_df["firing_rate"] = synapse_df["psth"].apply(
        lambda psth: compute_firing_rate_exclude_onset(
            psth_array=psth,
            fs=peripheral_fs,
            stim_duration_sec=stim_duration_sec,
            tau_ramp_sec=tau_ramp_sec
        )
    )

    # Group and compute stats
    grouped_df = synapse_df.groupby(
        ["cf_idx", "cf", "tone_idx", "freq", "db"], as_index=False
    )["firing_rate"].agg(mean_firing_rate='mean', std_firing_rate='std')

    # Save aggregated results
    grouped_fname = f"{output_dir}/mean_std_rate_df_{anf_type}_{anf_num}.pkl"
    grouped_df.to_pickle(grouped_fname)

    # Plot and discard
    for cf_idx in grouped_df["cf_idx"].unique():
        cf_data = grouped_df[grouped_df["cf_idx"] == cf_idx]
        cf_value = cf_data["cf"].iloc[0]

        plt.figure(figsize=(8, 6))
        for freq in sorted(cf_data["freq"].unique()):
            sub = cf_data[cf_data["freq"] == freq]
            plt.errorbar(
                sub["db"],
                sub["mean_firing_rate"],
                yerr=sub["std_firing_rate"],
                label=f"{freq} Hz",
                capsize=3
            )

        plt.xlabel("Stimulus Level (dB SPL)")
        plt.ylabel("Mean Firing Rate (spikes/s)")
        plt.title(f"{anf_type.upper()} ({anf_num} fibers) Rate-Level Function – CF = {cf_value:.0f} Hz")
        plt.legend(title="Tone Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Clean up to free memory
    del synapse_df
    del grouped_df
    gc.collect()
