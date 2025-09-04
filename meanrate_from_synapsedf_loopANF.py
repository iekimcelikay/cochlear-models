import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

def compute_firing_rate_exclude_onset(psth_array, fs, stim_duration_sec, tau_ramp_sec = 0.01):
	# Convert MATLAB double to NumPy array if needed
	if not isinstance(psth_array, np.ndarray):
		psth_array = np.array(psth_array).squeeze()
	onset_sample = int(tau_ramp_sec * fs)
	offset_sample = int(stim_duration_sec * fs)
	stim_window = psth_array[onset_sample:offset_sample]
	spike_count = stim_window.sum()
	stim_window_duration = stim_duration_sec - tau_ramp_sec
	return spike_count / stim_window_duration

def matlab_double_to_np_array(matlab_double_obj):
	np_array = np.array(matlab_double_obj).squeeze()
	return np_array


anf_types = ['lsr', 'msr', 'hsr']
anf_num = 8
peripheral_fs = 100e3
stim_duration_sec = 0.2
tau_ramp_sec = 0.01
root = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF"

# Dictionaries to hold data for each ANF type
synapse_dfs = {}
mean_std_rate_dfs = {}

for anf_type in anf_types:
    fname = f"{root}/synapse_df_{anf_type}_{anf_num}.pkl"

    with open(fname, 'rb') as f:
        synapse_df = pickle.load(f)

    synapse_df["firing_rate"] = synapse_df["psth"].apply(
        lambda psth: compute_firing_rate_exclude_onset(
            psth_array=psth,
            fs=peripheral_fs,
            stim_duration_sec=stim_duration_sec,
            tau_ramp_sec=tau_ramp_sec
        )
    )

    grouped_df = synapse_df.groupby(
        ["cf_idx", "cf", "tone_idx", "freq", "db"], as_index=False
    )["firing_rate"].agg(mean_firing_rate='mean', std_firing_rate='std')

    # Store in dictionaries
    synapse_dfs[anf_type] = synapse_df
    mean_std_rate_dfs[anf_type] = grouped_df

    # PLOT for each ANF type
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
