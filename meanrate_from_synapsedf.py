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

# ===============================
# LOAD FILES
# ===============================
peripheral_fs = 100e3
root = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF"
anf_num = 8
anf_type = 'lsr'
fname = f"{root}/synapse_df_{anf_type}_{anf_num}.pkl"

with open(fname, 'rb') as f:
    synapse_df_hsr = pickle.load(f)

# Add firing rate column to the HSR Dataframe
synapse_df_hsr["firing_rate"] = synapse_df_hsr["psth"].apply(
	lambda psth: compute_firing_rate_exclude_onset(
		psth_array=psth,
		fs=peripheral_fs,
		stim_duration_sec = 0.2,
		tau_ramp_sec=0.01
		)
	)
# step 2: group by cf, tone index, freq and level


mean_std_rate_df_hsr = synapse_df_hsr.groupby(
    ["cf_idx", "cf", "tone_idx", "freq", "db"], as_index=False
)["firing_rate"].agg(mean_firing_rate = 'mean', std_firing_rate = 'std')

num_cfs = mean_std_rate_df_hsr["cf_idx"].max()
for cf_idx in mean_std_rate_df_hsr["cf_idx"].unique():
	# Filter the rows for this CF
	cf_data = mean_std_rate_df_hsr[mean_std_rate_df_hsr["cf_idx"] == cf_idx]
	cf_value = cf_data["cf"].iloc[0]

	plt.figure(figsize=(8,6))

	# Plot error bars for each stimulus frequency
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
	plt.title(f"{anf_type} ({anf_num} fibers) Rate-Level Function – CF = {cf_value:.0f} Hz")
	plt.legend(title="Tone Frequency")
	plt.grid(True)
	plt.tight_layout()
	plt.show()

# # Add firing rate column to the HSR Dataframe
# synapse_df_hsr["firing_rate"] = synapse_df_hsr["psth"].apply(
# 	lambda psth: compute_firing_rate_exclude_onset(
# 		psth_array=psth,
# 		fs=peripheral_fs,
# 		stim_duration_sec = 0.2,
# 		tau_ramp_sec=0.01
# 		)
# 	)
# # step 2: group by cf, tone index, freq and level
# mean_rate_df_hsr = synapse_df_hsr.groupby(
# 	["cf_idx", "freq", "db"], as_index=False)["firing_rate"].mean()
#

# mean_rate_df_lsr = synapse_df_lsr.groupby(
# 	["cf_idx", "cf", "tone_idx", "freq", "db"], as_index=False)["firing_rate"].mean()