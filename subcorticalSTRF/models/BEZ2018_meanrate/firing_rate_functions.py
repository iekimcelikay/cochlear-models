import numpy as np

def compute_firing_rate_with_onset(psth_array, fs, psth_bin_width):
	# Convert MATLAB double to NumPy array if needed
	if not isinstance(psth_array, np.ndarray):
		psth_array = np.array(psth_array).squeeze()
	total_spikes = np.sum(psth_array)
	duration_sec = len(psth_array) * psth_bin_width
	return total_spikes / duration_sec

def compute_firing_rate_no_onset(psth_array, fs, psth_bin_width, onset_skip_sec=0.01):
	# Convert MATLAB double to NumPy array if needed
	if not isinstance(psth_array, np.ndarray):
		psth_array = np.array(psth_array).squeeze()
	skip_bins = int(onset_skip_sec / psth_bin_width)
	psth_no_onset = psth_array[skip_bins:]
	duration_sec = len(psth_no_onset) * psth_bin_width
	total_spikes = np.sum(psth_no_onset)
	return total_spikes / duration_sec

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

