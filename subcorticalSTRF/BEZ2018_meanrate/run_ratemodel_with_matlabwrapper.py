# run_ratemodel_with_matlabwrapper.py
# created: 27.06.2025 
# last change: 27.06.2025
import sys
import os
from memory_profiler import memory_usage
import psutil

import time

import matlab
import numpy as np
import pandas as pd
from matlab_interface import setup_matlab_engine

from datetime import datetime
import gc

start = time.time()
from subcorticalSTRF.rf_simulator import (
    # Stimulus utilities
    generate_stimuli_params, generate_ramped_tone, generate_tone_dictionary, generate_tone_generator,
    # Simulation utilities
    calc_cfs, generateANpopulation, get_fiber_params, get_fiber_struct_array, generateANpopulation_separate_arrays
)

from subcorticalSTRF.rf_simulator.simulator_utils import _run_ihc_only_channel
from subcorticalSTRF.stim_generator import SoundGen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
###----------------------------------------------------------------------------###
def print_mem_usage():
	process = psutil.Process(os.getpid())
	mem = process.memory_info()
	print(f"Memory usage: RSS = {mem.rss / (1024 **2):.2f} MB, VMS = {mem.vms / (1024 ** 2):.2f} MB")
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

def matlab_double_to_np_array(matlab_double_obj):
	np_array = np.array(matlab_double_obj).squeeze()
	return np_array


def run_bruce2018_matlabwrapper(
    tone_generator_fn, #this is now a generator or any iterable yielding (db, freq, tone)
    cfs,
    peripheral_fs,
    num_ANF,
    eng,
	run_idx=0
):
	print("[INFO] Memory usage BEFORE running model:")
	print_mem_usage()

	timestamp = time.strftime("%Y%m%d_%H%M%S")
	nrep = float(1)
	dt = float(1 / peripheral_fs)
	expliketype = float(1)  # softplus
	implmnt = float(1)      # actual power-law implementation
	noiseType = float(0)    # Random gaussian noise

	all_synapse_results = []  # Python list to collect all results
	all_synapse_params = [] # Python list to collect the relevant parameters 


	for cf_idx, cf in enumerate(cfs):
		print(f"\n[INFO] Starting CF index {cf_idx} with CF = {cf:.1f} Hz")
		# recreate the generator for each CF
		tones_iter = tone_generator_fn()
		for tone_idx, tone_info in enumerate(tones_iter):
			# now tone is generated here with the generator function, not upfront 
			if isinstance(tone_info, tuple) and len(tone_info) == 3:
				db, freq, tone = tone_info
			else:
				print(f"[WARNING] Unexpected tone_info format at tone_idx = {tone_idx}: {tone_info}")
				continue 

			print(f"[INFO] Processing tone_idx= {tone_idx}:freq = {freq} Hz @ {db} dB for CF= {cf:.1f} Hz")

            # --- Run IHC for this CF-tone combo ---
			chan = {
				'signal': tone,
				'cf': cf,
				'fs': peripheral_fs,
				'db': db,
				'freq': freq,
				'cf_idx': cf_idx,
				'i_tone': tone_idx,
			}

			# Run IHC
			vihc = _run_ihc_only_channel(chan, eng)
			print(f" cf is {cf} Hz ")	
			print(f"cf index is {cf_idx}")

			# --- Run MATLAB Synapse Model per CF-tone ---
			try:
				synapse_result_struct = eng.run_fiber_synapse_batch_parfor(
					vihc,cf,cf_idx,freq, db, tone_idx, nrep, dt, noiseType, implmnt, expliketype, f"/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/ANpopulation_run{run_idx}.mat",nargout=1
				)
			except matlab.engine.MatlabExecutionError as e:
				print("MATLAB ERROR:\n", e)
				raise
			

			# Optionally convert MATLAB struct to Python
			all_synapse_results.append(synapse_result_struct)

	num_lsr, num_msr, num_hsr = num_ANF
	# Optionally save in MATLAB workspace
	eng.workspace['all_synapse_results'] = all_synapse_results
	filename = f"synapse_results_2runs_run{run_idx}_{timestamp}_{num_lsr}-{num_msr}-{num_hsr}_RandomNoise.mat"
	eng.save(filename, 'all_synapse_results', nargout=0)
	print(f"[PARAMETERS] Number of ANF fibers: {num_lsr}, {num_msr}, {num_hsr}")

	return all_synapse_results

	###===============================================================================================================
# save results to 

results_folder = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF"
filename_fibers = os.path.join(results_folder, "ANpopulation.mat")

# Stimulus parameters
sample_rate = 100e3
tau_ramp = 0.01
num_harmonics = 1
tone_duration = 0.200
harmonic_factor = 0.5
db_range = (50, 90, 10)
total_number_of_tone_freq = 5
tone_freq_range = (125, 2500, total_number_of_tone_freq)
psth_bin_width = 0.0005

# Inner ear model parameters
peripheral_fs = 100e3
num_cf = 3
freq_range = (125, 2500, num_cf)
num_ANF = (32, 32, 32)  # LSR, MSR, HSR. this parameter is needed for the generateANpopulation.
cfs = calc_cfs(freq_range, species='human')
print(cfs)

fibers_dict = generateANpopulation_separate_arrays(num_cf, num_ANF, filename=filename_fibers)

# ================================
# 2. STIMULI GENERATION
# ================================
sound_gen = SoundGen(sample_rate, tau_ramp)
desired_dbs, desired_freqs = generate_stimuli_params(
	tone_freq_range, num_cf=total_number_of_tone_freq, db_range=db_range
)
def tone_generator_fn():
	return generate_tone_generator(sound_gen, db_range, tone_freq_range, total_number_of_tone_freq, num_harmonics, tone_duration, harmonic_factor)

# ================================
# 3. START MATLAB ENGINE
# ================================
eng = setup_matlab_engine('/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/BEZ2018_meanrate/BEZ2018a_model')
eng.addpath('/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/BEZ2018_meanrate', nargout=0)
eng.addpath('/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/BEZ2018_meanrate/results', nargout=0)



# ================================
# 4. RUN BRUCE2018 MODEL
# ================================
if __name__ == "__main__":
	num_repeats = 16
	print("[INFO] Starting full batch simulation...")
	start_time = time.time()
	
	for run_idx in range(num_repeats):
		print(f"\n[INFO] Starting trial {run_idx + 1} of {num_repeats}")

		filename = f"/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/ANpopulation_run{run_idx}.mat"
		fibers_dict = generateANpopulation_separate_arrays(num_cf, num_ANF, filename=filename)

		eng.addpath('/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF', nargout=0)
		run_bruce2018_matlabwrapper(tone_generator_fn, cfs, peripheral_fs, num_ANF, eng, run_idx=run_idx)

	end_time = time.time()
	elapsed = end_time - start_time
	print(f"[INFO] Total elapsed time for all {num_repeats} runs: {elapsed:.2f} seconds")
	print("[INFO] Memory usage AFTER running model:")
	print_mem_usage()

