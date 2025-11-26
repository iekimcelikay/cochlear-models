# run_ratemodel_with_matlabwrapper.py
# created: 27.06.2025 
# last change: 25.07.2025

import sys
import os
import psutil
import time
import gc
from datetime import datetime
import matlab
import numpy as np

from folder_manager import FolderManager
from an_simulator import AuditoryNerveSimulator
from simulator_utils import generateANpopulation_separate_arrays, calc_cfs
from stimulus_utils import generate_tone_generator, generate_stimuli_params
from matlab_interface import setup_matlab_engine
from subcorticalSTRF.stim_generator import SoundGen

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

###----------------------------------------------------------------------------###
def print_mem_usage():
	process = psutil.Process(os.getpid())
	mem = process.memory_info()
	print(f"Memory usage: RSS = {mem.rss / (1024 **2):.2f} MB, VMS = {mem.vms / (1024 ** 2):.2f} MB")

def tone_generator_fn():
	return generate_tone_generator(sound_gen, db_range, tone_freq_range, total_number_of_tone_freq, num_harmonics, tone_duration, harmonic_factor)

def run_bruce2018_matlabwrapper(
	tone_generator_fn,
	cfs,
	peripheral_fs,
	num_ANF,
	simulator,  # ← pass AuditoryNerveSimulator instance
	run_idx=0,
	save_dir="."
):
	print("[INFO] Memory usage BEFORE running model:")
	print_mem_usage()

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	for cf_idx, cf in enumerate(cfs):
		print(f"\n[INFO] Starting CF index {cf_idx} with CF = {cf:.1f} Hz")

		tones_iter = tone_generator_fn()
		for tone_idx, tone_info in enumerate(tones_iter):
			if isinstance(tone_info, tuple) and len(tone_info) == 3:
				db, freq, tone = tone_info
			else:
				print(f"[WARNING] Unexpected tone_info format at tone_idx = {tone_idx}: {tone_info}")
				continue

			print(f"[INFO] Processing tone_idx= {tone_idx}:freq = {freq} Hz @ {db} dB for CF= {cf:.1f} Hz")

			# Build channel dict
			chan = {
				'signal': tone,
				'cf': cf,
				'fs': peripheral_fs,
				'db': db,
				'freq': freq,
				'cf_idx': cf_idx,
				'i_tone': tone_idx,
			}

			# --- Run IHC ---
			vihc = simulator.run_ihc(chan)

			# --- Run Synapse ---
			synapse_result_struct = simulator.run_synapse(
				vihc, cf, cf_idx, freq, db, tone_idx
			)

			# Save immediately
			filename = os.path.join(
				save_dir,
				f"synapse_result_cf{cf:.1f}hz_tone{freq:.1f}hz_{db}db_{num_ANF[0]}-{num_ANF[1]}-{num_ANF[2]}fibers_run{run_idx}.mat"
			)
			simulator.eng.workspace['synapse_result_struct'] = synapse_result_struct
			simulator.eng.save(filename, 'synapse_result_struct', nargout=0)

			# Free memory
			del synapse_result_struct, tone, vihc, chan
			gc.collect()

	return

###===============================================================================================================

# ==== Parameters ====
sample_rate = 100e3
tau_ramp = 0.01
num_harmonics = 1
tone_duration = 0.200
harmonic_factor = 0.5
db_range = (50, 90, 10)
total_number_of_tone_freq = 20
min_f0 = 125
max_f0 = 2500
tone_freq_range = (min_f0, max_f0, total_number_of_tone_freq)
psth_bin_width = 0.0005

peripheral_fs = 100e3
num_cf = 20
min_cf = 125
max_cf = 2500
freq_range = (min_cf, max_cf, num_cf)
num_ANF = (128,128,128)

num_lsr, num_msr, num_hsr = num_ANF
cfs = calc_cfs(freq_range, species='human')
#cfs = calc_cfs([203.01017321, 303.51357184, 432.99582678,
#        599.81261681, 814.72887269, 1091.61345387, 1448.3341262 ,
#       1907.91059497, 2500.0], 'human')

model_folder = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/BEZ2018_meanrate"
fibers_filename = os.path.join(model_folder, f"ANpopulation_{num_lsr}-{num_msr}-{num_hsr}.mat")
fibers_dict = generateANpopulation_separate_arrays(num_cf, num_ANF, filename=fibers_filename)

results_folder = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/BEZ2018_meanrate/results"

# ==== Stimulus ====
sound_gen = SoundGen(sample_rate, tau_ramp)
desired_dbs, desired_freqs = generate_stimuli_params(
	tone_freq_range, num_cf=total_number_of_tone_freq, db_range=db_range
)

# ==== MATLAB Engine ====
eng = setup_matlab_engine(model_folder)
eng.addpath(model_folder, nargout=0)
eng.addpath(results_folder, nargout=0)
eng.eval("if isempty(gcp('nocreate')); parpool('local', 8); end", nargout=0)


# ==== FolderManager setup ====
stimulus_parameters = {
	"sample_rate": int(sample_rate),
	"tau_ramp": float(tau_ramp),
	"num_harmonics": int(num_harmonics),
	"tone_duration": float(tone_duration),
	"harmonic_factor": float(harmonic_factor),
	"total_number_of_tone_freq": int(total_number_of_tone_freq),
	"dbs": [float(db) for db in desired_dbs],
	"freq_range": [float(freq) for freq in desired_freqs],
	"number_of_cfs": int(num_cf),
	"cfs": [float(cf) for cf in cfs],
	"num_ANF": [int(num_lsr), int(num_msr), int(num_hsr)],
}

# ==== MAIN ====
if __name__ == "__main__":
	num_repeats = 10
	print("[INFO] Starting full batch simulation...")
	start_time = time.time()

	folder_manager = FolderManager(
		base_dir=results_folder,
		num_repeats=num_repeats,
		num_cf=num_cf,
        min_cf=min_cf,
        max_cf=max_cf,
		num_ANF=num_ANF,
		stimulus_params=stimulus_parameters
	)
	folder_manager.create_batch_folder()

	simulator = AuditoryNerveSimulator(peripheral_fs, num_ANF, eng, fibers_filename)

	for run_idx in range(num_repeats):
		print(f"\n[INFO] Starting trial {run_idx + 1} of {num_repeats}")
		run_bruce2018_matlabwrapper(
			tone_generator_fn,
			cfs,
			peripheral_fs,
			num_ANF,
			simulator,
			run_idx=run_idx,
			save_dir=folder_manager.results_folder
		)

	end_time = time.time()
	elapsed = end_time - start_time
	print(f"[INFO] Total elapsed time for all {num_repeats} runs: {elapsed:.2f} seconds")
	print(f"[INFO] Number of synapses each: {num_lsr}")
	print("[INFO] Memory usage AFTER running model:")
	print_mem_usage()
