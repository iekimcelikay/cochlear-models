# run_ratemodel_with_matlabwrapper.py
# created: 27.06.2025 
# last change: 24.07.2025
import sys
import os
import psutil
import time
import matlab


from datetime import datetime
import gc
from folder_manager import FolderManager
from simulator_utils import _run_ihc_only_channel, generateANpopulation_separate_arrays
from stimulus_utils import generate_tone_generator, calc_cfs, generate_stimuli_params
from matlab_interface import setup_matlab_engine
from soundgen import SoundGen

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

###----------------------------------------------------------------------------###
def print_mem_usage():
	process = psutil.Process(os.getpid())
	mem = process.memory_info()
	print(f"Memory usage: RSS = {mem.rss / (1024 **2):.2f} MB, VMS = {mem.vms / (1024 ** 2):.2f} MB")

def tone_generator_fn():
	return generate_tone_generator(sound_gen, db_range, tone_freq_range, total_number_of_tone_freq, num_harmonics, tone_duration, harmonic_factor)

def run_bruce2018_matlabwrapper(
    tone_generator_fn, #this is now a generator or any iterable yielding (db, freq, tone)
    cfs,
    peripheral_fs,
    num_ANF,
    eng,
	fibers_filename,
	run_idx=0,
	save_dir="." # Default to current directory
):
	print("[INFO] Memory usage BEFORE running model:")
	print_mem_usage()

	timestamp = time.strftime("%Y%m%d_%H%M%S")
	nrep = float(1)
	dt = float(1 / peripheral_fs)
	expliketype = float(1)  # softplus
	implmnt = float(1)      # actual power-law implementation
	noiseType = float(1)    # Random gaussian noise

	all_synapse_results = []  # Python list to collect all results
	all_synapse_params = [] # Python list to collect the relevant parameters

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

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

			# --- Run MATLAB Synapse Model per CF-tone ---
			try:
				num_lsr, num_msr, num_hsr = num_ANF
				print(f"[INFO] SYNAPSE tone_idx= {tone_idx}:freq = {freq} Hz @ {db} dB for CF= {cf:.1f} Hz")
				print(f"[PARAMETERS] Number of ANF fibers: {num_lsr}, {num_msr}, {num_hsr}")
				synapse_result_struct = eng.run_fiber_synapse_batch_parfor(
					vihc,cf,cf_idx,freq, db, tone_idx, nrep, dt, noiseType, implmnt, expliketype, fibers_filename, nargout=1
				)
			except matlab.engine.MatlabExecutionError as e:
				print("MATLAB ERROR:\n", e)
				raise

			# Save each synapse result immediately to disk
			eng.workspace['synapse_result_struct'] = synapse_result_struct
			filename =os.path.join(save_dir, f"synapse_result_cf{cf:.1f}hz_tone{freq:.1f}hz_{db}db_{num_lsr}-{num_msr}-{num_hsr}fibers_run{run_idx}.mat")
			eng.save(filename, 'synapse_result_struct', nargout=0)

			# Free memory
		del synapse_result_struct, tone, vihc, chan
		gc.collect()

	return


###===============================================================================================================

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
num_cf = 5
freq_range = (125, 2500, num_cf)
num_ANF = (64, 64, 64)  # LSR, MSR, HSR. this parameter is needed for the generateANpopulation.
num_lsr, num_msr, num_hsr = num_ANF
cfs = calc_cfs(freq_range, species='human')

model_folder = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/BEZ2018_meanrate"
fibers_filename = os.path.join(model_folder, f"ANpopulation_{num_lsr}-{num_msr}-{num_hsr}.mat")
fibers_dict = generateANpopulation_separate_arrays(num_cf, num_ANF, filename=fibers_filename)

results_folder = "/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/BEZ2018_meanrate/results"
# ================================
# 2. STIMULI GENERATION
# ================================
sound_gen = SoundGen(sample_rate, tau_ramp)
desired_dbs, desired_freqs = generate_stimuli_params(
	tone_freq_range, num_cf=total_number_of_tone_freq, db_range=db_range
)

# ================================
# 3. START MATLAB ENGINE
# ================================
eng = setup_matlab_engine('/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/BEZ2018_meanrate/BEZ2018a_model')
eng.addpath('/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/BEZ2018_meanrate', nargout=0)
eng.addpath('/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/BEZ2018_meanrate/results', nargout=0)
# ✅ Start parallel pool with 2 workers if not already running
eng.eval("if isempty(gcp('nocreate')); parpool('local', 8); end", nargout=0)


# ================================
# 4. RUN BRUCE2018 MODEL
# ================================
# Initialize FolderManager with parameters
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

if __name__ == "__main__":
	num_repeats = 4
	print("[INFO] Starting full batch simulation...")
	start_time = time.time()

	# Initialize FolderManager once:
	folder_manager = FolderManager(base_dir=results_folder, num_repeats=num_repeats, num_cf=num_cf, num_ANF=num_ANF, stimulus_params=stimulus_parameters)

	# Create a folder for the entire batch, save stim params text
	folder_manager.create_batch_folder()

	for run_idx in range(num_repeats):
		print(f"\n[INFO] Starting trial {run_idx + 1} of {num_repeats}")
		run_bruce2018_matlabwrapper(tone_generator_fn, cfs, peripheral_fs, num_ANF, eng, fibers_filename, run_idx=run_idx, save_dir=folder_manager.results_folder)

	end_time = time.time()
	elapsed = end_time - start_time
	print(f"[INFO] Total elapsed time for all {num_repeats} runs: {elapsed:.2f} seconds")
	print("[INFO] Memory usage AFTER running model:")
	print_mem_usage()

