# run_model.py
# This script is not working.

import pickle
import os
from datetime import datetime

import numpy as np
from subcorticalSTRF.bruce2018_python import (
    # Stimulus utilities
    generate_stimuli_params, generate_ramped_tone, generate_tone_dictionary,

    # Simulation utilities
    calc_cfs, generateANpopulation, get_fiber_params,

    # MATLAB interface
    setup_matlab_engine,

    # Model stages
    run_bruce2018
)

from subcorticalSTRF.stim_generator import SoundGen



def main():
    # ================================
    # 1. PARAMETERS
    # ================================

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
    num_ANF = (5, 5, 5)  # LSR, MSR, HSR
    cfs = calc_cfs(freq_range, species='human')
    sponts, tabs, trels = generateANpopulation(len(cfs), num_ANF)

    # ================================
    # 2. STIMULI GENERATION
    # ================================
    sound_gen = SoundGen(sample_rate, tau_ramp)
    desired_dbs, desired_freqs = generate_stimuli_params(
        tone_freq_range, num_cf=total_number_of_tone_freq, db_range=db_range
    )
    tones = generate_tone_dictionary(sound_gen, db_range, tone_freq_range, total_number_of_tone_freq, num_harmonics,tone_duration, harmonic_factor)

    # ================================
    # 3. START MATLAB ENGINE
    # ================================
    eng = setup_matlab_engine('/home/ekim/audio_periph_models/BEZ2018a_model')

    # ================================
    # 4. RUN BRUCE2018 MODEL
    # ================================
    synapse_df, ihc_df = run_bruce2018(
        tones=tones,
        cfs=cfs,
        peripheral_fs=peripheral_fs,
        num_ANF=num_ANF,
        sponts=sponts,
        tabs=tabs,
        trels=trels,
        eng=eng
    )

    # ================================
    # 5. OUTPUT
    # ================================
    print("\n--- IHC Output Preview ---")
    print(ihc_df.head())

    print("\n--- Synapse Output Preview ---")
    print(synapse_df.head())
    eng.quit()

    # Compute PSTH and firing rate


    # --- Save synapse results to a pickle file
    synapse_data_bundle = {
        'synapse_df': synapse_df,
        'num_ANF': list(num_ANF)
    }
    # --- Save results to path: (will soft-coded later, now hard_coded): ---
    results_path = f"//subcorticalSTRF/bruce2018_results"
    os.makedirs(results_path, exist_ok=True)

    # --- To generate timestamped filename ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    num_ANF_str = str(num_ANF)

    synapse_results_fname = f'synapse_results_{timestamp}.pkl'

    # --- generate full path to save the file ---
    synapse_results_fullpath = os.path.join(results_path, synapse_results_fname)
    with open(synapse_results_fullpath, 'wb') as f:
        pickle.dump(synapse_data_bundle, f)

    print(f"Saved simulation bundle to: {synapse_results_fullpath}")
    print(synapse_df.columns())
    print("function ran and matlab closed")
if __name__ == "__main__":
    main()
