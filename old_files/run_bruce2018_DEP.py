
# IMPORTS
#-----------------------------------
import numpy as np
import pandas as pd
import matlab.engine
from subcorticalSTRF.stim_generator import soundgen
from subcorticalSTRF.bruce2018_python.helperfunctions.simulator_utils import calc_cfs, generateANpopulation, generate_stimuli_params, generate_ramped_tone, get_fiber_params
from subcorticalSTRF.bruce2018_python.modelfunctions.run_ihc import run_ihc, _run_ihc_only_channel
from subcorticalSTRF.bruce2018_python.modelfunctions.run_synapse import _run_synapse_channel
import pickle

# FUNCTION
def run_bruce2018(
        tones,
        cfs,
        peripheral_fs,
        get_fiber_params,
        num_ANF,
        sponts,
        tabs,
        trels,
        _run_ihc_only_channel,
        _run_synapse_channel,
        eng
):
    # 1. Build channel args. Outer loop will be CF

    channel_args = [
        {
            'signal': tone,
            'cf': cf,
            'fs': peripheral_fs,
            'db': db,
            'freq': frequency,
            'cf_idx': cf_idx,
            'i_tone': i_tone,
        }
        for cf_idx, cf in enumerate(cfs)
        for i_tone, ((db, frequency), tone) in enumerate(tones.items())
    ]

    #2. Run IHC stage
    ihc_outputs = []

    for chan in channel_args:
        vihc = _run_ihc_only_channel(chan, eng)
        ihc_outputs.append({**chan, 'vihc': vihc})

    ihc_df = pd.DataFrame(ihc_outputs)

    # Prepare synapse input args
    synapse_inputs = []

    for chan in ihc_outputs:
        fiber_params = get_fiber_params(
            chan['cf_idx'], sponts, tabs, trels, num_ANF
        )
        for fiber_idx, fiber in enumerate(fiber_params):
            synapse_arg = {
                'cf_idx': chan['cf_idx'],
                'cf': chan['cf'],
                'tone_idx': chan['i_tone'],
                'freq': chan['freq'],
                'db': chan['db'],
                'fs': chan['fs'],
                'vihc': chan['vihc'],
                'anf_type': fiber['anf_type'],
                'fiber_idx': fiber_idx,
                'spont': fiber['spont'],
                'tab': fiber['tab'],
                'trel': fiber['trel'],

            }
            synapse_inputs.append(synapse_arg)
    # Run Synapse stage
    synapse_results = [_run_synapse_channel(args, eng=eng) for args in synapse_inputs]
    synapse_df = pd.DataFrame.from_records(synapse_results)

    return synapse_df, ihc_df


# PARAMETERS
#------------------------------------

# Parameters for the stimuli generation
sample_rate = 100e3     # Sample rate of the sound stimuli
tau_ramp = 0.01         # Ramping window for the sound stimuli
num_harmonics = 1       # Number of harmonics, i.e., 1 for simple tone, >1 for complex harmonic tones.
tone_duration = 0.200   # The duration of single tone, in seconds.
harmonic_factor = 0.5   # Harmonic decay factor, that will change the timbre of the sound.
db_range = (90,110, 5)
total_number_of_tones = 5 # I can later equate this to num_cf, but for now let's keep it separate.
tone_freq_range = (125, 2500, total_number_of_tones)

# Parameters for the inner ear model
peripheral_fs = 100e3
num_cf = 5             # Number of characteristic frequencies (CF)s desired
freq_range = (125, 2500, num_cf)
num_ANF = (4,4,4) # 30 x 0.15 , 30 x 0.25, 30 x 0.60, (LSR, MSR, HSR); or 3-5-12 if using 20 fibers.
# or 4-4-12 configuration as Vecchi 2022.
# Create a list of Characteristic Frequencies of Fibers
cfs = calc_cfs(freq_range, species='human')
sponts, tabs, trels = generateANpopulation(len(cfs),num_ANF)

# 1. STIMULI GENERATION
# ---------------------------------
# Import the sound generation class
sound_gen = soundgen.SoundGen(sample_rate, tau_ramp)
desired_tone_dbs, desired_tone_frequencies = generate_stimuli_params(tone_freq_range, num_cf=total_number_of_tones, db_range=db_range)

# Create the tones
tones = {
    (db, frequency): generate_ramped_tone(sound_gen, frequency, num_harmonics, tone_duration, harmonic_factor, db)
    for db in desired_tone_dbs
    for frequency in desired_tone_frequencies
}
for key, tone in tones.items():
    db, freq = key
    print(f"[TONE DEBUG] dB: {db}, freq: {freq}, tone RMS: {np.sqrt(np.mean(tone**2)):.2f}")

# Start Matlab Engine here and pass it as an argument to the model functions
eng = matlab.engine.start_matlab("-desktop")
model_folder = eng.genpath("/home/ekim/audio_periph_models/BEZ2018a_model")
eng.addpath(model_folder, nargout =0)



synapse_data, ihc_data = run_bruce2018(
    tones=tones,
    cfs=cfs,
    peripheral_fs=peripheral_fs,
    get_fiber_params=get_fiber_params,
    num_ANF=num_ANF,
    sponts=sponts,
    tabs=tabs,
    trels=trels,
    _run_ihc_only_channel=_run_ihc_only_channel,
    _run_synapse_channel=_run_synapse_channel,
    eng=eng
)




#_______________________________________

# for cf_idx, cf in enumerate(cfs):
#     for fiber_idx, fiber in enumerate(fiber_params):
#         for i_tone, ((db, frequency), tone) in enumerate(tones.items()):
