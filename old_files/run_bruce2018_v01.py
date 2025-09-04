# MATLAB PYTHON INTEGRATION TEST FILE

# I have my sound file as an np.array, but I will need that in MATLAB, as a 'double'.
# apparently double() works?
#
# CODE STARTS HERE
############################################################################
# IMPORTS
#-----------------------------------
import numpy as np
import pandas as pd
import matlab.engine
from subcorticalSTRF.stim_generator import soundgen
from subcorticalSTRF.bruce2018_python.simulator_utils import calc_cfs, generateANpopulation, get_fiber_params
from subcorticalSTRF.bruce2018_python.stimulus_utils import generate_stimuli_params, generate_ramped_tone
from subcorticalSTRF.bruce2018_python.simulator_utils import run_ihc, _run_ihc_only_channel
from subcorticalSTRF.bruce2018_python.simulator_utils import run_synapse, _run_synapse_channel
import pickle


# PARAMETERS
#------------------------------------
# Parameters for the inner ear model
peripheral_fs = 100e3
num_cf = 5             # Number of characteristic frequencies (CF)s desired
freq_range = (125, 2500, num_cf)
num_ANF = (3,5,12) # 30 x 0.15 , 30 x 0.25, 30 x 0.60, (LSR, MSR, HSR); or 3-5-12 if using 20 fibers.
# or 4-4-12 configuration as Vecchi 2022.
# Create a list of Characteristic Frequencies of Fibers
cfs = calc_cfs(freq_range, species='human')
sponts, tabs, trels = generateANpopulation(len(cfs),num_ANF)

# Parameters for the stimuli generation
sample_rate = 100e3     # Sample rate of the sound stimuli
tau_ramp = 0.01         # Ramping window for the sound stimuli
num_harmonics = 1       # Number of harmonics, i.e., 1 for simple tone, >1 for complex harmonic tones.
tone_duration = 0.200   # The duration of single tone, in seconds.
harmonic_factor = 0.5   # Harmonic decay factor, that will change the timbre of the sound.
db_range = (90,110, 5)
total_number_of_tones = 5 # I can later equate this to num_cf, but for now let's keep it separate.
tone_freq_range = (125, 2500, total_number_of_tones)

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

# Save tones to a dictionary file using pickle (for inspection)
with open("tones.pkl", "wb") as f:
    pickle.dump(tones, f)




# Start Matlab Engine here and pass it as an argument to the model functions
eng = matlab.engine.start_matlab("-desktop")
model_folder = eng.genpath("/home/ekim/audio_periph_models/BEZ2018a_model")
eng.addpath(model_folder, nargout =0)


# 2. RUN IHC MODEL
#_______________________________________

ihc_channel_args=[ (
    {
        'signal': tone,
        'cf' :    cf,
        'fs' : peripheral_fs,
        'db': db,
        'freq': frequency
    },
        cf_idx,
        i_tone,
)
    for i_tone, ((db,frequency), tone) in enumerate(tones.items())
    for cf_idx, cf in enumerate(cfs)
]

# Print all (tone, cf) channel args for inspection
for args_dict, cf_idx, i_tone in ihc_channel_args:
    print("---- CHANNEL ARG ----")
    print(f"Tone index     (i_tone): {i_tone}")
    print(f"CF index       (cf_idx): {cf_idx}")
    print(f"dB level             : {args_dict['db']}")
    print(f"Tone frequency (Hz)  : {args_dict['freq']}")
    print(f"Characteristic Freq  : {args_dict['cf']}")
    print(f"Sampling Rate        : {args_dict['fs']}")
    print(f"Signal shape         : {args_dict['signal'].shape}")
    print()

ihc_results = list(map(lambda a: _run_ihc_only_channel(*a, eng), ihc_channel_args))
ihc_df = pd.DataFrame(ihc_results)

# Inspection
print(ihc_df[['cf_idx', 'cf', 'i_tone', 'db']].head(10))

# 3. RUN SYNAPSE MODEL
#______________________________________
synapse_input_args = []

for idx, ihc_result in enumerate(ihc_results):
    cf_idx = ihc_df.loc[idx, 'cf_idx']
    cf = ihc_df.loc[idx, 'cf']
    i_tone = ihc_df.loc[idx, 'i_tone']
    fs = peripheral_fs
    vihc = ihc_df.loc[idx, 'vihc'] # Should be a Numpy array or MATLAB double
    db_level = ihc_df.loc[idx, 'db']

    fiber_params = get_fiber_params(cf_idx, sponts, tabs, trels, num_ANF)
    print(f"idx={idx}, cf_idx={cf_idx}, i_tone={i_tone}, db={db_level}, cf={cf}")

    for fiber_idx, fiber in enumerate(fiber_params):
        fiber_args = {
            'cf_idx': cf_idx,
            'db': db_level,
            'cf': cf,
            'fs': fs,
            'vihc': vihc,
            'tone_idx': i_tone,
            'anf_type': fiber['anf_type'],
            'spont': fiber['spont'],
            'tab': fiber['tab'],
            'trel': fiber['trel'],
            'fiber_idx': fiber_idx,
        }
        synapse_input_args.append(fiber_args)

print(f"Total fibers to run synapse for: {len(synapse_input_args)}")
cf_set = set(arg['cf_idx'] for arg in synapse_input_args)
print(f"Unique CF indices in synapse input: {sorted(cf_set)}")
db_set = set(arg['db'] for arg in synapse_input_args)
print(f"Unique dB levels in synapse input: {sorted(db_set)}")

# Run the synapse model for all the parameters entered into the dictionary
synapse_results = [_run_synapse_channel(args, eng=eng) for args in synapse_input_args]
synapse_df = pd.DataFrame.from_records(synapse_results)

# Exit MATLAB engine (as a safety measure, otherwise for some reason it crashes easily)
eng.quit()


#_______________ COMMENTED
# index tones differently:
# Index them with their 'index keys' i_db, i_freq

# tones = {
#     (i_db, i_freq): generate_ramped_tone(sound_gen, frequency, num_harmonics, tone_duration, harmonic_factor, db)
#     for i_db, db in enumerate(desired_tone_dbs)
#     for i_freq, frequency in enumerate(desired_tone_frequencies)
# }

# Indexing them with the units themselves tones[(60, 1000)]

#_______________ COMMENTED
########################################################################################3

