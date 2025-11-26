from subcorticalSTRF.BEZ2018_meanrate.matlab_interface import setup_matlab_engine
from .bruce2018_pipeline import run_bruce2018


# rf_simulator/__init__.py
from .stimulus_utils import (
    generate_stimuli_params,
    generate_ramped_tone,
    generate_tone_dictionary,
    generate_tone_generator,
)
from .simulator_utils import (
    calc_cfs,
    generateANpopulation,
    get_fiber_params,
    _run_ihc_only_channel,
    _run_synapse_channel,
    run_ihc,
    run_synapse,
    get_fiber_struct_array,
    generateANpopulation_separate_arrays

)

__all__ = [
    "generate_stimuli_params",
    "generate_ramped_tone",
    "generate_tone_dictionary",
    "generate_tone_generator",
    "calc_cfs",
    "generateANpopulation",
    "get_fiber_params",
    "_run_ihc_only_channel",
    "_run_synapse_channel",
    "run_ihc",
    "run_synapse",
    "run_bruce2018",
    "setup_matlab_engine",
    "get_fiber_struct_array"
    "generateANpopulation_separate_arrays"

]



