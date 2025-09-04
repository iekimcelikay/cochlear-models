# This will call the MATLAB function actually.
## Notes (11 June)
# For some reason (that I haven't figured out yet) MATLAB does not take python ints as int.
# Rather, it accepts 'float' values for the parameters (arguments) that require a numeric integer.
# For this reason, in the species dictionary or in the `cohc` `cihc` parameters, the numeric integers are typed as `float`s.
# One difference from how cochlea was calling the IHCAN  function: it was just len(signal) there. However it needs to be
# divided by the fs here. Maybe they handled that in argument mapping? I should check.
import pandas as pd
import itertools
import numpy as np
import matlab

def _run_ihc_only_channel(chan, eng):
    fs = chan['fs']
    cf = chan['cf']
    tone = chan['signal']
    db = chan['db']
    freq = chan['freq']
    cf_idx = chan['cf_idx']
    i_tone = chan['i_tone']

    print(f"[DEBUG] Running IHC with: CF={cf}, i_tone={i_tone}, Tone RMS={np.sqrt(np.mean(tone ** 2)):.2f}, fs={fs}, len={len(tone)}")

    # Run IHC
    vihc = run_ihc(eng=eng, signal=tone, cf = cf, fs=fs)

    return vihc

def run_ihc(eng, signal, cf, fs, cohc = 1., cihc = 1., species = 'human'):
    """
    :param eng:
    :param signal:
    :param cf:
    :param fs:
    :param cohc:
    :param cihc:
    :param species:
    :return: <class> matlab.double
    """
    # I used a dictionary literal in the line below. If  I didn't add the [species] dictionary literal immediately
    # after the dictionary, I would need to put the argument inside the function as species_map[species].
    # Dictionary literals means " look up the value in the dictionary with the key `species`.
    species_map = {'cat': 1, 'human': 2.0, 'human_glasberg1990': 3}[species]
    # # Ensure correct type and shape
    # if isinstance(signal, np.ndarray):
    #     if signal.ndim > 1:
    #         signal = signal.flatten()  # Flatten to 1D
    #     signal = matlab.double(signal.tolist())
    print(type(signal))
    print(len(signal))
    reptime = float(len(signal)/fs)
    print(reptime)
    print(len(signal))
    vihc = eng.model_IHC_BEZ2018a(signal,cf, 1., 1.0/fs,reptime, cohc, cihc, species_map)
    print(f"[DEBUG] vihc output length: {len(vihc)}")

    return vihc

