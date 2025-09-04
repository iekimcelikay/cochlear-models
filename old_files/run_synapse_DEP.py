#
# We would want to define spont, tabs, and trel for each CF. But we would want to define these once, before callng this
#  function.

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


def ensure_matlab_row_vector(arr):
    if isinstance(arr, matlab.double):
        # Check if it's a column vector and transpose it
        if len(arr) > 0 and isinstance(arr[0], list) and len(arr[0]) == 1:
            return matlab.double([[x[0] for x in arr]])  # flatten column to row
        return arr
    elif isinstance(arr, np.ndarray):
        return matlab.double([arr.tolist()])
    elif isinstance(arr, list):
        return matlab.double([arr])
    else:
        raise TypeError(f"Unsupported vihc type: {type(arr)}")

def _run_synapse_channel(args, eng):
    cf_idx = args['cf_idx']
    cf = args['cf']
    tone_idx = args['tone_idx']
    freq = args['freq']
    db = args['db']
    fs = args['fs']
    vihc = args['vihc']
    anf_type = args['anf_type']
    fiber_idx = args['fiber_idx']
    spont = args['spont']
    tabs = args['tab']
    trels = args['trel']

    # Run Synapse
    psth, meanrate, varrate, synout, trd_vector, trel_vector = run_synapse(eng=eng, cf_idx = cf_idx,cf=cf, tone_idx=tone_idx, freq=freq, db=db, fs=fs, vihc=vihc, anf_type=anf_type, fiber_idx = fiber_idx, spont=spont, tabs=tabs, trel=trels)

    return {
        'cf_idx': cf_idx,
        'cf': cf,
        'tone_idx': tone_idx,
        'freq': freq,
        'db': db,
        'fiber_idx': fiber_idx,
        'anf_type': anf_type,
        'spont': spont,
        'tabs': tabs,
        'trel': trels,
        'psth': psth, # peri stimulus time histogram (or a spike train if nrep =1)
        'meanrate': meanrate, # analytical estimate of the mean firing rate in /s for each time bin from c code:"/* estimated instantaneous mean rate */"
        'varrate': varrate, # analytical estimate of the variance in firing rate in /s for each time bin / from c code: " /* estimated instananeous variance in the discharge rate */
        'synout': synout, # synapse output rate in /s for each time bin (before the effects of redocking are considered)
        'trd_vector': trd_vector, # is a vector of mean redocking time in s for each time bin
        'trel_vector': trel_vector, # a vector of mean relative refractory period in s for each time bin
    }

#     [psth,meanrate,varrate,synout,trd_vector,trel_vector] = model_Synapse_BEZ2018a(vihc,CF,nrep,dt,noiseType,implnt,spont,tabs,trel,expliketype);
def run_synapse(eng, cf_idx, cf, tone_idx, freq,  db, fs, vihc, anf_type, fiber_idx,  spont, tabs, trel, noiseType = 'fixed', implnt='actual', expliketype= 'softplus'):

        # I used a dictionary literal in the line below. If  I didn't add the [species] dictionary literal immediately
        # after the dictionary, I would need to put the argument inside the function as species_map[species].
        # Dictionary literals means " look up the value in the dictionary with the key `species`.
        noisetype_map = {'random':0.0, 'fixed': 1.0}[noiseType]
        implnt_map = {'approximate': 0.0, 'actual':1.0}[implnt]
        expliketype_map = {'noexp': 0.0, 'softplus': 1.0, 'shifted-exp': 2.0, 'shifted-boltmann': 3.0}[expliketype]
        dt = float(1.0/fs)
        nrep = float(1)

        # Convert vihc numpy array to matlab.double (column vector)
        vihc = ensure_matlab_row_vector(vihc)

        # # Ensure correct type and shape
        # if isinstance(signal, np.ndarray):
        #     if signal.ndim > 1:
        #         signal = signal.flatten()  # Flatten to 1D
        #     signal = matlab.double(signal.tolist())

        print(f"[DEBUG][Synapse] tone_idx={tone_idx}, fiber_idx={fiber_idx}, dB={db}, CF={cf:.1f} Hz, spont={spont:.1f}, tabs={tabs:.3f}, trel={trel:.3f}, anf_type={anf_type}")

        psth, meanrate, varrate, synout, trd_vector, trel_vector = eng.model_Synapse_BEZ2018a(vihc,cf, nrep, dt, noisetype_map, implnt_map, spont, tabs, trel, expliketype_map, nargout = 6)

        return psth, meanrate, varrate, synout, trd_vector, trel_vector

