# simulator_utils.py

import numpy as np

import matlab
from numpy.random import rand, randn, seed
import pprint
import numpy as np
from scipy.io import savemat, loadmat
import os

def spike_mean_rate():
    return

def spike_timing():
    return


def calc_cfs(cf, species):
    if np.isscalar(cf):
        cfs = [float(cf)]

    elif isinstance(cf, tuple) and ('cat' in species):
        # Based on GenerateGreenwood_CFList() from DSAM
        # Liberman (1982)
        aA = 456
        k = 0.8
        a = 2.1

        freq_min, freq_max, freq_num = cf

        xmin = np.log10(freq_min / aA + k) / a
        xmax = np.log10(freq_max / aA + k) / a

        x_map = np.linspace(xmin, xmax, freq_num)
        cfs = aA * ( 10**( a*x_map ) - k)

    elif isinstance(cf, tuple) and ('human' in species):
        # Based on GenerateGreenwood_CFList() from DSAM
        # Liberman (1982)
        aA = 165.4
        k = 0.88
        a = 2.1

        freq_min, freq_max, freq_num = cf

        xmin = np.log10(freq_min / aA + k) / a
        xmax = np.log10(freq_max / aA + k) / a

        x_map = np.linspace(xmin, xmax, freq_num)
        cfs = aA * ( 10**( a*x_map ) - k)

    elif isinstance(cf, list) or isinstance(cf, np.ndarray):
        cfs = cf

    else:
        raise RuntimeError("CF must be a scalar, a tuple or a list.")

    return cfs

def greenwood_human(cf):
        if np.isscalar(cf):
            cfs = [float(cf)]

        elif isinstance(cf, tuple):
            #Based on Greenwood (1990) parameters, function based on 'calc_cfs' from cochlea package.
            aA = 165.4
            k = 0.88
            a = 2.1

            freq_min, freq_max, freq_num = cf

            xmin = np.log10(freq_min / aA + k) / a
            xmax = np.log10(freq_max / aA + k) / a

            x_map = np.linspace(xmin, xmax, freq_num)
            cfs = aA * ( 10**( a*x_map ) - k)
        elif isinstance(cf, list) or isinstance(cf, np.ndarray):
            cfs = cf
        else:
            raise RuntimeError("CF must be a scalar, a tuple or a list.")

        return cfs



def generateANpopulation(numcfs,numsponts):
    """
    Generate spontaneous rates (sponts), absolute refractory periods (tabss),
    and relati
    :param numcfs: (int): Number of characteristic frequencies.
    :param numsponts: (List or tuple of int): [num_low, num_med, num_high] number of fibers per type.
    :return:
    """

    tabsmax = 1.5*461e-6
    tabsmin = 1.5*139e-6
    trelmax = 894e-6
    trelmin = 131e-6

    sponts = {}
    tabss = {}
    trels = {}
    # Generate sponts, tabss & trels for LS fibers (fiberType = 1)
    #sponts['low'] = min(max(0.1 + 0.1 * randn(numcfs,numsponts(1)),1e-3),0.2)
    ## min(max(value, lower_limit), upper_limit) in MATLAB
    # if value < lower_limit, it becomes Lower_limit etc.

    # sponts['type'] = np.clip( mean + std * randn(numcfs, low_fiber_number)), min_limit, max_limit)
    sponts['lsr'] = np.clip(0.1 + 0.1 * randn(numcfs, numsponts[0]), 1e-3, 0.2)
    refrand = rand(numcfs, numsponts[0])
    tabss['lsr'] = (tabsmax - tabsmin) * refrand + tabsmin
    trels['lsr'] = (trelmax - trelmin) * refrand + trelmin

    # Generate, sponts, tabss & trels for MS fibers (fiberType = 2)
    sponts['msr'] = np.clip(4 + 4 * randn(numcfs, numsponts[1]), 0.2, 18)
    refrand = rand(numcfs, numsponts[1])
    tabss['msr'] = (tabsmax - tabsmin) * refrand + tabsmin
    trels['msr'] = (trelmax - trelmin) * refrand + trelmin

    # HS fibers (fiberType = 3)
    sponts['hsr'] = np.clip(70 + 30 * randn(numcfs, numsponts[2]), 18, 180)
    refrand = rand(numcfs, numsponts[2])
    tabss['hsr'] = (tabsmax - tabsmin) * refrand + tabsmin
    trels['hsr'] = (trelmax - trelmin) * refrand + trelmin

    fname = "ANpopulation.npz"
    np.savez(fname, sponts=sponts, tabss=tabss, trels=trels)
    return sponts, tabss, trels


def generateANpopulation_separate_arrays(numcfs, numsponts, filename="ANpopulation.mat"):

    if filename and os.path.exists(filename):
        print(f"[INFO] Loading existing AN population from: {filename}")
        return loadmat(filename)

    print(f"[INFO] Generating new AN population for LSR:MSR:HSR = {numsponts} and saving to {filename}")
    tabsmax = 1.5 * 461e-6
    tabsmin = 1.5 * 139e-6
    trelmax = 894e-6
    trelmin = 131e-6

    # LSR fibers
    sponts_lsr = np.clip(0.1 + 0.1 * np.random.randn(numcfs, numsponts[0]), 1e-3, 0.2)
    refrand = np.random.rand(numcfs, numsponts[0])
    tabss_lsr = (tabsmax - tabsmin) * refrand + tabsmin
    trels_lsr = (trelmax - trelmin) * refrand + trelmin

    # MSR fibers
    sponts_msr = np.clip(4 + 4 * np.random.randn(numcfs, numsponts[1]), 0.2, 18)
    refrand = np.random.rand(numcfs, numsponts[1])
    tabss_msr = (tabsmax - tabsmin) * refrand + tabsmin
    trels_msr = (trelmax - trelmin) * refrand + trelmin

    # HSR fibers
    sponts_hsr = np.clip(70 + 30 * np.random.randn(numcfs, numsponts[2]), 18, 180)
    refrand = np.random.rand(numcfs, numsponts[2])
    tabss_hsr = (tabsmax - tabsmin) * refrand + tabsmin
    trels_hsr = (trelmax - trelmin) * refrand + trelmin

    # Save to .mat
    mat_dict = {
        'sponts_lsr': sponts_lsr,
        'tabss_lsr': tabss_lsr,
        'trels_lsr': trels_lsr,
        'sponts_msr': sponts_msr,
        'tabss_msr': tabss_msr,
        'trels_msr': trels_msr,
        'sponts_hsr': sponts_hsr,
        'tabss_hsr': tabss_hsr,
        'trels_hsr': trels_hsr
    }
    savemat(filename, mat_dict)

    return mat_dict  # Optional: for inspection


def get_fiber_struct_array_old(cf_idx, sponts, tabss, trels, num_ANF):
    anf_types = ['lsr', 'msr', 'hsr']
    fieldnames = ['anf_type', 'spont', 'tabs', 'trel']
    structs = []

    for anf_type, n_fibers in zip(anf_types, num_ANF):
        for fiber_type_idx in range(n_fibers):
            spont = float(sponts[anf_type][cf_idx, fiber_type_idx])
            tabs = float(tabss[anf_type][cf_idx, fiber_type_idx])
            trel = float(trels[anf_type][cf_idx, fiber_type_idx])


            structs.append({
                'anf_type': str(anf_type),
                'spont': spont,
                'tabs': tabs,
                'trel': trel
                })

            print(f"[DEBUG] {anf_type} fiber {fiber_type_idx} at cf_idx {cf_idx} => spont={spont:.2f}, tabs={tabs:.4f}, trel={trel:.4f}")

    return structs


def get_fiber_struct_array(cf_idx, sponts, tabss, trels, num_ANF, eng):
    anf_types = ['lsr', 'msr', 'hsr']

    # Start a MATLAB struct array manually
    total_fibers = sum(num_ANF)
    eng.eval(f"fibers({total_fibers}) = struct('anf_type', '', 'spont', 0, 'tabs', 0, 'trel', 0);", nargout=0)

    fiber_idx = 1  # MATLAB is 1-indexed

    for anf_type, n_fibers in zip(anf_types, num_ANF):
        for local_idx in range(n_fibers):
            spont = float(sponts[anf_type][cf_idx, local_idx])
            tabs = float(tabss[anf_type][cf_idx, local_idx])
            trel = float(trels[anf_type][cf_idx, local_idx])
            anf_str = anf_type  # Python string is okay for MATLAB char

            # Assign to MATLAB struct
            eng.eval(f"fibers({fiber_idx}).anf_type = '{anf_str}';", nargout=0)
            eng.eval(f"fibers({fiber_idx}).spont = {spont};", nargout=0)
            eng.eval(f"fibers({fiber_idx}).tabs = {tabs};", nargout=0)
            eng.eval(f"fibers({fiber_idx}).trel = {trel};", nargout=0)

            print(f"[DEBUG] Fiber {fiber_idx} ({anf_str}) => spont={spont:.2f}, tabs={tabs:.6f}, trel={trel:.6f}")
            fiber_idx += 1


    

def get_fiber_params(cf_idx, sponts, tabss, trels, num_ANF):

    anf_types = ['lsr', 'msr', 'hsr']
    fiber_params = []

    for anf_type, n_fibers in zip(anf_types, num_ANF):
        for fiber_type_idx in range(n_fibers):
            spont = sponts[anf_type][cf_idx, fiber_type_idx]
            tabs = tabss[anf_type][cf_idx, fiber_type_idx]
            trel = trels[anf_type][cf_idx, fiber_type_idx]
            fiber_params.append({
                'anf_type': anf_type,
                'spont': spont,
                'tabs': tabs,
                'trel': trel
            })
            print(
        f"[DEBUG] {anf_type} fiber {fiber_type_idx} at cf_idx {cf_idx} => spont={spont:.2f}, tabs={tabs:.4f}, trel={trel:.4f}")

    return fiber_params # Length = 100 for each CF (60,25,15)

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

def keep_ihc_params(chan):
    fs = chan['fs']
    cf = chan['cf']
    tone = chan['signal']
    db = chan['db']
    freq = chan['freq']
    cf_idx = chan['cf_idx']
    i_tone = chan['i_tone']

    return {'fs': fs, 'cf': cf, 'cf_idx': cf_idx, 'tone_freq': freq, 'db': db}

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
    vihc_mtl = ensure_matlab_row_vector(vihc)

    return vihc_mtl


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

    #print(type(signal))
    #print(len(signal))
    reptime = float(len(signal)/fs)

    vihc = eng.model_IHC_BEZ2018a(signal,cf, 1., 1.0/fs,reptime, cohc, cihc, species_map)

    return vihc

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
    tabs = args['tabs']
    trel = args['trel']

    # Run Synapse
    psth, meanrate, varrate, synout, trd_vector, trel_vector = run_synapse(eng=eng, cf_idx = cf_idx,cf=cf, tone_idx=tone_idx, freq=freq, db=db, fs=fs, vihc=vihc, anf_type=anf_type, fiber_idx = fiber_idx, spont=spont, tabs=tabs, trel=trel)

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
        'trel': trel,
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

        return psth
