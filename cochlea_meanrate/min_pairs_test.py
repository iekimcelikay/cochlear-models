from __future__ import print_function, division
import numpy as np
from cochlea.zilany2014 import calc_cfs
freq_range = (125,2500,20)
db_range = (50,90,10)

def generate_stimuli_params(fr, _unused, dr):
    dbs = np.arange(*dr)
    try:
        freqs = calc_cfs(fr, species='human')
    except TypeError:
        freqs = calc_cfs(fr)
    return dbs, freqs

def prepare_stimuli_pairs():
    dbs, freqs = generate_stimuli_params(freq_range, None, db_range)
    return [(float(f), float(d)) for d in dbs for f in freqs]

if __name__ == '__main__':
    pairs = prepare_stimuli_pairs()
    print("Pairs:", len(pairs))
    for i,(f,d) in enumerate(pairs[:10]):
        print(i, f, d)