from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.signal as dsp
import pandas as pd
import matplotlib.pyplot as plt

import cochlea
import cochlea.stats
import thorns as th
import thorns.waves as wv



fs = 100e3
cf = 1000
tone = wv.ramped_tone(
    fs=fs,
    freq=1000,
    duration=0.1,
    dbspl=50
)

wv.plot_signal(tone, fs)