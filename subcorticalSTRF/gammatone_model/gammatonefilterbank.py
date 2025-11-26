from brian2hears import *
from brian2 import *
import numpy as np
from subcorticalSTRF.stim_generator.soundgen import SoundGen
from matplotlib import pyplot as plt


class GammatoneFilterbank:
    def __init__(self, cfs, bw_factor=1.0, fs=44100):
        self.cfs = cfs
        self.bw_factor = bw_factor
        self.fs = fs
        self.filterbank = self._create_filterbank()
