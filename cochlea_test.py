#

#[Zilany2014]	Zilany, M. S., Bruce, I. C., & Carney, L. H. (2014).
# Updated parameters and expanded simulation options for a model of the auditory periphery.
# The Journal of the Acoustical Society of America, 135(1), 283-286.

import cochlea
import thorns as th
import thorns.waves as wv
import moch
import numpy as np
import subthalamic
import scipy.io
# Generate sound

fs = 100e3
sound = wv.ramped_tone(
    fs=fs,
    freq=1000,
    duration=0.15,
    dbspl=50
)

total_dur = 0.2
isi= 0.025

total_smp = total_dur * fs
isi_smp= np.zeros(int(isi * fs))
tone_with_silence = np.concatenate((isi_smp, sound, isi_smp))

# for run_zilany2014 function I Need:
#   - anf_nums
par = {'periphFs': 100000,
       'cochChanns': (125, 10000, 30),
       }
moch.peripheral(tone_with_silence, par, fs = -1)





# mochnp.array([])
