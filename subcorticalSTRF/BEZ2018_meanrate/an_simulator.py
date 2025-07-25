# an_simulator.py

import numpy as np
from matlab_interface import ensure_matlab_row_vector


class AuditoryNerveSimulator:
    def __init__(self, peripheral_fs, num_ANF, eng, fibers_filename):
        self.fs = peripheral_fs
        self.num_ANF = num_ANF
        self.eng = eng
        self.fibers_filename = fibers_filename

    def _run_ihc(self, signal, cf, cohc=1.0, cihc=1.0, species='human'):
        species_map = {'cat': 1, 'human': 2.0, 'human_glasberg1990': 3}
        if species not in species_map:
            raise ValueError(f"Unknown species: {species}")
        species_code = species_map[species]
        reptime = float(1)
        return self.eng.model_IHC_BEZ2018a(signal, cf, 1.0, 1.0 / self.fs, reptime, cohc, cihc, species_code)

    def run_ihc(self, chan):
        fs = chan['fs']
        cf = chan['cf']
        tone = chan['signal']
        print(f"[DEBUG] Running IHC with: CF={cf}, tone RMS={np.sqrt(np.mean(tone ** 2)):.2f}")
        vihc = self._run_ihc(signal=tone, cf=cf)
        return ensure_matlab_row_vector(vihc)

    def run_synapse(self, vihc, cf, cf_idx, freq, db, tone_idx):
        tdres = 1.0 / self.fs
        cohc, cihc = 1.0 , 1.0
        species_code = 2.0
        num_lsr, num_msr, num_hsr = self.num_ANF

        return self.eng.run_fiber_synapse_batch_parfor(vihc, cf, cf_idx, freq, db, tone_idx, 1.0, tdres, cohc, cihc, species_code, self.fibers_filename, nargout=1)

    def run_channel(self, chan, tone_idx):
        vihc = self.run_ihc(chan)
        return self.run_synapse(vihc,
                                cf = chan['cf'],
                                cf_idx = chan['cf_idx'],
                                freq = chan['freq'],
                                db = chan['db'],
                                tone_idx = tone_idx)

