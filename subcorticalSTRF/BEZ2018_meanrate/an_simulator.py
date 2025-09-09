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
        #vihc = model_IHC_BEZ2018a(pin,CF,nrep,dt,reptime,cohc,cihc,species);
        # reptime should be equal or longer than the duration of input sound wave. 
        species_map = {'cat': 1, 'human': 2.0, 'human_glasberg1990': 3}
        if species not in species_map:
            raise ValueError(f"Unknown species: {species}")
        species_code = species_map[species]
        reptime = float(len(signal) / self.fs) 
        return self.eng.model_IHC_BEZ2018a(signal, cf, 1.0, 1.0 / self.fs, reptime, cohc, cihc, species_code)

    def run_ihc(self, chan):
        fs = chan['fs']
        cf = chan['cf']
        tone = chan['signal']
        print(f"[DEBUG] Running IHC with: CF={cf}, tone RMS={np.sqrt(np.mean(tone ** 2)):.2f}")
        vihc = self._run_ihc(signal=tone, cf=cf)
        return ensure_matlab_row_vector(vihc)

    def run_synapse(self, vihc, cf, cf_idx, freq, db, tone_idx):
        """
        
        psth is the peri-stimulus time histogram (PSTH) (or a spike train if nrep = 1)

        vihc is the inner hair cell (IHC) relative transmembrane potential (in volts)
        CF is the characteristic frequency of the fiber in Hz
        nrep is the number of repetitions for the psth
        dt is the binsize in seconds, i.e., the reciprocal of the sampling rate (see instructions below)
        noiseType is for "variable" or "fixed (frozen)" fGn: 1 for variable fGn and 0 for fixed (frozen) fGn
        implnt is for "approxiate" or "actual" implementation of the power-law functions: "0" for approx. and "1" for actual implementation
        spont is the spontaneous firing rate in /s
        tabs is the absolute refractory period in s
        trel is the baselines mean relative refractory period in s
        expliketype sets the type of exp-like function in IHC -> synapse mapping: 1 for shifted softplus (preferred); 0 for no expontential-like function; 2 for shifted exponential; 3 for shifted Boltmann
        """

        # [psth,meanrate,varrate,synout,trd_vector,trel_vector] = model_Synapse_BEZ2018a(vihc,CF,nrep,dt,noiseType,implnt,spont,tabs,trel,expliketype)
        # synapse_results_fiber = run_fiber_synapse_batch_parfor(vihc,cf, cf_idx,freq, db, tone_idx, nrep, dt, noiseType, implnt, expliketype, fiber_filename)
        tdres = 1.0 / self.fs
        noiseType = 1.0 # 1.0 for variable fGN, and 0 for fixed fGN
        implnt = 1.0 # 1.0 for actual implementation, 0 for approximate
        expliketype = 1.0 # 1.0 for shifted softplus (preferred)
        num_lsr, num_msr, num_hsr = self.num_ANF
        return self.eng.run_fiber_synapse_batch_parfor(vihc, cf, cf_idx, freq, db, tone_idx, 1.0, tdres, noiseType, implnt, expliketype, self.fibers_filename, nargout=1)

    def run_channel(self, chan, tone_idx):
        vihc = self.run_ihc(chan)
        return self.run_synapse(vihc,
                                cf = chan['cf'],
                                cf_idx = chan['cf_idx'],
                                freq = chan['freq'],
                                db = chan['db'],
                                tone_idx = tone_idx)

