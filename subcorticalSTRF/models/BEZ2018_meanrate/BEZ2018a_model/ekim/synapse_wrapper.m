%
%     psth = model_Synapse_BEZ2018a(vihc,CF,nrep,dt,noiseType,implnt,spont,tabs,trel,expliketype);
%
% psth is the peri-stimulus time histogram (PSTH) (or a spike train if nrep = 1)
%
% vihc is the inner hair cell (IHC) relative transmembrane potential (in volts)
% CF is the characteristic frequency of the fiber in Hz
% nrep is the number of repetitions for the psth
% dt is the binsize in seconds, i.e., the reciprocal of the sampling rate (see instructions below)
% noiseType is for "variable" or "fixed (frozen)" fGn: 1 for variable fGn and 0 for fixed (frozen) fGn
% implnt is for "approxiate" or "actual" implementation of the power-law functions: "0" for approx. and "1" for actual implementation
% spont is the spontaneous firing rate in /s
% tabs is the absolute refractory period in s
% trel is the baselines mean relative refractory period in s
% expliketype sets the type of exp-like function in IHC -> synapse mapping: 1 for shifted softplus (preferred); 0 for no expontential-like function; 2 for shifted exponential; 3 for shifted Boltmann
% 

implmnt = 1; % Actual
expliketype = 1; % Softplus mapping
noiseType = 0; % fixed gaussian noise
dt = 1/100e3;
nrep = 1;

% Take the ihc output from Python 
for fiber_index = 1:length(fiber_list)

    psth_raw= model_Synapse_BEZ2018a(vihc,CF,nrep,dt,noiseType,implnt,spont,tabs,trel,expliketype);

end

