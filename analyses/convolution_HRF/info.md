## Folder information, file structure, used functions
author: Ekim Celikay
date: 22/10/2025
last update: 

- This folder is created on 22.10.2025
- It is for discrete convolution with canonical HRF, for the spike-rate outputs of cochlea and BEZ2018 models. 
- I will use a canonical HRF from nipye.
- The first script that is being written: `convolutionHRF_draft_221025.py`. 



The modules that will be used:
`modalities.fmri.hemodynamic_models`: This module is for canonical hrf specification. Includes SPM, glover hrfs
and finite impulse response (FIR) models.
- API link: https://nipy.org/nipy/api/generated/nipy.modalities.fmri.hemodynamic_models.html
This module already provides a convolve with regressors function, but Alex previously told me to use the basic
`convolve` function that is from `scipy` or `numpy`. I found the aforementioned function in `scipy` library. 


`nipy.modalities.fmri.hemodynamic_models.compute_regressor
 (exp_condition, hrf_model, frametimes, con_id='cond', oversampling=16, fir_delays=None, min_onset=-24)¶`
This is the main function to convolve regressors with hrf model

> Parameters:
    exp_condition: descriptor of an experimental condition
    hrf_model: string, the hrf model to be used. Can be chosen among:
    ‘spm’, ‘spm_time’, ‘spm_time_dispersion’, ‘canonical’, ‘canonical_derivative’, ‘fir’
    frametimes: array of shape (n):the sought
    con_id: string, optional identifier of the condition
    oversampling: int, optional, oversampling factor to perform the convolution
    fir_delays: array-like of int, onsets corresponding to the fir basis
    min_onset: float, optional
    minimal onset relative to frametimes[0] (in seconds) events that start before frametimes[0] + min_onset are not considered

> Returns:
    creg: array of shape(n_scans, n_reg): computed regressors sampled
    at frametimes 
>   reg_names: list of strings, corresponding regressor names

So I should set parameters to: 
exp_condition: 'puretone200ms'
hrf_model: 'spm' or 'canonical'
frametimes: ? -> I guess this should be my time vector ?
oversampling: 1
fir_delays = None,

But I can rather import an HRF, and use the convolve of scipy. 

- `nipy.modalities.fmri.hemodynamic_models.glover_hrf(tr, oversampling=16, time_length=32.0, onset=0.0)`
Implementation of the Glover hrf model

> Parameters:
    tr: float, scan repeat time, in seconds
    oversampling: int, temporal oversampling factor, optional
    time_length: float, hrf kernel length, in seconds
    onset: float, onset of the response
    Returns:
    hrf: array of shape(length / tr * oversampling, float),
    hrf sampling on the oversampled time grid