# bruce2018_pipeline.py
import pandas as pd
from .simulator_utils import get_fiber_params, _run_ihc_only_channel, _run_synapse_channel


def prepare_channel_args(tones, cfs, peripheral_fs):
    channel_args = []

    for cf_idx, cf in enumerate(cfs):
        for i_tone,((db, frequency), tone) in enumerate(tones.items()):
            channel_args.append({
                'signal': tone,
                'cf': cf,
                'fs': peripheral_fs,
                'db': db,
                'freq': frequency,
                'cf_idx': cf_idx,
                'i_tone': i_tone,
                })
    return channel_args


def run_ihc_stage(channel_args, eng):
    ihc_outputs = []
    for chan in channel_args:
        vihc = _run_ihc_only_channel(chan, eng)
        ihc_outputs.append({**chan, 'vihc': vihc})
    return pd.DataFrame(ihc_outputs)


def prepare_synapse_args(ihc_outputs, sponts, tabss, trels, num_ANF):
    synapse_inputs = []
    for idx in range(len(ihc_outputs)):
        chan = ihc_outputs.iloc[idx]
        cf_idx = chan['cf_idx']
        fiber_params = get_fiber_params(cf_idx, sponts, tabss, trels, num_ANF)

        for fiber_idx, fiber in enumerate(fiber_params):
            synapse_arg = {
                'cf_idx': chan['cf_idx'],
                'cf': chan['cf'],
                'tone_idx': chan['i_tone'],
                'freq': chan['freq'],
                'db': chan['db'],
                'fs': chan['fs'],
                'vihc': chan['vihc'],
                'anf_type': fiber['anf_type'],
                'fiber_idx': fiber_idx,
                'spont': fiber['spont'],
                'tabs': fiber['tabs'],
                'trel': fiber['trel'],
            }
            synapse_inputs.append(synapse_arg)
    return synapse_inputs


def run_synapse_stage(synapse_inputs, eng):
    synapse_results = []
    for args in synapse_inputs:
        syn_result = _run_synapse_channel(args, eng)
        synapse_results.append(syn_result)
    return pd.DataFrame.from_records(synapse_results)


def run_bruce2018(tones, cfs, peripheral_fs, num_ANF, sponts, tabss, trels, eng):
    channel_args = prepare_channel_args(tones, cfs, peripheral_fs)
    ihc_df = run_ihc_stage(channel_args, eng)
    synapse_inputs = prepare_synapse_args(ihc_df, sponts, tabss, trels, num_ANF)
    synapse_df = run_synapse_stage(synapse_inputs, eng)
    return synapse_df, ihc_df