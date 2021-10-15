### Utility functions for processing intrinsic ephys using IPFX

import numpy as np
from ipfx.data_set_features import extractors_for_sweeps
from ipfx.stimulus_protocol_analysis import LongSquareAnalysis
from ipfx.sweep import Sweep, SweepSet
from ipfx.error import FeatureError
from ipfx.chirp_features import extract_chirp_fft
from ipfx.bin.features_from_output_json import get_complete_long_square_features
from .nwb_recordings import get_pulse_times, get_intrinsic_recording_dict
from neuroanalysis.miesnwb import MiesNwb
from itertools import chain

import logging
logger = logging.getLogger(__name__)

def get_chirp_features(recordings, cell_id=''):
    errors = []
    if len(recordings) == 0:
        errors.append('No chirp sweeps for cell %s' % cell_id)
        return {}, errors
            
    sweep_list = []
    for rec in recordings:
        try:
            sweep = MPSweep(rec)
            sweep_list.append(sweep)
        except ValueError:
            continue
    
    if len(sweep_list) == 0:
        errors.append('No chirp sweeps passed qc for cell %s' % cell_id)
        return {}, errors

    sweep_set = SweepSet(sweep_list) 
    try:
        chirp_features = extract_chirp_fft(sweep_set, min_freq=1, max_freq=15)
    except FeatureError as exc:
        logger.warning(f'Error processing chirps for cell {cell_id}: {str(exc)}')
        errors.append('Error processing chirps for cell %s: %s' % (cell_id, str(exc)))
        chirp_features = {}
    
    return chirp_features, errors

def get_long_square_features(recordings, cell_id=''):
    errors = []
    if len(recordings) == 0:
        errors.append('No long pulse sweeps for cell %s' % cell_id)
        return {}, errors

    min_pulse_dur = np.inf
    sweep_list = []
    for rec in recordings:
        try:
            pulse_times = get_pulse_times(rec)
            if pulse_times is None:
                raise ValueError("Pulse times not found for sweep.")
            # pulses may have different durations as well, so we just use the smallest duration
            start, end = pulse_times
            min_pulse_dur = min(min_pulse_dur, end-start)
            sweep = MPSweep(rec, -start)
            sweep_list.append(sweep)
        except ValueError:
            # report these errors?
            continue
    
    if len(sweep_list) == 0:
        errors.append('No long square sweeps passed qc for cell %s' % cell_id)
        return {}, errors

    sweep_set = SweepSet(sweep_list)
    spx, spfx = extractors_for_sweeps(sweep_set, start=0, end=min_pulse_dur)
    lsa = LongSquareAnalysis(spx, spfx, subthresh_min_amp=-200,
                                require_subthreshold=False, require_suprathreshold=False
                                )
    
    try:
        analysis = lsa.analyze(sweep_set)
    except FeatureError as exc:
        err = f'Error running long square analysis for cell {cell_id}: {str(exc)}'
        logger.warning(err)
        errors.append(err)
        return {}, errors
    
    analysis_dict = lsa.as_dict(analysis)
    output = get_complete_long_square_features(analysis_dict) 
    
    return output, errors

def features_from_nwb(filename, channels=None):
    nwb = MiesNwb(filename)
    channel_key = {}
    # need to convert from AD channel to device id
    for sweep in nwb.contents:
        for props in sweep._channel_keys.values():
            channel_key[int(props['AD'])] = int(props['ElectrodeName'])
    if channels is None:
        channels = channel_key.keys()

    records = list()
    for ch in channels:
        dev = channel_key[ch]
        recording_dict = get_intrinsic_recording_dict(nwb, dev)
        
        results = dict(filename=filename, channel=ch, device=dev)
        if 'LP' in recording_dict:
            lp_results, error = get_long_square_features(recording_dict['LP'])
            results.update(lp_results)
        if 'Chirp' in recording_dict:
            chirp_results, error = get_chirp_features(recording_dict['Chirp'])
            results.update({feature+"_chirp": chirp_results.get(feature) for feature in chirp_results.keys()})
        records.append(results)
    return records

def process_file_list(files, channels_list=None):
    if channels_list:
        records = chain(*(features_from_nwb(file, channels) 
                          for file, channels in zip(files, channels_list)))
    else:
        records = chain(*(features_from_nwb(file) for file in files))
    return records

class MPSweep(Sweep):
    """Adapter for neuroanalysis.Recording => ipfx.Sweep
    """
    def __init__(self, rec, t0=0):
        # pulses may have different start times, so we shift time values to make all pulses start at t=0
        pri = rec['primary'].copy(t0=t0)
        cmd = rec['command'].copy()
        t = pri.time_values
        v = pri.data * 1e3  # convert to mV
        holding = [i for i in rec.stimulus.items if i.description=='holding current']
        if len(holding) == 0:
            raise ValueError("Sweep missing holding current data.")
        holding = holding[0].amplitude
        i = (cmd.data - holding) * 1e12   # convert to pA with holding current removed
        srate = pri.sample_rate
        sweep_num = rec.parent.key
        # modes 'ic' and 'vc' should be expanded
        clamp_mode = "CurrentClamp" if rec.clamp_mode=="ic" else "VoltageClamp"

        valid_data = (v != 0) & ~np.isnan(v)
        # not sure where exactly to put this cutoff - maybe need to refine
        if t[valid_data][-1] < 0.5:
            raise ValueError("Incomplete / nan sweep.")

        Sweep.__init__(self, t[valid_data], v[valid_data], i[valid_data], clamp_mode, srate, sweep_number=sweep_num)