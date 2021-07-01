### Utility functions for processing intrinsic ephys using IPFX

import numpy as np
from ipfx.data_set_features import extractors_for_sweeps
from ipfx.stimulus_protocol_analysis import LongSquareAnalysis
from ipfx.sweep import Sweep, SweepSet
from ipfx.error import FeatureError
from ipfx.chirp_features import extract_chirp_fft
from ipfx.bin.features_from_output_json import get_complete_long_square_features
from .nwb_recordings import get_pulse_times

import logging
logger = logging.getLogger(__name__)

def get_chirp_features(recordings, cell_id=''):
    errors = []
    if len(recordings) == 0:
        errors.append('No chirp sweeps for cell %s' % cell_id)
        return {}, errors
            
    sweep_list = []
    for rec in recordings:
        sweep = MPSweep(rec)
        if sweep is not None:
            sweep_list.append(sweep)
    
    if len(sweep_list) == 0:
        errors.append('No chirp sweeps passed qc for cell %s' % cell_id)
        return {}, errors

    sweep_set = SweepSet(sweep_list) 
    try:
        chirp_features = extract_chirp_fft(sweep_set, min_freq=1, max_freq=15)
    except FeatureError as exc:
        logger.warning(f'Error processing chirps for cell {cell_id}: {str(exc)}')
        errors.append('Error processing chirps for cell %s: %s' % (cell_id, str(exc)))
        results = {}
    
    return chirp_features, errors

def get_long_square_features(recordings, cell_id=''):
    errors = []
    if len(recordings) == 0:
        errors.append('No long pulse sweeps for cell %s' % cell_id)
        return {}, errors

    min_pulse_dur = np.inf
    sweep_list = []
    for rec in recordings:
        pulse_times = get_pulse_times(rec)
        if pulse_times is None:
            continue
        
        # pulses may have different durations as well, so we just use the smallest duration
        start, end = pulse_times
        min_pulse_dur = min(min_pulse_dur, end-start)
        
        sweep = MPSweep(rec, -start)
        if sweep is not None:
            sweep_list.append(sweep)
    
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
            # TODO: maybe log this error
            return None
        holding = holding[0].amplitude
        i = (cmd.data - holding) * 1e12   # convert to pA with holding current removed
        srate = pri.sample_rate
        sweep_num = rec.parent.key
        # modes 'ic' and 'vc' should be expanded
        clamp_mode = "CurrentClamp" if rec.clamp_mode=="ic" else "VoltageClamp"

        Sweep.__init__(self, t, v, i, clamp_mode, srate, sweep_number=sweep_num)