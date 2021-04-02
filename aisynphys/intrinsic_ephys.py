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
        all_chirp_features = extract_chirp_fft(sweep_set, min_freq=1, max_freq=15)
        results = {
            'chirp_peak_freq': all_chirp_features['peak_freq'],
            'chirp_3db_freq': all_chirp_features['3db_freq'],
            'chirp_peak_ratio': all_chirp_features['peak_ratio'],
            'chirp_peak_impedance': all_chirp_features['peak_impedance'] * 1e9, #unscale from mV/pA,
            'chirp_sync_freq': all_chirp_features['sync_freq'],
            'chirp_inductive_phase': all_chirp_features['total_inductive_phase'],
        }
    except FeatureError as exc:
        logger.warning(f'Error processing chirps for cell {cell_id}: {str(exc)}')
        errors.append('Error processing chirps for cell %s: %s' % (cell_id, str(exc)))
        results = {}
    
    return results, errors

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
    
    results = {
        'rheobase': output.get('rheobase_i', np.nan) * 1e-12, #unscale from pA,
        'fi_slope': output.get('fi_fit_slope', np.nan) * 1e-12, #unscale from pA,
        'input_resistance': output.get('input_resistance', np.nan) * 1e6, #unscale from MOhm,
        'input_resistance_ss': output.get('input_resistance_ss', np.nan) * 1e6, #unscale from MOhm,
        'tau': output.get('tau', np.nan),
        'sag': output.get('sag', np.nan),
        'sag_peak_t': output.get('sag_peak_t', np.nan),
        'sag_depol': output.get('sag_depol', np.nan),
        'sag_peak_t_depol': output.get('sag_peak_t_depol', np.nan),
        
        'ap_upstroke_downstroke_ratio': output.get('upstroke_downstroke_ratio_hero', np.nan),
        'ap_upstroke': output.get('upstroke_hero', np.nan) * 1e-3, #unscale from mV
        'ap_downstroke': output.get('downstroke_hero', np.nan) * 1e-3, #unscale from mV
        'ap_width': output.get('width_hero', np.nan),
        'ap_threshold_v': output.get('threshold_v_hero', np.nan) * 1e-3, #unscale from mV
        'ap_peak_deltav': output.get('peak_deltav_hero', np.nan) * 1e-3, #unscale from mV
        'ap_fast_trough_deltav': output.get('fast_trough_deltav_hero', np.nan) * 1e-3, #unscale from mV

        'firing_rate_rheo': output.get('avg_rate_rheo', np.nan),
        'latency_rheo': output.get('latency_rheo', np.nan),
        'firing_rate_40pa': output.get('avg_rate_hero', np.nan),
        'latency_40pa': output.get('latency_hero', np.nan),
        
        'adaptation_index': output.get('adapt_mean', np.nan),
        'isi_cv': output.get('isi_cv_mean', np.nan),

        'isi_adapt_ratio': output.get('isi_adapt_ratio', np.nan),
        'upstroke_adapt_ratio': output.get('upstroke_adapt_ratio', np.nan),
        'downstroke_adapt_ratio': output.get('downstroke_adapt_ratio', np.nan),
        'width_adapt_ratio': output.get('width_adapt_ratio', np.nan),
        'threshold_v_adapt_ratio': output.get('threshold_v_adapt_ratio', np.nan),
    }
    return results, errors

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