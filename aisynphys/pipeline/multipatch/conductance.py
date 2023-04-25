import functools
import numpy as np
import uncertainties
from scipy.optimize import curve_fit

from .pipeline_module import MultipatchPipelineModule
from .pulse_response import PulseResponsePipelineModule
from .resting_state import RestingStatePipelineModule
from ...avg_response_fit import sort_responses


class ConductancePipelineModule(MultipatchPipelineModule):
    """ Measure the effective conductance of a chemical synapse using reversal potential calculated from VC.
    Additionally calculate the predicted psp amplitude at the target holding potential for that connection type,
    -55 mV for inhibitory connections and -70 mV for excitatory. Currently this is limited to monosynaptic responses.
    """

    name = 'conductance'
    dependencies = [PulseResponsePipelineModule, RestingStatePipelineModule]
    table_group = ['conductance']
    
    @classmethod
    def create_db_entries(cls, job, session):
        errors = []
        db = job['database']
        expt_id = job['job_id']

        expt = db.experiment_from_ext_id(expt_id, session=session)
       
        for pair in expt.pairs.values():
            # load voltage clamp data (or bail out)
            result = get_raw_vc_data(pair, db, session)
            if result['error'] is not None:
                errors.append(f"pair {pair}: {result['error']}")
                continue
            pr_amps = result['pr_amps']
            adj_baseline = result['adj_baseline']

            # calculate conductance and reversal potential
            try:
                syn_type = pair.synapse.synapse_type
                fixed_reversal = None if syn_type == 'in' else 10e-3
                conductance, reversal, r2 = calculate_conductance(pr_amps, adj_baseline, syn_type, fixed_reversal=fixed_reversal)
            except RuntimeError:
                errors.append(f"pair {pair}: linear regression failed")
                print(f"pair {pair}: linear regression failed -----------------------------------------")
                continue

            rec = db.Conductance(
                synapse_id=pair.synapse.id,
                reversal_potential=reversal.n,
                effective_conductance=conductance.n,
                meta={'reversal_std': reversal.s, 'conductance_std': conductance.s, 'fit_r_squared': r2},
            )
            session.add(rec)

            psp_amp = pair.synapse.psp_amplitude
            if psp_amp is None:
                errors.append(f"pair {pair}: has no PSP amplitude")
                continue

            # get pulse responses that contributed to resting state PSP amp and get average baseline potential
            ic_pr_ids = pair.synapse.resting_state_fit.ic_pulse_ids[0].tolist()
            ic_prs = session.query(db.PulseResponse).filter(db.PulseResponse.id.in_(ic_pr_ids)).all()
            avg_baseline_potential = np.nanmean([pr.recording.patch_clamp_recording.baseline_potential for pr in ic_prs])
            
            eff_cond = conductance.n
            target_holding = -55e-3 if pair.synapse.synapse_type == 'in' else -70e-3
            adj_psp_amplitude = eff_cond * target_holding - eff_cond * reversal.n # y = m * x(target_voltage) + b, b = -m * x2 (reversal)

            rec.ideal_holding_potential = target_holding
            rec.adj_psp_amplitude = adj_psp_amplitude
            rec.avg_baseline_potential = avg_baseline_potential

        return errors

    def job_records(self, job_ids, session): 
        """Return a list of records associated with a list of job IDs.
        
        This method is used by drop_jobs to delete records for specific job IDs.
        """
        db = self.database
        q = session.query(db.Conductance)
        q = q.filter(db.Conductance.synapse_id==db.Synapse.id)
        q = q.filter(db.Synapse.pair_id==db.Pair.id)
        q = q.filter(db.Pair.experiment_id==db.Experiment.id)
        q = q.filter(db.Experiment.ext_id.in_(job_ids))
        return q.all()


def vc_pr_query(pair, db, session):
    q = session.query(db.PulseResponse)
    q = q.join(db.PatchClampRecording, db.PulseResponse.recording_id==db.PatchClampRecording.recording_id)
    q = q.filter(db.PulseResponse.pair_id==pair.id)
    q = q.filter(db.PatchClampRecording.clamp_mode=='vc')
   
    return q


def get_raw_vc_data(pair, db, session):
    """Return a dict containing PSC amplitudes and baseline potentials (adjusted for access resistance) from VC recordings for a Pair.
    
    If data are not available for this Pair, then the return value will have ['error'] set to an error message.
    """
    result = {'error': None}
    if pair.has_synapse is not True:
        result['error'] = "Cell pair has no synapse"
        return result

    # get qc-pass responses in VC at both holding potentials (ie ex_qc_pass and in_qc_pass)
    # require that both holding potentials be present to calculate reversal
    vc_pulses = vc_pr_query(pair, db, session).all()
    
    sorted_vc_pulses = sort_responses(vc_pulses)
    qc_pass_70 = sorted_vc_pulses[('vc', -70)]['qc_pass']
    qc_pass_55 = sorted_vc_pulses[('vc', -55)]['qc_pass']
    if len(qc_pass_70) < 10 or len(qc_pass_55) < 10:
        result['error'] = "Insufficient QC-passed PSC data"
        return result

    pr_with_fit_70 = [pr for pr in qc_pass_70 if pr.pulse_response_fit is not None]
    pr_with_fit_55 = [pr for pr in qc_pass_55 if pr.pulse_response_fit is not None]
    if len(pr_with_fit_70) < 1 or len(pr_with_fit_55) < 1:
        result['error'] = "Insufficient PSC data with fit amplitudes"
        return result

    pulse_responses = pr_with_fit_70 + pr_with_fit_55
    adj_baseline = np.array([pr.recording.patch_clamp_recording.access_adj_baseline_potential for pr in pulse_responses])
    pr_amps = np.array([pr.pulse_response_fit.fit_amp for pr in pulse_responses])
                
    result['pr_amps'] = pr_amps
    result['adj_baseline'] = adj_baseline
    result['pulse_responses'] = pulse_responses
    return result


def ipsc_amp_fn(holding, conductance, reversal):
    return conductance * (holding - reversal)


def calculate_conductance(amps, adj_holding, syn_type, fixed_reversal=None):
    """Given IPSC amplitudes and holding potentials, return an estimated
    conductance, reversal potential, and r^2 value for the linear regression.

    Returns
    -------
        conductance : uncertainties.ufloat
            Conductance value (.n for nominal value, .s for stdev)
        reversal : uncertainties.ufloat
            Reversal potential  (.n for nominal value, .s for stdev)
        r2 : float
            Fit R^2
    """
    p0 = {
        'in': (10e-9, -70e-3),
        'ex': (1e-9, 10e-3),
    }[syn_type]

    if fixed_reversal is None:
        # fit both conductance and reversal
        curve_fit_fn = ipsc_amp_fn
    else:
        # fit conductance with reversal fixed
        p0 = p0[:1]
        curve_fit_fn = functools.partial(ipsc_amp_fn, reversal=fixed_reversal)

    popt, pcov = curve_fit(curve_fit_fn, adj_holding, amps, p0=p0)

    r2 = 1 - (sum((amps - curve_fit_fn(adj_holding, *popt))**2) / ((len(amps) - 1) * np.var(amps, ddof=1)))

    corval = uncertainties.correlated_values(popt, pcov)
    if fixed_reversal is None:
        conductance, reversal = corval
        return conductance, reversal, r2
    else:
        conductance = corval[0]
        return conductance, uncertainties.ufloat(float(fixed_reversal), 0.0), r2
