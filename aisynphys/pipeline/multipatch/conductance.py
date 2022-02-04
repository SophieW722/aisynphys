# coding: utf8
from __future__ import print_function, division

import numpy as np
from .pipeline_module import MultipatchPipelineModule
from .synapse import SynapsePipelineModule
from .pulse_response import PulseResponsePipelineModule
from .resting_state import RestingStatePipelineModule
from sklearn.linear_model import LinearRegression
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
        db = job['database']
        expt_id = job['job_id']

        expt = db.experiment_from_ext_id(expt_id, session=session)
       
        for pair in expt.pairs.values():
            if pair.has_synapse is not True:
                continue

            # get qc-pass responses in VC at both holding potentials (ie ex_qc_pass and in_qc_pass)
            # require that both holding potentials be present to calculate reversal
            vc_pulses = vc_pr_query(pair, db, session).all()
            
            sorted_vc_pulses = sort_responses(vc_pulses)
            qc_pass_70 = sorted_vc_pulses[('vc', -70)]['qc_pass']
            qc_pass_55 = sorted_vc_pulses[('vc', -55)]['qc_pass']

            if len(qc_pass_70) < 1 or len(qc_pass_55) < 1:
                continue
            
            pulse_responses = qc_pass_70 + qc_pass_55
            adj_baseline = np.array([pr.recording.patch_clamp_recording.access_adj_baseline_potential for pr in pulse_responses if pr.pulse_response_fit is not None])
            pr_amps = np.array([pr.pulse_response_fit.fit_amp for pr in pulse_responses if pr.pulse_response_fit is not None]) 
            if len(adj_baseline) < 1 or len(pr_amps) < 1:
                continue
            model = LinearRegression().fit(adj_baseline.reshape((-1,1)), pr_amps)
            slope = model.coef_ 
            intercept = model.intercept_
            reversal = -intercept / slope

            rec = db.Conductance(
                synapse_id=pair.synapse.id,
                reversal_potential=reversal
            )

            psp_amp = pair.synapse.psp_amplitude
            if psp_amp is None:
                continue
            # get pulse responses that contributed to resting state PSP amp and get average baseline potential
            ic_pr_ids = pair.synapse.resting_state_fit.ic_pulse_ids[0].tolist()
            ic_prs = session.query(db.PulseResponse).filter(db.PulseResponse.id.in_(ic_pr_ids)).all()
            avg_baseline_potential = np.nanmean([pr.recording.patch_clamp_recording.baseline_potential for pr in ic_prs])
            
            
            if psp_amp is not None:
                eff_cond = effective_conductance = (0 - psp_amp) / (reversal - avg_baseline_potential) # m = (y2 - y1) / (x2 - x1)
                target_holding = -55e-3 if pair.synapse.synapse_type == 'in' else -70e-3
                adj_psp_amplitude = eff_cond * target_holding - eff_cond * reversal # y = m * x(target_voltage) + b, b = -m * x2 (reversal)

                rec.ideal_holding_potential = target_holding
                rec.adj_psp_amplitude = adj_psp_amplitude
                rec.effective_conductance = eff_cond
                rec.avg_baseline_potential = avg_baseline_potential
            
            session.add(rec)

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