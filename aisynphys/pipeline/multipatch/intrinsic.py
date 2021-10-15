# coding: utf8
"""
For generating a table that describes cell intrinisic properties

"""
from __future__ import print_function, division

from neuroanalysis.util.optional_import import optional_import
get_long_square_features, get_chirp_features = optional_import(
    'aisynphys.intrinsic_ephys', ['get_long_square_features', 'get_chirp_features'])

from .pipeline_module import MultipatchPipelineModule
from .experiment import ExperimentPipelineModule
from .dataset import DatasetPipelineModule
from ...nwb_recordings import get_intrinsic_recording_dict, qc_recordings
import numpy as np


class IntrinsicPipelineModule(MultipatchPipelineModule):
    
    name = 'intrinsic'
    dependencies = [ExperimentPipelineModule, 
                    DatasetPipelineModule
                    ]
    table_group = ['intrinsic']

    @classmethod
    def create_db_entries(cls, job, session):
        db = job['database']
        job_id = job['job_id']

        # Load experiment from DB
        expt = db.experiment_from_ext_id(job_id, session=session)
        try:
            assert expt.data is not None
            # this should catch corrupt NWBs
            assert expt.data.contents is not None
        except Exception:
            error = 'No NWB data for this experiment'
            return [error]

        n_cells = len(expt.cell_list)
        errors = []
        for cell in expt.cell_list:
            dev_id = cell.electrode.device_id
            recording_dict = get_intrinsic_recording_dict(expt.data, dev_id)
            for rec_list in recording_dict.values():
                qc_recordings(expt, rec_list)
            
            lp_results, error = get_long_square_features(recording_dict['LP'], cell_id=cell.id)
            errors += error
            chirp_results, error = get_chirp_features(recording_dict['Chirp'], cell_id=cell.id)
            errors += error
            # Write new record to DB
            
            results = {
                'chirp_peak_freq': chirp_results.get('peak_freq', np.nan),
                'chirp_3db_freq': chirp_results.get('3db_freq', np.nan),
                'chirp_peak_ratio': chirp_results.get('peak_ratio', np.nan),
                'chirp_peak_impedance': chirp_results.get('peak_impedance', np.nan) * 1e9, #unscale from mV/pA,
                'chirp_sync_freq': chirp_results.get('sync_freq', np.nan),
                'chirp_inductive_phase': chirp_results.get('total_inductive_phase', np.nan),
                
                'rheobase': lp_results.get('rheobase_i', np.nan) * 1e-12, #unscale from pA,
                'fi_slope': lp_results.get('fi_fit_slope', np.nan) * 1e-12, #unscale from pA,
                'input_resistance': lp_results.get('input_resistance', np.nan) * 1e6, #unscale from MOhm,
                'input_resistance_ss': lp_results.get('input_resistance_ss', np.nan) * 1e6, #unscale from MOhm,
                'tau': lp_results.get('tau', np.nan),
                'sag': lp_results.get('sag', np.nan),
                'sag_peak_t': lp_results.get('sag_peak_t', np.nan),
                'sag_depol': lp_results.get('sag_depol', np.nan),
                'sag_peak_t_depol': lp_results.get('sag_peak_t_depol', np.nan),
                
                'ap_upstroke_downstroke_ratio': lp_results.get('upstroke_downstroke_ratio_hero', np.nan),
                'ap_upstroke': lp_results.get('upstroke_hero', np.nan) * 1e-3, #unscale from mV
                'ap_downstroke': lp_results.get('downstroke_hero', np.nan) * 1e-3, #unscale from mV
                'ap_width': lp_results.get('width_hero', np.nan),
                'ap_threshold_v': lp_results.get('threshold_v_hero', np.nan) * 1e-3, #unscale from mV
                'ap_peak_deltav': lp_results.get('peak_deltav_hero', np.nan) * 1e-3, #unscale from mV
                'ap_fast_trough_deltav': lp_results.get('fast_trough_deltav_hero', np.nan) * 1e-3, #unscale from mV

                'firing_rate_rheo': lp_results.get('avg_rate_rheo', np.nan),
                'latency_rheo': lp_results.get('latency_rheo', np.nan),
                'firing_rate_40pa': lp_results.get('avg_rate_hero', np.nan),
                'latency_40pa': lp_results.get('latency_hero', np.nan),
                
                'adaptation_index': lp_results.get('adapt_mean', np.nan),
                'isi_cv': lp_results.get('isi_cv_mean', np.nan),

                'isi_adapt_ratio': lp_results.get('isi_adapt_ratio', np.nan),
                'upstroke_adapt_ratio': lp_results.get('upstroke_adapt_ratio', np.nan),
                'downstroke_adapt_ratio': lp_results.get('downstroke_adapt_ratio', np.nan),
                'width_adapt_ratio': lp_results.get('width_adapt_ratio', np.nan),
                'threshold_v_adapt_ratio': lp_results.get('threshold_v_adapt_ratio', np.nan),
            }
            conn = db.Intrinsic(cell_id=cell.id, **results)
            session.add(conn)

        return errors

    def job_records(self, job_ids, session):
        """Return a list of records associated with a list of job IDs.
        
        This method is used by drop_jobs to delete records for specific job IDs.
        """
        db = self.database
        q = session.query(db.Intrinsic)
        q = q.filter(db.Intrinsic.cell_id==db.Cell.id)
        q = q.filter(db.Cell.experiment_id==db.Experiment.id)
        q = q.filter(db.Experiment.ext_id.in_(job_ids))
        return q.all()