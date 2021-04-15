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
from ...nwb_recordings import get_intrinsic_recording_dict


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
            recording_dict = get_intrinsic_recording_dict(expt, dev_id, check_qc=True)
            
            lp_results, error = get_long_square_features(recording_dict['LP'], cell_id=cell.id)
            errors += error
            chirp_results, error = get_chirp_features(recording_dict['Chirp'], cell_id=cell.id)
            errors += error
            # Write new record to DB
            conn = db.Intrinsic(cell_id=cell.id, **lp_results, **chirp_results)
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