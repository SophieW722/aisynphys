"""
For generating DB tables describing 1) a cells location within cortex, 2) a cortical site.

"""
from __future__ import print_function, division
from ..pipeline_module import DatabasePipelineModule
from .opto_experiment import OptoExperimentPipelineModule, load_experiment
from collections import OrderedDict
import datetime, os
import json
import numpy as np
from aisynphys import config
from pyqtgraph.debug import Profiler

class OptoCortexLocationPipelineModule(DatabasePipelineModule):
    """Imports cell and site location data for each experiment
    """
    name = 'opto_cortical_location'
    dependencies = [OptoExperimentPipelineModule]
    table_group = ['cortical_cell_location', 'cortical_site']
    
    @classmethod
    def create_db_entries(cls, job, session):
        db = job['database']
        job_id = job['job_id'] ## an expt id
        errors = []

        prof = Profiler('opto_cortical_location', disabled=True)

        expt = load_experiment(job_id)
        expt_entry = db.experiment_from_ext_id(expt.ext_id, session=session)

        prof.mark('got experiment')

        try:
            # look up slice record in DB
            ts = expt.info.get('slice_info', {}).get('__timestamp__')
            if ts is None:
                ts = 0.0
            slice_entry = db.slice_from_timestamp(ts, session=session)

            cortex = {}
            if expt.loader.cnx_file != 'not found':
                with open(expt.loader.cnx_file) as f:
                    cnx_json = json.load(f)
                    cortex = cnx_json.get('CortexMarker', {})
            if len(cortex) == 0: ## if we don't have a cortex marker in our cnx file maybe we can still pull some positions from the .mosaic
                if expt.mosaic_file is not None:
                    with open(expt.mosaic_file) as f:
                        mosaic = json.load(f)
                        for item in mosaic['items']:
                            if item['name'] == 'CortexMarker':
                                cortex['wmPos'] = item.get('wmPos')
                                cortex['piaPos'] = item.get('piaPos')
                                if cortex['wmPos'] is not None and cortex['piaPos'] is not None:
                                    p1, p2 = cortex['piaPos'], cortex['wmPos']
                                    cortex['pia_to_wm_distance'] = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5
                                cortex['layerBounds_percentDepth'] = item.get('roiState', {}).get('layerBounds_percentDepth')
                                cortex['sliceAngle'] = item.get('sliceAngle')
                                break
            if len(cortex) == 0:
                errors.append('No cortex marker found in connections file or mosaic file.')

            prof.mark('got cortex info')

            site_entry = db.CorticalSite(
                    pia_to_wm_distance=cortex.get('pia_to_wm_distance'),
                    pia_position=cortex.get('piaPos'),
                    wm_position=cortex.get('wmPos'),
                    layer_boundaries=cortex.get('layerBounds_percentDepth'),
                    #brain_region=None,     # The name of the brain region for the site.
                    meta={'slice_angle':cortex.get('sliceAngle')}
                    )

            site_entry.slice = slice_entry
            site_entry.experiment = expt_entry
            session.add(site_entry)

            prof.mark('created site entry')

            cells = expt.cells.items()

            prof.mark('loaded cells')

            for cell_id, cell in cells:
                layer_depth = None
                fractional_layer_depth = None
                layer_thickness = None
                if cortex.get('layerBounds_percentDepth') is not None:
                    frac_layer_bounds = cortex['layerBounds_percentDepth'].get('L'+cell.target_layer)
                    if frac_layer_bounds is None:
                        errors.append("Could not match cell layer to layer bounds for %s. target_layer:%s, layers:%s" %(cell, cell.target_layer, list(cortex['layerBounds_percentDepth'].keys())))
                        #raise Exception('Could not find layer bounds for layer:"%s" (cell: %s). options are:%s' %('L'+cell.target_layer, cell, cortex.get('layers')))
                    else:
                        layer_depth = (cell.percent_depth-frac_layer_bounds[0])*cortex['pia_to_wm_distance'] 
                        fracional_layer_depth = (cell.percent_depth-frac_layer_bounds[0])/(frac_layer_bounds[1]-frac_layer_bounds[0]) 
                        layer_thickness = (frac_layer_bounds[1]-frac_layer_bounds[0])/cortex['pia_to_wm_distance']
                loc_entry = db.CorticalCellLocation(
                    #cell_id=cell.cell_id,
                    cortical_layer=cell.target_layer,
                    distance_to_pia=cell.distance_to_pia,
                    distance_to_wm=cell.distance_to_wm,
                    fractional_depth=cell.percent_depth,
                    layer_depth=layer_depth,                            # Absolute depth within the layer in m.
                    layer_thickness=layer_thickness,
                    fractional_layer_depth=fractional_layer_depth,      # Fractional depth within the cells layer.
                    position=None,                                      # 2D array, position of cell in slice image coordinates (in m) -- i think this is for having lims images?
                                                                        #      - possibly for us this should just be the x,y position of the cell - but this would be redundant, so 
                                                                        #        I'll leave it blank for now so it can't accidentally be misused
                )

                q = session.query(db.Cell)
                q = q.filter(db.Cell.experiment_id==db.Experiment.id)
                q = q.filter(db.Experiment.ext_id==job_id)
                q = q.filter(db.Cell.ext_id==cell_id)
                cell_entry = q.all()

                if len(cell_entry) == 1:
                    loc_entry.cell = cell_entry[0]
                else:
                    raise Exception("Found %i cell entries for experiment %s, cell %s" %(len(cell_entry),job_id, cell_id))
                loc_entry.cortical_site = site_entry
                session.add(loc_entry)

            prof.mark('created cell_location entries')

            ## fill in lateral_distance and vertical_distance for each pair
            if cortex.get('sliceAngle') is not None:
                angle = cortex.get('sliceAngle') * np.pi/180.
                for pair_entry in expt_entry.pair_list:
                    pre_cell_pos = pair_entry.pre_cell.position
                    post_cell_pos = pair_entry.post_cell.position
                    if pre_cell_pos is None or post_cell_pos is None:
                        continue
                    x1, y1 = post_cell_pos[0]-pre_cell_pos[0], post_cell_pos[1] - pre_cell_pos[1]
                    x2 = abs(np.cos(angle)*x1 - np.sin(angle)*y1)
                    y2 = abs(np.sin(angle)*x1 + np.cos(angle)*y1)
                    pair_entry.lateral_distance = x2
                    pair_entry.vertical_distance = y2

            prof.mark('updted pair entries')

            return errors

        except:
            session.rollback()
            raise


    def job_records(self, job_ids, session):
        """Return a list of records associated with a list of job IDs.
        
        This method is used by drop_jobs to delete records for specific job IDs.
        """
        db = self.database
        return session.query(db.CorticalSite).filter(db.CorticalSite.experiment_id==db.Experiment.id).filter(db.Experiment.ext_id.in_(job_ids)).all()

        #return session.query(db.Morphology).filter(db.Morphology.cell_id==db.Cell.id).filter(db.Cell.experiment_id==db.Experiment.id).filter(db.Experiment.acq_timestamp.in_(job_ids)).all()

    def ready_jobs(self):
        """Return an ordered dict of all jobs that are ready to be processed (all dependencies are present)
        and the dates that dependencies were created.
        """
        db = self.database
        # All experiments and their creation times in the DB
        expts = self.pipeline.get_module('opto_experiment').finished_jobs()

        session = db.session()
        session.rollback()
        
        ready = OrderedDict()

        for expt_id, (expt_mtime, success) in expts.items():
            if success is not True:
                continue
            expt = load_experiment(expt_id)
            if expt.loader.cnx_file == 'not found':
                if expt.mosaic_file is not None:
                    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(expt.mosaic_file))
                else:
                    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(config.distance_csv))
            elif expt.loader.get_cnx_file_version(expt.loader.cnx_file) >= 3:
                mtime = datetime.datetime.fromtimestamp(os.path.getmtime(expt.loader.cnx_file))
            else:
                mtime = datetime.datetime.fromtimestamp(os.path.getmtime(config.distance_csv))
            ready[expt_id] = {'dep_time':mtime, 'meta':{}}
    
        return ready