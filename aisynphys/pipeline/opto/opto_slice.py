from aisynphys.pipeline.pipeline_module import DatabasePipelineModule
from optoanalysis import data_model
import os, glob, re, pickle, time, csv
from aisynphys import config, lims, constants
from acq4.analysis.dataModels.PatchEPhys import getParent
from collections import OrderedDict
from aisynphys.util import datetime_to_timestamp, timestamp_to_datetime
from neuroanalysis.util.optional_import import optional_import
getDirHandle = optional_import('acq4.util.DataManager', 'getDirHandle')


class OptoSlicePipelineModule(DatabasePipelineModule):

    name = 'opto_slice'
    depencencies = []
    table_group = ['slice']

    @classmethod
    def create_db_entries(cls, job, session):
        job_id = job['job_id']
        db = job['database']
        errors = []

        slices = all_slices()
        path = slices[job_id]

        if path == 'place_holder':
            sl = db.Slice(storage_path='place_holder', acq_timestamp=0.0)
            session.add(sl)
            return

        dh = getDirHandle(path)
        info = dh.info()
        parent_info = dh.parent().info()

        # pull some metadata from LIMS
        #sid = self.find_specimen_name(dh)
        sids = data_model.find_lims_specimen_ids(dh)
        if len(sids) == 0:
            limsdata = {}
        elif len(sids) == 1:
            limsdata = lims.specimen_info(specimen_id=sids[0])
        elif len(sids) > 1:
            data = []
            for i in sids:
                data.append(lims.specimen_info(specimen_id=i))
            limsdata = {}
            for key in ['organism', 'date_of_birth', 'age', 'sex', 'plane_of_section', 'exposed_surface', 'hemisphere', 'specimen_name', 'genotype']:
                vals = list(set([d[key] for d in data]))
                if len(vals) == 1:
                    limsdata[key] = vals[0]

        if len(limsdata) == 0:
            errors.append("Could not find limsdata for slice %s" % path)

        quality = info.get('slice quality', None)
        try:
            quality = int(quality)
        except Exception:
            quality = None

        # Interpret slice time
        slice_time = parent_info.get('time_of_dissection', None)
        if slice_time is not None:
            m = re.match(r'((20\d\d)-(\d{1,2})-(\d{1,2}) )?(\d+):(\d+)', slice_time.strip())
            if m is not None:
                _, year, mon, day, hh, mm = m.groups()
                if year is None:
                    date = datetime.fromtimestamp(dh.parent().info()['__timestamp__'])
                    slice_time = datetime(date.year, date.month, date.day, int(hh), int(mm))
                else:
                    slice_time = datetime(int(year), int(mon), int(day), int(hh), int(mm))

        # construct full genotype string 
        genotype = limsdata.get('genotype', '')
        for info in (parent_info, info):
            inj = info.get('injections')
            if inj in (None, ''):
                continue
            if inj not in constants.INJECTIONS:
                raise KeyError("Injection %r is unknown in constants.INJECTIONS" % inj)
            genotype = genotype + ';' + constants.INJECTIONS[inj]

        ## calculate animal age if that data is not entered in lims
        age = limsdata.get('age')
        if age == 0 and len(limsdata) > 0:
            age = (timestamp_to_datetime(info['__timestamp__'])-limsdata['date_of_birth']).days

        fields = {
            'ext_id':'%.3f'%info['__timestamp__'],
            'acq_timestamp': info['__timestamp__'],
            'species': limsdata.get('organism'),
            'date_of_birth': limsdata.get('date_of_birth'),
            'age': age,
            'sex': limsdata.get('sex'),
            'genotype': genotype,
            'orientation': limsdata.get('plane_of_section'),
            'surface': limsdata.get('exposed_surface'),
            'hemisphere': limsdata.get('hemisphere'),
            'quality': quality,
            'slice_time': slice_time,
            'slice_conditions': {},
            'lims_specimen_name': limsdata.get('specimen_name'),
            'storage_path': dh.name(relativeTo=getDirHandle(config.synphys_data)),
        }

        sl = db.Slice(**fields)
        session.add(sl)
        session.commit()
        return errors

    def job_records(self, job_ids, session):
        """Return a list of records associated with a list of job IDs.
        """
        db = self.database
        return session.query(db.Slice).filter(db.Slice.ext_id.in_(job_ids)).all()

    def ready_jobs(self):
        """Return an ordered dict of all jobs that are ready to be processed (all dependencies are present)
        and the dates that dependencies were created.
        """
        slices = all_slices()
        ready = OrderedDict()
        for ts, path in slices.items():
            if path == 'place_holder':
                ready['%.3f'%0.0] = {'dep_time':timestamp_to_datetime(0.0), 'meta':{'source':path}}
            else:
                mtime = os.stat(os.path.join(path, '.index')).st_mtime
                # test file updates:
                # import random
                # if random.random() > 0.8:
                #     mtime *= 2
                ready[ts] = {'dep_time':timestamp_to_datetime(mtime), 'meta':{'source':path}}
        return ready






_all_slices = None
def all_slices():
    """Return a dict mapping {slice_timestamp: path} for all known slices.
    
    This is only generated once per running process; set _all_slices = None
    to force the list to be regenerated.
    """
    global _all_slices
    if _all_slices is not None:
        return _all_slices
        
    # Speed things up by caching this list with a 4 hour timeout
    cachefile = os.path.join(config.cache_path, 'all_slices.pkl')
    if os.path.exists(cachefile):
        age = time.time() - os.stat(cachefile).st_mtime
        if age < 4 * 3600:
            print("Loaded slice timestamps from cache (%0.1f hours old)" % (age/3600.))
            return pickle.load(open(cachefile, 'r'))
    
    expt_csv = config.experiment_csv
    csv_entries = []
    with open(expt_csv, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            csv_entries.append(row)

    _all_slices = OrderedDict()
    errors = {}
    for exp in csv_entries:
        expt_name = exp['experiment'].split('_conn')[0]
        if exp['site_path'] == '':
            _all_slices.update([('%.3f'%0.0, 'place_holder')])
            continue

        site_path = os.path.join(config.synphys_data, exp['rig_name'].lower(), 'phys', exp['site_path'])
        site_dh = getDirHandle(site_path)
        if not site_dh.exists():
            errors[expt_name]="%s does not exist." % site_path
            continue
        if site_dh.info().get('dirType') != 'Site':
            errors[expt_name]="%s is not a site directory." % site_path
            continue
        slice_dh = site_dh.parent()
        if slice_dh.info().get('dirType') != 'Slice':
            errors[expt_name]="Parent of %s is not a slice directory." % site_path
            continue

        ts = slice_dh.info().get('__timestamp__')
        if ts is None:
            #print("MISSING TIMESTAMP: %s" % path)
            #_all_slices.update([('%.3f'%0.0, 'place_holder')])
            errors[expt_name]= "Slice directory %s is missing a timestamp." % slice_dh.path
            continue

        ts = '%0.3f'%ts ## convert timestamp to string here, make sure it has 3 decimal places
        _all_slices[ts] = slice_dh.path

    if len(errors) > 0:
        print("Encountered %i errors:" % len(errors))
        for k, v in errors.items():
            print("    %s : %s" %(k,v))
        
    try:
        tmpfile = cachefile+'.tmp'
        pickle.dump(_all_slices, open(tmpfile, 'w'))
        os.rename(tmpfile, cachefile)
    except:
        if os.path.exists(tmpfile):
            os.remove(tmpfile)
    
    return _all_slices
