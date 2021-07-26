import os, re, shutil, tempfile, datetime, argparse, sys, six
from aisynphys.database import default_db as db
from aisynphys import config
from glob import glob
import pyqtgraph as pg
import pandas as pd

COLUMNS = [
        'sample_id', 
        'res_index', 
        'cluster_label', 
        'Tree_call', 
        'Tree_first_cl',
        'Tree_first_bt',
        'Tree_first_KL',
        'Tree_first_cor',
        'Tree_second_bt',
        ]

def get_mapping_files(species='mouse'):
    base_path = config.mapping_report_address
    if species == 'mouse':
        path_fmt = r"^mouse_patchseq_VISp_(?P<date>\d{8})_collapsed40_cpm$"
        map_file = "/mapping.df.with.bp.40.lastmap.csv"
    elif species == 'human':
        base_path = base_path + "/human"
        path_fmt = r"^human_patchseq_MTG_(?P<date>\d{8})$"
        map_file = "/mapping.df.lastmap.csv"
    else:
        raise Exception("Species must either be mouse or human")

    all_batch_paths = glob(base_path + "/*/")
    mapping_files = {}

    for path in all_batch_paths:
        rel_path = os.path.relpath(path, base_path)
        match = re.match(path_fmt, rel_path)
        if not match:
            continue
        fil = path + map_file
        if not os.path.exists(fil):
            continue

        map_date_str = match.groups('date')[0]
        # print('Adding mapping file: %s' % fil)
        mapping_files[map_date_str] = fil

    return  mapping_files

def extract_data_from_file(map_file, columns):
    open_file, local = tempfile.mkstemp(prefix='mapping_temp', suffix='.csv')
    shutil.copy(map_file, local)
    results = pd.read_csv(local, header=0, index_col=False, dtype=object)
    os.close(open_file)
    os.remove(local)

    if not all(x in results.columns for x in columns):
        # print('All data not present in mapping file %s, skipping' % map_file)
        return
    return results

def get_patchseq_history(columns=COLUMNS, species='mouse'):
    
    q = db.query(db.PatchSeq.tube_id)
    q = q.join(db.Cell).join(db.Experiment).join(db.Slice)
    q = q.filter(db.Slice.species==species)
    patchseq_tubes = q.all()
    patchseq_tubes = [t[0] for t in patchseq_tubes] 
    
    print('Getting mapping files')
    mapping_files = get_mapping_files(species=species)
    result_merge = None
    print('Constructing mapping history')
    for map_date, map_file in mapping_files.items():
        results = extract_data_from_file(map_file, columns)
        if results is None:
            continue
        current_results = {(map_date, feature): results[feature] for feature in columns}
        current_df = pd.DataFrame(current_results)
        current_df = current_df.set_index((map_date, 'sample_id')).rename_axis('sample_id', axis=0)
        if result_merge is None:
            result_merge = current_df
        else:
            result_merge = result_merge.merge(current_df, on='sample_id', how='outer')

    result_history = result_merge.copy()
    result_history.columns.names = ['map_date', 'features']
    dates = result_history.columns.get_level_values('map_date').to_list()
    result_history=result_history.sort_index(axis=1,level=['map_date', 'features'], ascending=[True, True])
    tube_history = result_history[result_history.index.isin(patchseq_tubes)] 
    print('Done!')
    return tube_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--species', type=str, default='mouse')
    parser.add_argument('--dbg', default=False, action='store_true')

    args = parser.parse_args(sys.argv[1:])

    if args.dbg is True:
        pg.dbg()

    

    get_patchseq_history(columns, species=args.species)
