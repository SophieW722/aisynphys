import os, re, shutil, tempfile, datetime, argparse, sys, six
from aisynphys.database import default_db as db
import aisynphys.data.data_notes_db as notes_db
from aisynphys import config
from glob import glob
import pyqtgraph as pg
import pandas as pd

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
        print('Adding mapping file: %s' % fil)
        mapping_files[map_date_str] = fil

    return  mapping_files

def compile_patchseq_history(mapping_files):
    result_merge = None
    for map_date, map_file in mapping_files.items():
        open_file, local = tempfile.mkstemp(prefix='mapping_temp', suffix='.csv')
        shutil.copy(map_file, local)
        results = pd.read_csv(local, header=0, index_col=False, dtype=object)
        os.close(open_file)
        os.remove(local)

        if not all(x in results.columns for x in columns):
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

    return result_history

def build_tube_history(columns):
    mouse_mapping_files = get_mapping_files(species='mouse')
    human_mapping_files = (get_mapping_files(species='human'))
    q = db.query(db.PatchSeq.tube_id, db.Slice.species)
    q = q.join(db.Cell).join(db.Experiment).join(db.Slice)
    patchseq_tubes = q.all()
    
    print('Constructing merged mapping history for mouse...')
    mouse_result_history = compile_patchseq_history(mouse_mapping_files)
    print('Constructing merged mapping history for human...')
    human_result_history = compile_patchseq_history(human_mapping_files)

    print('Dropping patchseq_history table...')
    notes_db.db.drop_tables(tables=['patchseq_history'])
    print('Creating new patchseq_history table...')
    notes_db.db.create_tables(tables=['patchseq_history'])
    print('Building history for %d tubes...' % len(patchseq_tubes))
    session = notes_db.db.session(readonly=False)
    for i, tube in enumerate(patchseq_tubes):
        tube_id, species = tube
        print('tube %d/%d: %s' % (i, len(patchseq_tubes), tube_id))
        if re.match(r'P(M|T|X)S4_(?P<date>\d{6})_(?P<tube_id>\d{3})_A01', tube_id) is None:
            print('\ttube name does not have proper format, carrying on...')
            continue
        if species == 'mouse':
            result_history = mouse_result_history
        elif species == 'human':
            result_history = human_result_history
        else:
            print('Species %s does not match mouse or human' % species)
            continue
        if tube_id not in result_history.index:
            print('\ttube has no mapping results')
            continue
        tube_data = pd.DataFrame(result_history.loc[tube_id]).reset_index()
        tube_data = tube_data.pivot(index='map_date', columns='features', values=tube_id).dropna()
        tube_notes = tube_data.to_dict('index')

        tube_rec = notes_db.PatchseqHistory(
            tube_id=tube_id,
            notes=tube_notes,
            modification_time=datetime.datetime.now(),
        )

        session.add(tube_rec)
        print('\tadded record')
    session.commit()
    session.close()
    print('Done!')

def update_tube_history(columns, species='mouse'):
    mapping_files = get_mapping_files(species=species)
    patchseq_tubes = db.query(db.PatchSeq.tube_id).all()

    map_dates = sorted(mapping_files.keys())
    #probably should change this to do a date match with the last date for each tube. In which case I can probably use the same build_history and just check that the date is in the keys
    most_recent_date = map_dates[-1]
    map_file = mapping_files[most_recent_date]

    open_file, local = tempfile.mkstemp(prefix='mapping_temp', suffix='.csv')
    shutil.copy(map_file, local)
    results = pd.read_csv(local, header=0, index_col=False, dtype=object)
    os.close(open_file)
    os.remove(local)

    if not all(x in results.columns for x in columns):
        print('All data not present in mapping file, exiting')
        return
    results.set_index('sample_id', inplace=True)
    session = notes_db.db.session(readonly=False)
    print('Building history for %d tubes...' % len(patchseq_tubes))
    for i, tube_id in enumerate(patchseq_tubes):
        tube_id = tube_id[0]
        print('tube %d/%d: %s' % (i, len(patchseq_tubes), tube_id))
        if re.match(r'P(M|T)S4_(?P<date>\d{6})_(?P<tube_id>\d{3})_A01', tube_id) is None:
            print('\ttube name does not have proper format, carrying on...')
            continue
        if tube_id not in results.index:
            print('\ttube has no mapping results')
            continue
        tube_data = results.loc[tube_id][columns].to_dict()
        try:
            tube_rec = notes_db.get_tube_history_record(tube_id, session=session)
        except Exception as e:
            print(e)
            continue
        if tube_rec is not None:
            tube_notes = tube_rec.notes
        else:
            tube_rec = notes_db.PatchseqHistory(
            tube_id=tube_id,
            )
            tube_notes = {}
        tube_notes[most_recent_date] = tube_data
        tube_rec.notes = tube_notes
        tube_rec.modification_time = datetime.datetime.now()
        session.commit()
        print('\tupdated record')

    session.close()
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', default=False, action='store_true')
    parser.add_argument('--update', default=False, action='store_true')
    parser.add_argument('--species', type=str, default='mouse')
    parser.add_argument('--dbg', default=False, action='store_true')

    args = parser.parse_args(sys.argv[1:])

    if args.dbg is True:
        pg.dbg()

    columns = [
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

    if args.rebuild is True:
        if six.moves.input("Rebuild patchseq_history table? (y/n) ") != 'y':
            print("  Nuts.")
            sys.exit(-1)
        build_tube_history(columns)

    if args.update is True:
        update_tube_history(columns, species=args.species)