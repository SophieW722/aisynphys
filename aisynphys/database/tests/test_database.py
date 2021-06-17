import sqlalchemy
from sqlalchemy.orm import aliased
from aisynphys.database import default_db as db


def mk_test_query():
    aliased_slice = aliased(db.Slice, name='aliased_slice')
    q = db.query(
        db.Experiment.id,
        db.Experiment.project_name,
        db.Experiment.target_temperature.label('temp'),
        db.Slice,
        aliased_slice,
        db.slice.id,
        aliased_slice.id,
        db.Experiment.slice_id,
        db.slice.id + db.Experiment.target_temperature,
        (db.slice.id + db.Experiment.target_temperature).label('sum_of_fields'),
        (-db.Experiment.target_temperature).label('negative temp'),
    ).join(db.Slice).join(aliased_slice).limit(10)
    return q


def test_recarray():
    q = mk_test_query()

    # First test query without tables expanded
    arr = q.recarray()

    assert arr.shape == (10,)
    fields = dict(arr.dtype.fields)
    assert list(fields.keys()) == [
        'experiment.id',
        'experiment.project_name',
        'temp',
        'slice',
        'aliased_slice',
        'slice.id',
        'aliased_slice.id',
        'experiment.slice_id',
        'slice.id + experiment.target_temperature',
        'sum_of_fields',
        'negative temp',
    ]

    assert fields['experiment.id'][0].kind == 'i'
    assert fields['experiment.project_name'][0].kind == 'O'
    assert fields['temp'][0].kind == 'f'
    assert fields['slice'][0].kind == 'O'
    assert fields['aliased_slice'][0].kind == 'O'
    assert fields['slice.id'][0].kind == 'i'
    assert fields['aliased_slice.id'][0].kind == 'i'
    assert fields['experiment.slice_id'][0].kind == 'i'
    assert fields['slice.id + experiment.target_temperature'][0].kind == 'f'
    assert fields['sum_of_fields'][0].kind == 'f'
    assert fields['negative temp'][0].kind == 'f'


    # Now test with expanded tables

    arr = q.recarray(expand_tables=True)
    assert arr.shape == (10,)
    fields = dict(arr.dtype.fields)
    assert list(fields.keys()) == ([
        'experiment.id',
        'experiment.project_name',
        'temp',] +
        get_table_nondeferred_columns(db.Slice, prefix='slice.') +
        get_table_nondeferred_columns(db.Slice, prefix='aliased_slice.') + [
        'slice.id_1',
        'aliased_slice.id_1',
        'experiment.slice_id',
        'slice.id + experiment.target_temperature',
        'sum_of_fields',
        'negative temp',
    ])

    # queries behave differently with just a single item
    arr = db.query(db.PulseResponse).limit(3).recarray(expand_tables=False)
    assert arr.dtype.names == ('pulse_response',)

    arr = db.query(db.PulseResponse).limit(3).recarray(expand_tables=True)
    attr_names = get_table_nondeferred_columns(db.PulseResponse, prefix='pulse_response.')
    assert set(arr.dtype.names) == set(attr_names)


def get_table_nondeferred_columns(table, prefix=''):
    meta = sqlalchemy.inspect(table)
    return [prefix + name for name in meta.columns.keys() if not meta.column_attrs[name].deferred]


def test_dataframe():
    q = mk_test_query()
    df = q.dataframe()

    assert list(df.columns) == [
        'experiment.id',
        'experiment.project_name',
        'temp',
        'slice',
        'aliased_slice',
        'slice.id',
        'aliased_slice.id',
        'experiment.slice_id',
        'slice.id + experiment.target_temperature',
        'sum_of_fields',
        'negative temp',
    ]
