from sqlalchemy.orm import aliased
from aisynphys.database import default_db as db


def mk_test_query():
    aliased_slice = aliased(db.Slice, name='aliased_slice')
    q = db.query(
        db.Experiment.id,
        db.Experiment.project_name,
        db.Experiment.target_temperature.label('temp'),
        db.slice.id,
        aliased_slice.id,
        db.Slice,
        aliased_slice,
        db.Experiment.slice_id,
        db.slice.id + db.Experiment.target_temperature,
        (db.slice.id + db.Experiment.target_temperature).label('sum_of_fields'),
        (-db.Experiment.target_temperature).label('negative temp'),
    ).join(db.Slice).join(aliased_slice).limit(10)
    return q


def test_recarray():
    q = mk_test_query()
    arr = q.recarray()

    assert arr.shape == (10,)
    fields = dict(arr.dtype.fields)
    assert list(fields.keys()) == [
        'experiment.id',
        'experiment.project_name',
        'temp',
        'slice.id',
        'aliased_slice.id',
        'slice',
        'aliased_slice',
        'experiment.slice_id',
        'slice.id + experiment.target_temperature',
        'sum_of_fields',
        'negative temp',
    ]

    assert fields['experiment.id'][0].kind == 'i'
    assert fields['experiment.project_name'][0].kind == 'O'
    assert fields['temp'][0].kind == 'f'
    assert fields['slice.id'][0].kind == 'i'
    assert fields['aliased_slice.id'][0].kind == 'i'
    assert fields['slice'][0].kind == 'O'
    assert fields['aliased_slice'][0].kind == 'O'
    assert fields['experiment.slice_id'][0].kind == 'i'
    assert fields['slice.id + experiment.target_temperature'][0].kind == 'f'
    assert fields['sum_of_fields'][0].kind == 'f'
    assert fields['negative temp'][0].kind == 'f'


def test_dataframe():
    q = mk_test_query()
    df = q.dataframe()

    assert list(df.columns) == [
        'experiment.id',
        'experiment.project_name',
        'temp',
        'slice.id',
        'aliased_slice.id',
        'slice',
        'aliased_slice',
        'experiment.slice_id',
        'slice.id + experiment.target_temperature',
        'sum_of_fields',
        'negative temp',
    ]
