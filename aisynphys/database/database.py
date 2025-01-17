"""
Low-level relational database / sqlalchemy interaction.

The actual schemas for database tables are implemented in other files in this subpackage.
"""
from __future__ import division, print_function

import os, sys, io, json, threading, gc, re, weakref
from collections import OrderedDict, namedtuple
import numpy as np
try:
    import queue
except ImportError:
    import Queue as queue
from inspect import isclass

import sqlalchemy
from distutils.version import LooseVersion
if LooseVersion(sqlalchemy.__version__) < '1.2':
    raise Exception('requires at least sqlalchemy 1.2')

import sqlalchemy.inspection, sqlalchemy.pool
from sqlalchemy import create_engine, Column, Integer, BigInteger, String, Boolean, Float, Date, DateTime, LargeBinary, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, deferred, sessionmaker, reconstructor
from sqlalchemy.types import TypeDecorator
from sqlalchemy.sql.expression import func


from .. import config
from neuroanalysis.util.optional_import import optional_import
pandas = optional_import('pandas')


class NDArray(TypeDecorator):
    """For marshalling arrays in/out of binary DB fields.
    """
    impl = LargeBinary
    hashable = False
    cache_ok = False
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return b'' 
        buf = io.BytesIO()
        np.save(buf, value, allow_pickle=False)
        return buf.getvalue()
        
    def process_result_value(self, value, dialect):
        if value is None or value == b'':
            return None
        buf = io.BytesIO(value)
        return np.load(buf, allow_pickle=False)

    @property
    def python_type(self):
        return np.ndarray

class CustomEncoder(json.JSONEncoder):
    """ For encoding nonserializable floats into json
    """
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

class JSONObject(TypeDecorator):
    """For marshalling objects in/out of json-encoded text.
    """
    impl = String
    hashable = False
    
    def process_bind_param(self, value, dialect):
        return json.dumps(value, cls=CustomEncoder)
        
    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return json.loads(value)

    @property
    def python_type(self):
        object


class FloatType(TypeDecorator):
    """For marshalling float types (including numpy).
    """
    impl = Float
    cache_ok = False
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return float(value)

    @property
    def python_type(self):
        return float

    #def process_result_value(self, value, dialect):
        #buf = io.BytesIO(value)
        #return np.load(buf, allow_pickle=False)


column_data_types = {
    'int': Integer,
    'bigint': BigInteger,
    'float': FloatType,
    'bool': Boolean,
    'str': String,
    'date': Date,
    'datetime': DateTime,
    'array': NDArray,
#    'object': JSONB,  # provides support for postges jsonb, but conflicts with sqlite
    'object': JSONObject,
}


def make_table_docstring(table):
    """Introspect ORM table class to generate a nice docstring.
    """
    docstr = ['Sqlalchemy model for "%s" database table.\n' % table.__name__]
    comment = table.__table_args__.get('comment', None)
    if comment is not None:
        docstr.append(comment.strip() + '\n')
            
    
    insp = sqlalchemy.inspection.inspect(table)
    
    docstr.append("Attributes\n----------")
    for name, prop in insp.relationships.items():
        docstr.append("%s : relationship" % name)
        if hasattr(prop, 'entity'):
            # entity attribute only available in recent sqlalchemy (>=1.3 ?)
            docstr.append("    Reference to %s.%s" % (prop.entity.primary_key[0].table.name, prop.entity.primary_key[0].name))
    for name, col in insp.columns.items():
        typ_str = str(col.type)
        docstr.append("%s : %s" % (name, typ_str))
        if col.comment is not None:
            docstr.append("    " + col.comment)
        
    return '\n'.join(docstr)


def make_table(ormbase, name, columns, base=None, **table_args):
    """Generate an ORM mapping class from a simplified schema format.

    Columns named 'id' (int) and 'meta' (object) are added automatically.

    Parameters
    ----------
    ormbase : ORMBase instance
        The sqlalchemy ORM base on which to create this table.
    name : str
        Name of the table, used to set __tablename__ in the new class
    base : class or None
        Base class on which to build the new table class
    table_args : keyword arguments
        Extra keyword arguments are used to set __table_args__ in the new class
    columns : list of tuple
        List of column specifications. Each column is given as a tuple:
        ``(col_name, data_type, comment, {options})``. Where *col_name* and *comment* 
        are strings, *data_type* is a key in the column_data_types global, and
        *options* is a dict providing extra initialization arguments to the sqlalchemy
        Column (for example: 'index', 'unique'). Optionally, *data_type* may be a 'tablename.id'
        string indicating that this column is a foreign key referencing another table.
    """
    class_name = ''.join([part.title() for part in name.split('_')])

    props = {
        '__tablename__': name,
        '__table_args__': table_args,
        'id': Column(Integer, primary_key=True),
    }

    for column in columns:
        colname, coltype = column[:2]
        
        # avoid weird sqlalchemy issues with case handling
        assert colname == colname.lower(), "Column names must be all lowercase (got %s.%s)" % (name, colname)
        
        kwds = {} if len(column) < 4 else column[3]
        kwds['comment'] = None if len(column) < 3 else column[2]
        defer_col = kwds.pop('deferred', False)
        ondelete = kwds.pop('ondelete', None)

        if coltype not in column_data_types:
            if not coltype.endswith('.id'):
                raise ValueError("Unrecognized column type %s" % coltype)
            # force indexing on all foreign keys; otherwise deletes can become vrey slow
            kwds['index'] = True
            props[colname] = Column(Integer, ForeignKey(coltype, ondelete=ondelete), **kwds)
        else:
            ctyp = column_data_types[coltype]
            props[colname] = Column(ctyp, **kwds)

        if defer_col:
            props[colname] = deferred(props[colname])

    props['meta'] = Column(column_data_types['object'])

    if base is None:
        new_table = type(class_name, (ormbase,), props)
    else:
        # need to jump through a hoop to allow __init__ on table classes;
        # see: https://docs.sqlalchemy.org/en/latest/orm/constructors.html
        if hasattr(base, '_init_on_load'):
            @reconstructor
            def _init_on_load(self, *args, **kwds):
                base._init_on_load(self)
            props['_init_on_load'] = _init_on_load
        new_table = type(class_name, (base, ormbase), props)

    return new_table


class Database(object):
    """Methods for doing relational database maintenance via sqlalchemy.
    
    Supported backends: postgres, sqlite.
    
    Features:
    
    * Automatically build/dispose ro and rw engines (especially after fork)
    * Generate ro/rw sessions on demand
    * Methods for creating / dropping databases
    * Clone databases across backends
    """
    _all_dbs = weakref.WeakSet()
    default_app_name = (' '.join(sys.argv))[-63:]

    def __init__(self, ro_host, rw_host, db_name, ormbase):
        self.ormbase = ormbase
        self._mappings = {}

        # default options for creating DB engines
        self._engine_opts = {
            'postgresql': {
                'ro': {'echo': False, 'poolclass': sqlalchemy.pool.NullPool, 'isolation_level': 'AUTOCOMMIT'}, # {'pool_size': 0, 'max_overflow': 40, }
                'rw': {'poolclass': sqlalchemy.pool.NullPool}, #{'pool_size': 0, 'max_overflow': 40},
                'maint': {'poolclass': sqlalchemy.pool.NullPool},
            }
        }

        self.ro_host = ro_host
        self.rw_host = rw_host
        self.db_name = db_name
        self._ro_engine = None
        self._rw_engine = None
        self._maint_engine = None
        self._engine_pid = None  # pid of process that created these engines.
        self._ro_sessionmaker = None
        self._rw_sessionmaker = None

        self.ro_address = self.db_address(ro_host, db_name)
        self.rw_address = None if rw_host is None else self.db_address(rw_host, db_name)
        self._all_dbs.add(self)
        
        self._default_session = None

    @property
    def default_session(self):
        self._check_engines()
        if self._default_session is None:
            self._default_session = self.session(readonly=True)
        return self._default_session

    def query(self, *args, **kwds):
        return self.default_session.query(*args, **kwds)

    def _find_mappings(self):
        mappings = {cls.__tablename__:cls for cls in self.ormbase.__subclasses__()}
        order = [t.name for t in self.ormbase.metadata.sorted_tables]
        self._mappings = OrderedDict([(t, mappings[t]) for t in order if t in mappings])

    def __getattr__(self, attr):
        try:
            # pretty sure I'll regret this later:  I want to be able to ask for db.TableName
            # and return the ORM object for a table.
            
            # convert CamelCase to snake_case  (credit: https://stackoverflow.com/a/12867228/643629)
            table = re.sub(r'((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))', r'_\1', attr).lower()
            if table not in self._mappings:
                self._find_mappings()
            return self._mappings[table]
        except Exception:
            return object.__getattribute__(self, attr)

    def __repr__(self):
        return "<%s %s (%s)>" % (type(self).__name__, self.ro_address, 'ro' if self.rw_address is None else 'rw')

    def __str__(self):
        # str(engine) does a nice job of masking passwords
        s = str(self.ro_engine)[7:]
        s = s.rstrip(')')
        s = s.partition('?')[0]
        return s

    @property
    def backend(self):
        """Return the backend used by this database (sqlite, postgres, etc.)
        """
        # maybe ro_engine.name instead?
        return self.ro_host.partition(':')[0]

    @classmethod
    def db_address(cls, host, db_name=None, app_name=None):
        """Return a complete address for DB access given a host (like postgres://user:pw@host) and database name.

        Appends an app name to postgres addresses.
        """
        if host.startswith('postgres'):
            app_name = app_name or cls.default_app_name
            return "{host}/{db_name}?application_name={app_name}".format(host=host, db_name=db_name, app_name=app_name)
        else:
            # for sqlite, db_name is the file path
            if not host.endswith('/'):
                host = host + '/'
            return host + db_name

    def get_database(self, db_name):
        """Return a new Database object with the same hosts and orm base, but different db name
        """
        return Database(self.ro_host, self.rw_host, db_name, self.ormbase)
        
    def dispose_engines(self):
        """Dispose any existing DB engines. This is necessary when forking to avoid accessing the same DB
        connection simultaneously from two processes.
        """
        if self._ro_engine is not None:
            self._ro_engine.dispose()
        if self._rw_engine is not None:
            self._rw_engine.dispose()
        if self._maint_engine is not None:
            self._maint_engine.dispose()
        self._ro_engine = None
        self._ro_sessionmaker = None
        self._rw_engine = None
        self._rw_sessionmaker = None
        self._maint_engine = None
        self._engine_pid = None    
        self._default_session = None
            
        # collect now or else we might try to collect engine-related garbage in forked processes,
        # which can lead to "OperationalError: server closed the connection unexpectedly"
        # Note: if this turns out to be flaky as well, we can just disable connection pooling.
        gc.collect()

    @classmethod
    def dispose_all_engines(cls):
        """Dispose engines on all Database instances.
        """
        for db in cls._all_dbs:
            db.dispose_engines()

    def _check_engines(self):
        """Dispose engines if they were built for a different PID
        """
        if os.getpid() != self._engine_pid:
            # In forked processes, we need to re-initialize the engine before
            # creating a new session, otherwise child processes will
            # inherit and muck with the same connections. See:
            # https://docs.sqlalchemy.org/en/latest/faq/connections.html#how-do-i-use-engines-connections-sessions-with-python-multiprocessing-or-os-fork
            if self._engine_pid is not None:
                print("Making new session for subprocess %d != %d" % (os.getpid(), self._engine_pid))
            self.dispose_engines()

    @property
    def ro_engine(self):
        """The read-only database engine.
        """
        self._check_engines()
        if self._ro_engine is None:
            opts = self._engine_opts.get(self.backend, {}).get('ro', {})
            self._ro_engine = create_engine(self.ro_address, **opts)
            self._engine_pid = os.getpid()
        return self._ro_engine
    
    @property
    def rw_engine(self):
        """The read-write database engine.
        """
        self._check_engines()
        if self._rw_engine is None:
            if self.rw_address is None:
                return None
            opts = self._engine_opts.get(self.backend, {}).get('rw', {})
            self._rw_engine = create_engine(self.rw_address, **opts)
            self._engine_pid = os.getpid()
        return self._rw_engine

    @property
    def maint_engine(self):
        """The maintenance engine.
        
        For postgres DBs, this connects to the "postgres" database.
        """
        self._check_engines()
        if self._maint_engine is None:
            opts = self._engine_opts.get(self.backend, {}).get('maint', None)
            if opts is None:
                # maybe just return rw engine for postgres?
                raise Exception("no maintenance connection configured for DB %s" % self)
            maint_addr = self.db_address(self.rw_host, 'postgres')
            self._maint_engine = create_engine(maint_addr, **opts)
            self._engine_pid = os.getpid()
        return self._maint_engine

    # external users should create sessions from here.
    def session(self, readonly=True):
        """Create and return a new database Session instance.
        
        If readonly is True, then the session is created using read-only credentials and has autocommit enabled.
        This prevents idle-in-transaction timeouts that occur when GUI analysis tools would otherwise leave transactions
        open after each request.
        """
        if readonly:
            if self._ro_sessionmaker is None:
                self._ro_sessionmaker = sessionmaker(bind=self.ro_engine, query_cls=DBQuery)
            return self._ro_sessionmaker()
        else:
            if self.rw_engine is None:
                raise RuntimeError("Cannot start read-write DB session; no write access engine is defined (see config.synphys_db_host_rw)")
            if self._rw_sessionmaker is None:
                self._rw_sessionmaker = sessionmaker(bind=self.rw_engine, query_cls=DBQuery)
            return self._rw_sessionmaker()

    def reset_db(self):
        """Drop the existing database and initialize a new one.
        """
        self.dispose_engines()
        
        self.drop_database()
        self.create_database()
        
        self.create_tables()
        self.grant_readonly_permission()

    def list_databases(self):
        engine = self.maint_engine
        with engine.begin() as conn:
            conn.connection.set_isolation_level(0)
            return [rec[0] for rec in conn.execute('SELECT datname FROM pg_catalog.pg_database;')]

    @property
    def exists(self):
        """Bool indicating whether this DB exists yet.
        """
        if self.backend == 'sqlite':
            return os.path.isfile(self.db_name)
        else:
            return self.db_name in self.list_databases()

    def drop_database(self):
        if self.backend == 'sqlite':
            if os.path.isfile(self.db_name):
                os.remove(self.db_name)
        elif self.backend == 'postgresql':
            self.dispose_all_engines()
            engine = self.maint_engine
            with engine.begin() as conn:
                conn.connection.set_isolation_level(0)
                try:
                    conn.execute('drop database %s' % self.db_name)
                except sqlalchemy.exc.ProgrammingError as err:
                    if 'does not exist' not in err.args[0]:
                        raise
        else:
            raise TypeError("Unsupported database backend %s" % self.backend)

    def create_database(self):
        if self.backend == 'sqlite':
            return
        elif self.backend == 'postgresql':
            # connect to postgres db just so we can create the new DB
            engine = self.maint_engine
            with engine.begin() as conn:
                conn.connection.set_isolation_level(0)
                conn.execute('create database %s' % self.db_name)
                # conn.execute('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO %s;' % ro_user)
        else:
            raise TypeError("Unsupported database backend %s" % self.backend)

    def grant_readonly_permission(self):
        if self.backend == 'sqlite':
            return
        elif self.backend == 'postgresql':
            ro_user = config.synphys_db_readonly_user

            # grant readonly permissions
            with self.rw_engine.begin() as conn:
                conn.connection.set_isolation_level(0)
                for cmd in [
                    ('GRANT CONNECT ON DATABASE %s TO %s' % (self.db_name, ro_user)),
                    ('GRANT USAGE ON SCHEMA public TO %s' % ro_user),
                    ('GRANT SELECT ON ALL TABLES IN SCHEMA public to %s' % ro_user)]:
                    conn.execute(cmd)
        else:
            raise TypeError("Unsupported database backend %s" % self.backend)

    def orm_tables(self):
        """Return a dependency-sorted of ORM mapping objects (tables) that are described by the ORM base for this database.
        """
        # need to re-run every time because we can't tell when a new mapping has been added.
        self._find_mappings()
        return self._mappings

    def metadata_tables(self):
        """Return an ordered dictionary (dependency-sorted) of {name:Table} pairs, one for 
        each table in the sqlalchemy metadata for this database.
        """
        return OrderedDict([(t.name, t) for t in self.ormbase.metadata.sorted_tables])

    def table_names(self):
        """Return a list of the names of tables in this database.
        
        May contain names that are not present in metadata_tables or orm_tables.
        """
        return self.ro_engine.table_names()

    def create_tables(self, tables=None, initialize=True):
        """Create tables in the database from the ORM base specification.
        
        A list of the names of *tables* may be optionally specified to 
        create a subset of known tables.
        """
        # Create all tables
        meta_tables = self.metadata_tables()
        if tables is not None:
            tables = [meta_tables[t] for t in tables]
        self.ormbase.metadata.create_all(bind=self.rw_engine, tables=tables)
        self.grant_readonly_permission()

        if initialize:
            self.initialize_database()

    def initialize_database(self):
        """Optionally called after create_tables. 

        Initialize is _not_ called when cloning databases.

        Default does nothing; subclasses may override.
        """
        pass

    def drop_tables(self, tables=None):
        """Drop a list of tables (or all ORM-defined tables, if no list is given) from this database.
        """
        drops = []
        meta_tables = self.metadata_tables()
        db_tables = self.table_names()
        for k in meta_tables:
            if tables is not None and k not in tables:
                continue
            if k in db_tables:
                drops.append(k)
        if len(drops) == 0:
            return
            
        if self.backend == 'sqlite':
            for table in drops:
                self.rw_engine.execute('drop table %s' % table)
        else:
            self.rw_engine.execute('drop table %s cascade' % (','.join(drops)))

    # Seems to be not working correctly        
    # def enable_triggers(self, enable):
    #     """Enable or disable triggers for all tables in this group.
    #    
    #     This can be used to temporarily disable constraint checking on tables that are under development,
    #     allowing the rest of the pipeline to continue operating (for example, if removing an object from 
    #     the pipeline would violate a foreign key constraint, disabling triggers will allow this constraint
    #     to go unchecked).
    #     """
    #     s = Session(readonly=False)
    #     enable = 'enable' if enable else 'disable'
    #     for table in self.tables.keys():
    #         s.execute("alter table %s %s trigger all;" % (table, enable))
    #     s.commit()

    def vacuum(self, tables=None):
        """Cleans up database and analyzes table statistics in order to improve query planning.
        Should be run after any significant changes to the database.
        """
        with self.rw_engine.begin() as conn:
            if self.backend == 'postgresql':
                conn.connection.set_isolation_level(0)
                if tables is None:
                    conn.execute('vacuum analyze')
                else:
                    for table in tables:
                        conn.execute('vacuum analyze %s' % table)
            else:
                conn.execute('vacuum')

    def bake_sqlite(self, sqlite_file, **kwds):
        """Dump a copy of this database to an sqlite file.
        """
        sqlite_db = Database(ro_host="sqlite:///", rw_host="sqlite:///", db_name=sqlite_file, ormbase=self.ormbase)
        sqlite_db.create_tables()
        
        last_size = 0
        for table in self.iter_copy_tables(self, sqlite_db, **kwds):
            size = os.stat(sqlite_file).st_size
            diff = size - last_size
            last_size = size
            print("   sqlite file size:  %0.4fGB  (+%0.4fGB for %s)" % (size*1e-9, diff*1e-9, table))

    def clone_database(self, dest_db_name=None, dest_db=None, overwrite=False, **kwds):
        """Copy this database to a new one.
        """
        if dest_db_name is not None:
            assert isinstance(dest_db_name, str), "Destination DB name bust be a string"
            assert dest_db is None, "Only specify one of dest_db_name or dest_db, not both"
            dest_db = Database(self.ro_host, self.rw_host, dest_db_name, self.ormbase)    
        
        if dest_db.exists:
            if overwrite:
                dest_db.drop_database()
            else:
                raise Exception("Destination database %s already exists." % dest_db)

        dest_db.create_database()
        dest_db.create_tables(initialize=False)
        
        for table in self.iter_copy_tables(self, dest_db, **kwds):
            pass

    @staticmethod
    def iter_copy_tables(source_db, dest_db, tables=None, skip_tables=(), skip_columns={}, skip_errors=False, vacuum=True):
        """Iterator that copies all tables from one database to another.
        
        Yields each table name as it is completed.
        
        This function does not create tables in dest_db; use db.create_tables if needed.
        """
        read_session = source_db.session(readonly=True)
        write_session = dest_db.session(readonly=False)
        
        try:
            if dest_db.backend == 'postgres':
                # disables some consistency checks to allow easier replication
                write_session.execute("SET session_replication_role = 'replica';")

            for table_name, table in source_db.metadata_tables().items():
                if (table_name in skip_tables) or (tables is not None and table_name not in tables):
                    print("Skipping %s.." % table_name)
                    continue
                print("Cloning %s.." % table_name)
                
                # read from table in background thread, write to table in main thread.
                skip_cols = skip_columns.get(table_name, [])
                reader = TableReadThread(source_db, table, skip_columns=skip_cols)
                i = 0
                for i,rec in enumerate(reader):
                    try:
                        # Note: it is allowed to write `rec` directly back to the db, but
                        # in some cases (json columns) we run into a sqlalchemy bug. Converting
                        # to dict first is a workaround.
                        rec = {k:getattr(rec, k) for k in rec.keys()}
                        
                        write_session.execute(table.insert(rec))
                    except Exception:
                        if skip_errors:
                            print("Skip record %d:" % i)
                            sys.excepthook(*sys.exc_info())
                        else:
                            raise
                    if i%1000 == 0:
                        print("%d/%d   %0.2f%%\r" % (i, reader.max_id, (100.0*(i+1.0)/reader.max_id)), end="")
                        sys.stdout.flush()
                    
                print("   committing %d rows..                    " % i)
                write_session.commit()
                read_session.rollback()
                
                yield table_name

            if vacuum:
                print("Optimizing database..")
                dest_db.vacuum()
            print("All finished!")
        finally:
            if dest_db.backend == 'postgres':
                write_session.execute("SET session_replication_role = 'origin';")


class DBQuery(sqlalchemy.orm.Query):
    def add_table_columns(self, table, load_deferred=False):
        """Return a new query with all columns in *table* added.

        Parameters
        ----------
        table : sqlalchemy ORM table
            The table from which columns will be added.
        load_deferred : bool | list
            If True, load all columns that are marked as deferred (by default,
            these are ignored). Optionally, may specify a list of deferred
            column names to load.
        """
        assert isinstance(load_deferred, (list, bool)), "load_deferred must be bool or list"
        meta = sqlalchemy.inspect(table)
        cols = []
        for col in meta.columns.keys():
            load = (
                (not meta.column_attrs[col].deferred) or
                (load_deferred is True) or
                (isinstance(load_deferred, list) and col in load_deferred)
            )
            if load:
                cols.append(col)
        return self.add_columns(*cols)

    def dataframe(self, expand_tables=True, rename_columns=True):
        """Return a pandas dataframe constructed from the results of this query.

        Columns are renamed from the original query (see DBQuery.recarray)

        Parameters
        ----------
        expand_tables : bool | list
            If True, expand all table entities included in the query into individual
            columns. Optionally, a list of table names to expand may be provided instead.
        rename_columns : bool
            If True, columns are renamed (see DBQuery.recarray).
        """
        # don't like this; we want a bit more control over how columns are unpacked / renamed
        if not rename_columns:
            if expand_tables is False:
                raise NotImplementedError("The combination expand_tables=False, rename_columns=False is not implemented")
            return pandas.read_sql(self.statement, self.session.bind)

        recs, col_names, col_types, rec_fields = self._prepare_array(expand_tables=expand_tables)

        # coerce types
        type_map = {float: 'float', int: pandas.Int64Dtype()}
        col_types = [type_map.get(t, 'object') for t in col_types]

        data = {}
        for i, dest_col_name in enumerate(col_names):
            source_col_name = rec_fields[i]
            if isinstance(source_col_name, str):
                col_data = [getattr(rec, source_col_name) for rec in recs]
            elif isinstance(source_col_name, tuple):
                col_data = [getattr(getattr(rec, source_col_name[0]), source_col_name[1], None) for rec in recs]
            data[dest_col_name] = pandas.Series(col_data, dtype=col_types[i])

        return pandas.concat(data, axis=1)

    def recarray(self, expand_tables=True):
        """Return a numpy record array constructed from the results of this query.

        Columns are renamed from the original query based on the following rules:
        - If a column label is explicitly provided, that label is used without modification
        - Result columns that contain a single DB column are renamed to `table.column`
        - Result columns that are derived from more complex expressions use a string representation of the expression
        - Duplicate column names have `_N` appended

        Parameters
        ----------
        expand_tables : bool | list
            If True, expand all table entities included in the query into individual
            columns. Optionally, a list of table names to expand may be provided instead.
        """
        recs, col_names, col_types, rec_fields = self._prepare_array(expand_tables=expand_tables)

        # need to represent everything as either float or obj in order to support null values
        col_types = ['float' if t is float else 'object' for t in col_types]

        dtype = list(zip(col_names, col_types))

        # convert records to numpy array
        if expand_tables is False:
            arr = np.array(recs, dtype=dtype)
        else:
            arr = np.empty(len(recs), dtype=dtype)
            for i, dest_col_name in enumerate(col_names):
                source_col_name = rec_fields[i]
                if isinstance(source_col_name, str):
                    arr[dest_col_name] = [getattr(rec, source_col_name) for rec in recs]
                elif isinstance(source_col_name, tuple):
                    arr[dest_col_name] = [getattr(getattr(rec, source_col_name[0]), source_col_name[1], None) for rec in recs]

        return arr

    def _prepare_array(self, expand_tables):
        recs = self.all()
        row_types = (tuple,)
        try:
            row_types = row_types + (sqlalchemy.engine.row.Row,)
        except AttributeError:
            pass

        if len(recs) > 0 and not isinstance(recs[0], row_types):
            # sqlalchemy returns lists of keyed tuples in most cases, but lists of ORM instances if only one
            # column was requested. This is a pain to handle later on, so we're normalizing the output here.
            rectyp = namedtuple('record', [self.column_descriptions[0]['name']])
            recs = [rectyp(x) for x in recs]

        # decide on column names and dtypes to use
        col_names = []
        col_types = []
        rec_fields = []
        for col in self.column_descriptions:
            try:
                from sqlalchemy.ext.declarative.api import DeclarativeMeta
            except ImportError:
                from sqlalchemy.orm.decl_api import DeclarativeMeta

            if isinstance(col['type'], DeclarativeMeta):
                # this column holds an entire table; use table name unless aliased
                table_name = col['entity'].__table__.name
                aliased_table_name = col['name'] if col['aliased'] else table_name
                expand = (
                    expand_tables is True or (
                        isinstance(expand_tables, list) and (
                            (table_name in expand_tables) or
                            (col['entity'] in expand_tables)
                        )
                    )
                )
                if expand:
                    # Which columns to expand?
                    expanded_cols = self._get_expanded_cols(col, recs)
                    for attribute_name in expanded_cols:
                        col_names.append(aliased_table_name + '.' + attribute_name)
                        rec_fields.append((col['name'], attribute_name))
                        col_types.append(self._get_column_type(getattr(col['entity'], attribute_name)))
                else:
                    col_names.append(aliased_table_name)
                    col_types.append('object')
                    rec_fields.append(col['name'])
            else:
                rec_fields.append(col['name'])

                expr = col['expr']
                if isinstance(expr, sqlalchemy.sql.elements.Label):
                    # query specifies a label here; use that name unconditionally
                    col_names.append(expr.name)
                else:
                    # assign column names as table.column
                    if isinstance(expr, sqlalchemy.orm.attributes.InstrumentedAttribute):
                        table_name = sqlalchemy.inspect(col['entity']).name if col['aliased'] else col['entity'].__table__.name
                        col_names.append(table_name + '.' + col['name'])
                    elif isinstance(expr, sqlalchemy.sql.annotation.AnnotatedColumn):
                        col_names.append(expr.table.name + '.' + expr.name)
                    elif isinstance(expr, sqlalchemy.sql.elements.BinaryExpression):
                        col_names.append(str(col['expr']) if col['name'] is None else col['name'])
                    else:
                        raise TypeError(f"recarray() does not support column of type {repr(expr)} (name: {col['name']})")

                col_types.append(self._get_column_type(expr))

        # modify any repeated names
        seen_names = set()
        for i,name in enumerate(col_names):
            j = 0
            while True:
                new_name = name if j == 0 else f"{name}_{j}"
                j += 1
                if new_name not in seen_names:
                    seen_names.add(new_name)
                    col_names[i] = new_name
                    break

        return recs, col_names, col_types, rec_fields

    def _get_expanded_cols(self, column_desc, records):
        """Return the list of columns to use when expanding one entity column.

        Ideally we'd ask sqlalchemy somehow, but for now we just find the first
        non-null entry and ask which attributes were loaded. If no entries are found, then we can only
        guess and the safest option is to include all columns.
        """
        col_name = column_desc['name']
        first_item = None
        for rec in records:
            first_item = getattr(rec, col_name)
            if first_item is not None:
                break

        insp = sqlalchemy.inspect(column_desc['entity'])
        if getattr(insp, 'is_aliased_class', False):
            insp = insp.mapper
        all_columns = list(insp.columns.keys())
        if first_item is None:
            # no items returned in this query; just guess all columns
            return all_columns
        else:
            unloaded = sqlalchemy.orm.attributes.instance_state(first_item).unloaded
            return [c for c in all_columns if c not in unloaded]

    def _get_column_type(self, column_expr):
        if isinstance(column_expr, sqlalchemy.sql.elements.Label):
            column_expr = column_expr._element
        return column_expr.type.python_type


class TableReadThread(threading.Thread):
    """Iterator that yields records (all columns) from a table.
    
    Records are queried chunkwise and queued in a background thread to enable more efficient streaming.
    """
    def __init__(self, db, table, chunksize=1000, skip_columns=()):
        threading.Thread.__init__(self)
        self.daemon = True
        
        self.db = db
        self.table = table
        self.chunksize = chunksize
        self.skip_columns = skip_columns
        self.queue = queue.Queue(maxsize=5)
        self.max_id = db.session().query(func.max(table.columns['id'])).all()[0][0] or 0
        self.start()
        
    def run(self):
        try:
            session = self.db.session()
            table = self.table
            chunksize = self.chunksize
            all_columns = [col for col in table.columns if col.name not in self.skip_columns]
            for i in range(0, self.max_id, chunksize):
                query = session.query(*all_columns).filter((table.columns['id'] >= i) & (table.columns['id'] < i+chunksize))
                records = query.all()
                self.queue.put(records)
            self.queue.put(None)
            session.rollback()
            session.close()
        except Exception as exc:
            sys.excepthook(*sys.exc_info())
            self.queue.put(exc)
            raise
    
    def __iter__(self):
        while True:
            recs = self.queue.get()
            if recs is None:
                break
            if isinstance(recs, Exception):
                raise recs
            for rec in recs:
                yield rec
