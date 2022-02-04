from .. import config
from .database import Database
from .synphys_database import SynphysDatabase


class NoDatabase:
    """Raises an exception on first attempt to access an attribute.
    """
    def __init__(self, exception):
        self.exception = exception
    def __getattr__(self, attr):
        raise Exception("Cannot access default databse.") from self.exception


def create_default_db(**kwds):
    """Create a SynphysDatabase using the synphys_db and synphys_db_host config options.
    """
    if config.synphys_db_host is None:
        raise Exception("No database was specified in config.synphys_db_host or with CLI flags --db-version or --db-host")

    if config.synphys_db_host.startswith('postgres'):
        db_name = '{database}_{version}'.format(database=config.synphys_db, version=SynphysDatabase.schema_version)
    else:
        db_name = config.synphys_db

    return SynphysDatabase(config.synphys_db_host, config.synphys_db_host_rw, db_name, **kwds)


def dispose_all_engines():
    """Dispose all engines across all Database instances.
    
    This function should be called before forking.
    """
    Database.dispose_all_engines()


def init_default_db():
    global default_db
    # initialize a default database connection if configured or requested via CLI
    try:
        default_db = create_default_db()
    except Exception as exc:
        default_db = NoDatabase(exc)


init_default_db()
