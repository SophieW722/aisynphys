
def datetime_to_timestamp(d):
    return time.mktime(d.timetuple()) + d.microsecond * 1e-6
    

def timestamp_to_datetime(ts):
    return datetime.datetime.fromtimestamp(ts)


def dir_timestamp(path):
    """Get the timestamp from an index file.

    This is just a very lightweight version of the same functionality provided by ACQ4's DirHandle.info()['__timestamp__'].
    We'd prefer not to duplicate this functionality, but acq4 has UI dependencies that make automated scripting more difficult.
    """
    index_file = os.path.join(path, '.index')
    in_dir = False
    search_indent = None
    for line in open(index_file, 'rb').readlines():
        line = line.decode('latin1')
        if line.startswith('.:'):
            in_dir = True
            continue
        if line[0] != ' ':
            if in_dir is True:
                return None
        if not in_dir:
            continue
        indent = len(line) - len(line.lstrip(' '))
        if search_indent is None:
            search_indent = indent
        if indent != search_indent:
            continue
        line = line.lstrip()
        key = '__timestamp__:'
        if line.startswith(key):
            return float(line[len(key):])
