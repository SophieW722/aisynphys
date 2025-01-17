import os, pickle, base64, urllib, json, re
from . import config
from .util.download import interactive_download


_db_versions = None
def list_db_versions():
    """Return a list of database versions that are available for download, sorted by release version.

    Each item in the list is a dictionary with keys db_file, release_version, db_size, and schema_version.
    """
    global _db_versions
    if _db_versions is not None:
        return _db_versions

    # DB urls are stored as a base64-encoded pickle on GitHub.
    # This allows us to change download URLs without requiring users to pull new code.
    # The b64 encoding is just intended to prevent bots scraping our URLs
    b64_urls = urllib.request.urlopen('https://raw.githubusercontent.com/AllenInstitute/aisynphys/download_urls/download_urls').read()
    version_info = pickle.loads(base64.b64decode(b64_urls))

    # parse version and size information from file names
    _db_versions = []
    for name,desc in version_info.items():
        # accepted formats are "synphys_rX.Y_size.sqlite" or "synphys_rX.Y-preZ_size.sqlite"
        #   .. although we do handle one older filename
        m = re.match('synphys_r(\d+\.\d+(-pre\d+)?)(_2019-08-29)?_(small|medium|full).sqlite', name)
        assert m is not None, "unsupported DB file name: " + name
        desc['db_file'] = name
        desc['release_version'] = m.groups()[0]
        desc['db_size'] = m.groups()[3]
        _db_versions.append(desc)

    def version_value(desc):
        m = re.match(r'(\d+)\.(\d+)(-pre(\d+))?', desc['release_version'])
        major, minor, _, pre = m.groups()
        val = int(major) * 1e9 + int(minor) * 1e6
        if pre is not None:
            val = (val - 1000) + int(pre)
        return val
    _db_versions.sort(key=version_value)

    return _db_versions


def get_db_path(db_version):
    """Return the filesystem path of a known database file.
    
    If the file does not exist locally, then it will be downloaded before returning
    the path.
    """
    cache_path = os.path.join(config.cache_path, 'database')
    cache_file = os.path.join(cache_path, db_version)
    
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    if not os.path.exists(cache_file):
        versions = {v['db_file']:v for v in list_db_versions()}
        if db_version not in versions:
            raise KeyError("Unknown database version %r; options are: %s" % (db_version, str(list(versions.keys()))))
        url = versions[db_version]['url']
        interactive_download(url, cache_file)
        
    return cache_file


_file_index = None
def get_data_file_index():
    global _file_index
    if _file_index is None:
        query_url = "http://api.brain-map.org/api/v2/data/WellKnownFile/query.json?criteria=[path$il*synphys*]&num_rows=%d"

        # request number of downloadable files
        count_json = urllib.request.urlopen(query_url % 0).read()
        count = json.loads(count_json)
        if not count['success']:
            raise Exception("Error loading file index: %s" % count['msg'])

        # request full index
        index_json = urllib.request.urlopen(query_url % count['total_rows']).read()
        index = json.loads(index_json)
        if not index['success']:
            raise Exception("Error loading file index: %s" % index['msg'])

        # extract {expt_id:url} mapping from index
        _file_index = {}
        for rec in index['msg']:
            m = re.match(r'.*-(\d+\.\d+)\.nwb$', rec['path'])
            if m is None:
                # skip non-nwb files
                continue
            expt_id = m.groups()[0]
            _file_index[expt_id] = rec['download_link']

    return _file_index


def get_nwb_path(expt_id):
    """Return the local filesystem path to an experiment's nwb file. 

    If the file does not exist locally, then attempt to download.
    """
    cache_path = os.path.join(config.cache_path, 'raw_data_files', expt_id)
    cache_file = os.path.join(cache_path, 'data.nwb')
    
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    if not os.path.exists(cache_file):
        index = get_data_file_index()
        url = index.get(expt_id, None)
        if url is None:
            return None
        url = "http://api.brain-map.org" + url
        interactive_download(url, cache_file)
        
    return cache_file
