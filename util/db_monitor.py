"""Simple tool for displaying DB connections, used for debugging performance issues
"""

import os, sys, subprocess, re
from multipatch_analysis import config
from sqlalchemy import create_engine


engine = create_engine(config.synphys_db_host_rw + '/postgres?application_name=db_monitor')



def list_db_connections():
    with engine.begin() as conn:
        result = conn.execute("select client_addr, client_port, datname, query, state, usename, application_name, pid from pg_stat_activity;")
        connections = result.fetchall()
    return connections


_known_hostnames = {}
def hostname(ip):
    global _known_hostnames
    if ip in _known_hostnames:
        return _known_hostnames[ip]
    try:
        host = subprocess.check_output(['host', ip]).partition('pointer ')[2].rstrip('.\n')
    except subprocess.CalledProcessError:
        host = "hostname not found"
    _known_hostnames[ip] = host
    return host


def check():
    print("====================  DB connections ======================")    
    connects = list_db_connections()
    connect_ips = [conn[0] for conn in connects]
    
    ips = list(set(connect_ips))
    counts = {ip:connect_ips.count(ip) for ip in ips}
    ips.sort(key=lambda ip: counts[ip], reverse=True)
    known_addrs = config.known_addrs
    
    for ip in ips:
        if ip is None:
            host = "[None]"
        else:
            host = hostname(ip)
        name = known_addrs.get(ip, known_addrs.get(host, '???'))
        count = counts[ip]
        print("{:10s}{:15s}".format(name, ip))
        
        for con in connects:
            if con[0] == ip:
                app = con.application_name
                for pkg in ['acq4', 'multipatch_analysis']:
                    a,b,c = app.partition(pkg)
                    if b == '':
                        continue
                    else:
                        app = c
                        break
                        
                state = '[%s]' % con.state
                query = con.query.replace('\n', ' ')[:120]
                
                print("          {:15s} {:15s} {:45s} {:6d} {:10s} {:s}   ".format(con.usename, con.datname, app[:45], con.pid, state, query))
    

check()
