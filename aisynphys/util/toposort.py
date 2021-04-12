

def toposort(deps, nodes=None, seen=None, stack=None, depth=0):
    """Topological sort
    
    (credit: pyqtgraph) 
    
    Parameters
    ----------
    deps : dict
        Dictionary describing dependencies where a:[b,c] means "a depends on b and c".
    nodes : list | None
        Optional; specifies list of starting nodes (these should be the nodes 
        which are not depended on by any other nodes). Other candidate starting
        nodes will be ignored.
  

    Example::

        # Sort the following graph:
        # 
        #   B ──┬─────> C <── D
        #       │       │       
        #   E <─┴─> A <─┘
        #     
        deps = {'a': ['b', 'c'], 'c': ['b', 'd'], 'e': ['b']}
        toposort(deps)
         => ['b', 'd', 'c', 'a', 'e']
    """
    # fill in empty dep lists
    deps = deps.copy()
    for k,v in list(deps.items()):
        for k in v:
            if k not in deps:
                deps[k] = []
    
    if nodes is None:
        ## run through deps to find nodes that are not depended upon
        rem = set()
        for dep in deps.values():
            rem |= set(dep)
        nodes = set(deps.keys()) - rem
    if seen is None:
        seen = set()
        stack = []
    sorted = []
    for n in nodes:
        if n in stack:
            raise Exception("Cyclic dependency detected", stack + [n])
        if n in seen:
            continue
        seen.add(n)
        sorted.extend( toposort(deps, deps[n], seen, stack+[n], depth=depth+1))
        sorted.append(n)
    return sorted
