"""
SI prefix scaling borrowed form pyqtgraph
"""
import numpy as np


SI_PREFIXES = u'yzafpnµm kMGTPEZY'
SI_PREFIXES_ASCII = 'yzafpnum kMGTPEZY'

    
def si_scale(x, min_val=1e-25, allow_unicode=True):
    """
    Return the recommended scale factor and SI prefix string for x.
    
    Example::
    
        si_scale(0.0001)   # returns (1e6, 'μ')
        # This indicates that the number 0.0001 is best represented as 0.0001 * 1e6 = 100 μUnits
    
    credit: pyqtgraph
    """
    try:
        if np.isnan(x) or np.isinf(x):
            return(1, '')
    except:
        print(x, type(x))
        raise
    if abs(x) < min_val:
        m = 0
        x = 0
    else:
        m = int(np.clip(np.floor(np.log(abs(x))/np.log(1000)), -9.0, 9.0))
    
    if m == 0:
        pref = ''
    elif m < -8 or m > 8:
        pref = 'e%d' % (m*3)
    else:
        if allow_unicode:
            pref = SI_PREFIXES[m+8]
        else:
            pref = SI_PREFIXES_ASCII[m+8]
    p = .001**m
    
    return (p, pref)    


def si_format(x, precision=3, suffix='', float_format='g', space=True, error=None, min_val=1e-25, allow_unicode=True):
    """
    Return the number x formatted in engineering notation with SI prefix.
    
    Example::
        si_format(0.0001, suffix='V')  # returns "100 μV"
    
    credit: pyqtgraph
    """
    
    if space is True:
        space = ' '
    if space is False:
        space = ''
    
    (p, pref) = si_scale(x, min_val, allow_unicode)
    if not (len(pref) > 0 and pref[0] == 'e'):
        pref = space + pref
    
    if error is None:
        fmt = "%." + str(precision) + float_format + "%s%s"
        return fmt % (x*p, pref, suffix)
    else:
        if allow_unicode:
            plusminus = space + u"±" + space
        else:
            plusminus = " +/- "
        fmt = "%." + str(precision) + float_format + "%s%s%s%s"
        return fmt % (x*p, pref, suffix, plusminus, si_format(error, precision=precision, suffix=suffix, space=space, min_val=min_val))

