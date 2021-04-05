# -*- coding: utf-8 -*-
"""
configfile.py modified from pyqtgraph, used for reading some acq4 data
"""

import re, os, sys, datetime
import numpy
from collections import OrderedDict
from . import units
import PyQt5.QtCore as QtCore
GLOBAL_PATH = None # so not thread safe.


class ParseError(Exception):
    def __init__(self, message, lineNum, line, fileName=None):
        self.lineNum = lineNum
        self.line = line
        self.message = message
        self.fileName = fileName
        Exception.__init__(self, message)
        
    def __str__(self):
        if self.fileName is None:
            msg = "Error parsing string at line %d:\n" % self.lineNum
        else:
            msg = "Error parsing config file '%s' at line %d:\n" % (self.fileName, self.lineNum)
        msg += "%s\n%s" % (self.line, Exception.__str__(self))
        return msg
        

def clip(x, mn, mx):
    if x > mx:
        return mx
    if x < mn:
        return mn
    return x


class Point(QtCore.QPointF):
    """Extension of QPointF which adds a few missing methods."""
    
    def __init__(self, *args):
        if len(args) == 1:
            if isinstance(args[0], QtCore.QSizeF):
                QtCore.QPointF.__init__(self, float(args[0].width()), float(args[0].height()))
                return
            elif isinstance(args[0], float) or isinstance(args[0], int):
                QtCore.QPointF.__init__(self, float(args[0]), float(args[0]))
                return
            elif hasattr(args[0], '__getitem__'):
                QtCore.QPointF.__init__(self, float(args[0][0]), float(args[0][1]))
                return
        elif len(args) == 2:
            QtCore.QPointF.__init__(self, args[0], args[1])
            return
        QtCore.QPointF.__init__(self, *args)
        
    def __len__(self):
        return 2
        
    def __reduce__(self):
        return (Point, (self.x(), self.y()))
        
    def __getitem__(self, i):
        if i == 0:
            return self.x()
        elif i == 1:
            return self.y()
        else:
            raise IndexError("Point has no index %s" % str(i))
        
    def __setitem__(self, i, x):
        if i == 0:
            return self.setX(x)
        elif i == 1:
            return self.setY(x)
        else:
            raise IndexError("Point has no index %s" % str(i))
        
    def __radd__(self, a):
        return self._math_('__radd__', a)
    
    def __add__(self, a):
        return self._math_('__add__', a)
    
    def __rsub__(self, a):
        return self._math_('__rsub__', a)
    
    def __sub__(self, a):
        return self._math_('__sub__', a)
    
    def __rmul__(self, a):
        return self._math_('__rmul__', a)
    
    def __mul__(self, a):
        return self._math_('__mul__', a)
    
    def __rdiv__(self, a):
        return self._math_('__rdiv__', a)
    
    def __div__(self, a):
        return self._math_('__div__', a)
    
    def __truediv__(self, a):
        return self._math_('__truediv__', a)
    
    def __rtruediv__(self, a):
        return self._math_('__rtruediv__', a)
    
    def __rpow__(self, a):
        return self._math_('__rpow__', a)
    
    def __pow__(self, a):
        return self._math_('__pow__', a)
    
    def _math_(self, op, x):
        #print "point math:", op
        #try:
            #fn  = getattr(QtCore.QPointF, op)
            #pt = fn(self, x)
            #print fn, pt, self, x
            #return Point(pt)
        #except AttributeError:
        x = Point(x)
        return Point(getattr(self[0], op)(x[0]), getattr(self[1], op)(x[1]))
    
    def length(self):
        """Returns the vector length of this Point."""
        try:
            return (self[0]**2 + self[1]**2) ** 0.5
        except OverflowError:
            try:
                return self[1] / np.sin(np.arctan2(self[1], self[0]))
            except OverflowError:
                return np.inf
    
    def norm(self):
        """Returns a vector in the same direction with unit length."""
        return self / self.length()
    
    def angle(self, a):
        """Returns the angle in degrees between this vector and the vector a."""
        n1 = self.length()
        n2 = a.length()
        if n1 == 0. or n2 == 0.:
            return None
        ## Probably this should be done with arctan2 instead..
        ang = np.arccos(clip(self.dot(a) / (n1 * n2), -1.0, 1.0)) ### in radians
        c = self.cross(a)
        if c > 0:
            ang *= -1.
        return ang * 180. / np.pi
    
    def dot(self, a):
        """Returns the dot product of a and this Point."""
        a = Point(a)
        return self[0]*a[0] + self[1]*a[1]
    
    def cross(self, a):
        a = Point(a)
        return self[0]*a[1] - self[1]*a[0]
        
    def proj(self, b):
        """Return the projection of this vector onto the vector b"""
        b1 = b / b.length()
        return self.dot(b1) * b1
    
    def __repr__(self):
        return "Point(%f, %f)" % (self[0], self[1])
    
    
    def min(self):
        return min(self[0], self[1])
    
    def max(self):
        return max(self[0], self[1])
        
    def copy(self):
        return Point(self)
        
    def toQPoint(self):
        return QtCore.QPoint(int(self[0]), int(self[1]))


def writeConfigFile(data, fname):
    s = genString(data)
    with open(fname, 'w') as fd:
        fd.write(s)


def readConfigFile(fname):
    #cwd = os.getcwd()
    global GLOBAL_PATH
    if GLOBAL_PATH is not None:
        fname2 = os.path.join(GLOBAL_PATH, fname)
        if os.path.exists(fname2):
            fname = fname2

    GLOBAL_PATH = os.path.dirname(os.path.abspath(fname))
        
    try:
        #os.chdir(newDir)  ## bad.
        with open(fname) as fd:
            s = str(fd.read())
        s = s.replace("\r\n", "\n")
        s = s.replace("\r", "\n")
        data = parseString(s)[1]
    except ParseError:
        sys.exc_info()[1].fileName = fname
        raise
    except:
        print("Error while reading config file %s:"% fname)
        raise
    #finally:
        #os.chdir(cwd)
    return data

def appendConfigFile(data, fname):
    s = genString(data)
    with open(fname, 'a') as fd:
        fd.write(s)


def genString(data, indent=''):
    s = ''
    for k in data:
        sk = str(k)
        if len(sk) == 0:
            print(data)
            raise Exception('blank dict keys not allowed (see data above)')
        if sk[0] == ' ' or ':' in sk:
            print(data)
            raise Exception('dict keys must not contain ":" or start with spaces [offending key is "%s"]' % sk)
        if isinstance(data[k], dict):
            s += indent + sk + ':\n'
            s += genString(data[k], indent + '    ')
        else:
            s += indent + sk + ': ' + repr(data[k]).replace("\n", "\\\n") + '\n'
    return s
    
def parseString(lines, start=0):
    
    data = OrderedDict()
    if isinstance(lines, basestring):
        lines = lines.replace("\\\n", "")
        lines = lines.split('\n')
        lines = [l for l in lines if re.search(r'\S', l) and not re.match(r'\s*#', l)]  ## remove empty lines
        
    indent = measureIndent(lines[start])
    ln = start - 1
    
    try:
        while True:
            ln += 1
            #print ln
            if ln >= len(lines):
                break
            
            l = lines[ln]
            
            ## Skip blank lines or lines starting with #
            if re.match(r'\s*#', l) or not re.search(r'\S', l):
                continue
            
            ## Measure line indentation, make sure it is correct for this level
            lineInd = measureIndent(l)
            if lineInd < indent:
                ln -= 1
                break
            if lineInd > indent:
                #print lineInd, indent
                raise ParseError('Indentation is incorrect. Expected %d, got %d' % (indent, lineInd), ln+1, l)
            
            
            if ':' not in l:
                raise ParseError('Missing colon', ln+1, l)
            
            (k, p, v) = l.partition(':')
            k = k.strip()
            v = v.strip()
            
            ## set up local variables to use for eval
            local = units.allUnits.copy()
            local['OrderedDict'] = OrderedDict
            local['readConfigFile'] = readConfigFile
            local['Point'] = Point
            local['QtCore'] = QtCore
            local['ColorMap'] = ColorMap
            local['datetime'] = datetime
            # Needed for reconstructing numpy arrays
            local['array'] = numpy.array
            for dtype in ['int8', 'uint8', 
                          'int16', 'uint16', 'float16',
                          'int32', 'uint32', 'float32',
                          'int64', 'uint64', 'float64']:
                local[dtype] = getattr(numpy, dtype)
                
            if len(k) < 1:
                raise ParseError('Missing name preceding colon', ln+1, l)
            if k[0] == '(' and k[-1] == ')':  ## If the key looks like a tuple, try evaluating it.
                try:
                    k1 = eval(k, local)
                    if type(k1) is tuple:
                        k = k1
                except:
                    pass
            if re.search(r'\S', v) and v[0] != '#':  ## eval the value
                try:
                    val = eval(v, local)
                except:
                    ex = sys.exc_info()[1]
                    raise ParseError("Error evaluating expression '%s': [%s: %s]" % (v, ex.__class__.__name__, str(ex)), (ln+1), l)
            else:
                if ln+1 >= len(lines) or measureIndent(lines[ln+1]) <= indent:
                    #print "blank dict"
                    val = {}
                else:
                    #print "Going deeper..", ln+1
                    (ln, val) = parseString(lines, start=ln+1)
            data[k] = val
        #print k, repr(val)
    except ParseError:
        raise
    except:
        ex = sys.exc_info()[1]
        raise ParseError("%s: %s" % (ex.__class__.__name__, str(ex)), ln+1, l)
    #print "Returning shallower..", ln+1
    return (ln, data)
    
def measureIndent(s):
    n = 0
    while n < len(s) and s[n] == ' ':
        n += 1
    return n
    
    
    
if __name__ == '__main__':
    import tempfile
    cf = """
key: 'value'
key2:              ##comment
                   ##comment
    key21: 'value' ## comment
                   ##comment
    key22: [1,2,3]
    key23: 234  #comment
    """
    fn = tempfile.mktemp()
    with open(fn, 'w') as tf:
        tf.write(cf)
    print("=== Test:===")
    num = 1
    for line in cf.split('\n'):
        print("%02d   %s" % (num, line))
        num += 1
    print(cf)
    print("============")
    data = readConfigFile(fn)
    print(data)
    os.remove(fn)
