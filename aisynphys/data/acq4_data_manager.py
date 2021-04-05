# -*- coding: utf-8 -*-
"""
Read-only copy of data management from acq4
"""
import os, re, shutil
from .configfile import *
import time
from PyQt5.QtCore import QMutex as Mutex
from PyQt5 import QtCore as Qt
if not hasattr(QtCore, 'Signal'):
    Qt.Signal = Qt.pyqtSignal
    Qt.Slot = Qt.pyqtSlot
import acq4.filetypes as filetypes
from acq4.util.debug import *


def abspath(fileName):
    """Return an absolute path string which is guaranteed to uniquely identify a file."""
    return os.path.normcase(os.path.abspath(fileName))


def getDataManager():
    inst = DataManager.INSTANCE
    if inst is None:
        raise Exception('No DataManger created yet!')
    return inst


def getHandle(fileName):
    return getDataManager().getHandle(fileName)


def getDirHandle(fileName, create=False):
    return getDataManager().getDirHandle(fileName, create=create)


def getFileHandle(fileName):
    return getDataManager().getFileHandle(fileName)


def cleanup():
    """
    Free memory by deleting cached handles that are not in use elsewhere.
    This is useful in situations where a very large number of handles are
    being created, such as when scanning through large data sets.
    """
    getDataManager().cleanup()


class DataManager(Qt.QObject):
    """Class for creating and caching DirHandle objects to make sure there is only one manager object per file/directory. 
    This class is (supposedly) thread-safe.
    """
    
    INSTANCE = None
    
    def __init__(self):
        Qt.QObject.__init__(self)
        if DataManager.INSTANCE is not None:
            raise Exception("Attempted to create more than one DataManager!")
        DataManager.INSTANCE = self
        self.cache = {}
        self.lock = Mutex(Qt.QMutex.Recursive)
        
    def getDirHandle(self, dirName, create=False):
        with self.lock:
            dirName = os.path.abspath(dirName)
            if not self._cacheHasName(dirName):
                self._addHandle(dirName, DirHandle(dirName, self, create=create))
            return self._getCache(dirName)
        
    def getFileHandle(self, fileName):
        with self.lock:
            fileName = os.path.abspath(fileName)
            if not self._cacheHasName(fileName):
                self._addHandle(fileName, FileHandle(fileName, self))
            return self._getCache(fileName)
        
    def getHandle(self, fileName):
        """Return a FileHandle or DirHandle for the given fileName. 
        If the file does not exist, a handle will still be returned, but is not guaranteed to have the correct type.
        """
        fn = os.path.abspath(fileName)
        if os.path.isdir(fn) or (not os.path.exists(fn) and fn.endswith(os.path.sep)):
            return self.getDirHandle(fileName)
        else:
            return self.getFileHandle(fileName)
        
    def cleanup(self):
        """Attempt to free memory by allowing python to collect any unused handles."""
        import gc
        with self.lock:
            tmp = weakref.WeakValueDictionary(self.cache)
            self.cache = None
            gc.collect()
            self.cache = dict(tmp)

    def _addHandle(self, fileName, handle):
        """Cache a handle and watch it for changes"""
        self._setCache(fileName, handle)
        ## make sure all file handles belong to the main GUI thread
        app = Qt.QApplication.instance()
        if app is not None:
            handle.moveToThread(app.thread())
        ## No signals; handles should explicitly inform the manager of changes
        #Qt.QObject.connect(handle, Qt.SIGNAL('changed'), self._handleChanged)
        
    def _handleChanged(self, handle, change, *args):
        with self.lock:
            if change == 'renamed' or change == 'moved':
                oldName = args[0]
                newName = args[1]
                ## Inform all children that they have been moved and update cache
                tree = self._getTree(oldName)
                for h in tree:
                    ## Update key to cached handle
                    newh = os.path.abspath(os.path.join(newName, h[len(oldName+os.path.sep):]))
                    self._setCache(newh, self._getCache(h))
                    
                    ## If the change originated from h's parent, inform it that this change has occurred.
                    if h != oldName:
                        self._getCache(h)._parentMoved(oldName, newName)
                    self._delCache(h)
                
            elif change == 'deleted':
                oldName = args[0]

                ## Inform all children that they have been deleted and remove from cache
                tree = self._getTree(oldName)
                for path in tree:
                    self._getCache(path)._deleted()
                    self._delCache(path)

    def _getTree(self, parent):
        """Return the entire list of cached handles that are children or grandchildren of this handle"""
        
        ## If handle has no children, then there is no need to search for its tree.
        tree = [parent]
        ph = self._getCache(parent)
        prefix = os.path.normcase(os.path.join(parent, ''))
        
        for h in self.cache:
            if h[:len(prefix)] == prefix:
                tree.append(h)
        return tree

    def _getCache(self, name):
        return self.cache[abspath(name)]
        
    def _setCache(self, name, value):
        self.cache[abspath(name)] = value
        
    def _delCache(self, name):
        del self.cache[abspath(name)]
        
    def _cacheHasName(self, name):
        return abspath(name) in self.cache
        


class FileHandle(Qt.QObject):
    
    sigChanged = Qt.Signal(object, object, object)  # (self, change, (args))
    sigDelayedChange = Qt.Signal(object, object)  # (self, changes)
    
    def __init__(self, path, manager):
        Qt.QObject.__init__(self)
        self.manager = manager
        self.delayedChanges = []
        self.path = os.path.abspath(path)
        self.parentDir = None
        self.lock = Mutex(Qt.QMutex.Recursive)
        
    def getFile(self, fn):
        return getFileHandle(os.path.join(self.name(), fn))
        

    def __repr__(self):
        return "<%s '%s' (0x%x)>" % (self.__class__.__name__, self.name(), self.__hash__())

    def __reduce__(self):
        return (getHandle, (self.name(),))

    def name(self, relativeTo=None):
        """Return the full name of this file with its absolute path"""
        #self.checkExists()
        with self.lock:
            path = self.path
            if relativeTo == self:
                path = ''
            elif relativeTo is not None:
                commonParent = relativeTo
                pcount = 0
                while True:
                    if self is commonParent or self.isGrandchildOf(commonParent):
                        break
                    else:
                        pcount += 1
                        commonParent = commonParent.parent()
                        if commonParent is None:
                            raise Exception("No relative path found from %s to %s." % (relativeTo.name(), self.name()))
                rpath = path[len(os.path.join(commonParent.name(), '')):]
                if pcount == 0:
                    return rpath
                else:
                    ppath = os.path.join(*(['..'] * pcount))
                    if rpath != '':
                        return os.path.join(ppath, rpath)
                    else:
                        return ppath
            return path
        
    def shortName(self):
        """Return the name of this file without its path"""
        #self.checkExists()
        return os.path.split(self.name())[1]

    def ext(self):
        """Return file's extension"""
        return os.path.splitext(self.name())[1]

    def parent(self):
        self.checkExists()
        with self.lock:
            if self.parentDir is None:
                dirName = os.path.split(self.name())[0]
                self.parentDir = self.manager.getDirHandle(dirName)
            return self.parentDir
        
    def info(self):
        self.checkExists()
        info = self.parent()._fileInfo(self.shortName())
        return advancedTypes.ProtectedDict(info)
        
    def isManaged(self):
        self.checkExists()
        return self.parent().isManaged(self.shortName())
        
    def read(self, *args, **kargs):
        self.checkExists()
        with self.lock:
            typ = self.fileType()
            
            if typ is None:
                fd = open(self.name(), 'r')
                data = fd.read()
                fd.close()
            else:
                cls = filetypes.getFileType(typ)
                data = cls.read(self, *args, **kargs)
            
            return data
        
    def fileType(self):
        with self.lock:
            info = self.info()
            
            ## Use the recorded object_type to read the file if possible.
            ## Otherwise, ask the filetypes to choose the type for us.
            if '__object_type__' not in info:
                typ = filetypes.suggestReadType(self)
            else:
                typ = info['__object_type__']
            return typ

    def emitChanged(self, change, *args):
        self.delayedChanges.append(change)
        self.sigChanged.emit(self, change, args)

    def delayedChange(self, args):
        changes = list(set(self.delayedChanges))
        self.delayedChanges = []
        self.sigDelayedChange.emit(self, changes)
    
    def hasChildren(self):
        # self.checkExists()
        return False
    
    def _parentMoved(self, oldDir, newDir):
        """Inform this object that it has been moved as a result of its (grand)parent having moved."""
        prefix = os.path.join(oldDir, '')
        if self.path[:len(prefix)] != prefix:
            raise Exception("File %s is not in moved tree %s, should not update!" % (self.path, oldDir))
        subName = self.path[len(prefix):]
        newName = os.path.join(newDir, subName)
        if not os.path.exists(newName):
            raise Exception("File %s does not exist." % newName)
        self.path = newName
        self.parentDir = None
        self.emitChanged('parent')
        
    def exists(self, name=None):
        if self.path is None:
            return False
        if name is not None:
            raise Exception("Cannot check for subpath existence on FileHandle.")
        return os.path.exists(self.path)

    def checkExists(self):
        if not self.exists():
            raise Exception("File '%s' does not exist." % self.path)

    def checkDeleted(self):
        if self.path is None:
            raise Exception("File has been deleted.")

    def isDir(self, path=None):
        return False
        
    def isFile(self):
        return True
        
    def _deleted(self):
        self.path = None
    
    def isGrandchildOf(self, grandparent):
        """Return true if this files is anywhere in the tree beneath grandparent."""
        gname = os.path.join(abspath(grandparent.name()), '')
        return abspath(self.name())[:len(gname)] == gname


class DirHandle(FileHandle):
    def __init__(self, path, manager, create=False):
        FileHandle.__init__(self, path, manager)
        self._index = None
        self.lsCache = {}  # sortMode: [files...]
        self.cTimeCache = {}
        self._indexFileExists = False
        
        ## Let's avoid reading the index unless we really need to.
        self._indexFileExists = os.path.isfile(self._indexFile())
    
    def _indexFile(self):
        """Return the name of the index file for this directory. NOT the same as indexFile()"""
        return os.path.join(self.path, '.index')
    
    def _logFile(self):
        return os.path.join(self.path, '.log')
    
    def __getitem__(self, item):
        item = item.lstrip(os.path.sep)
        fileName = os.path.join(self.name(), item)
        return self.manager.getHandle(fileName)
    
    def readLog(self, recursive=0):
        """Return a list containing one dict for each log line"""
        with self.lock:
            logf = self._logFile()
            if not os.path.exists(logf):
                log = []
            else:
                try:
                    fd = open(logf, 'r')
                    lines = fd.readlines()
                    fd.close()
                    log = [eval(l.strip()) for l in lines]
                except:
                    print("****************** Error reading log file %s! *********************" % logf)
                    raise
            
            if recursive > 0:
                for d in self.subDirs():
                    dh = self[d]
                    subLog = dh.readLog(recursive=recursive-1)
                    for msg in subLog:
                        if 'subdir' not in msg:
                            msg['subdir'] = ''
                        msg['subdir'] = os.path.join(dh.shortName(), msg['subdir'])
                    log  = log + subLog
                log.sort(lambda a,b: cmp(a['__timestamp__'], b['__timestamp__']))
            
            return log
        
    def subDirs(self):
        """Return a list of string names for all sub-directories."""
        with self.lock:
            ls = self.ls()
            subdirs = [d for d in ls if os.path.isdir(os.path.join(self.name(), d))]
            return subdirs
    
    def incrementFileName(self, fileName, useExt=True):
        """Given fileName.ext, finds the next available fileName_NNN.ext"""
        files = self.ls()
        if useExt:
            (fileName, ext) = os.path.splitext(fileName)
        else:
            ext = ''
        regex = re.compile(fileName + r'_(\d+)')
        files = [f for f in files if regex.match(f)]
        if len(files) > 0:
            files.sort()
            maxVal = int(regex.match(files[-1]).group(1)) + 1
        else:
            maxVal = 0
        ret = fileName + ('_%03d' % maxVal) + ext
        return ret
        
    def getDir(self, subdir, create=False, autoIncrement=False):
        """Return a DirHandle for the specified subdirectory. If the subdir does not exist, it will be created only if create==True"""
        with self.lock:
            ndir = os.path.join(self.path, subdir)
            if not create or os.path.isdir(ndir):
                return self.manager.getDirHandle(ndir)
            else:
                if create:
                    return self.mkdir(subdir, autoIncrement=autoIncrement)
                else:
                    raise Exception('Directory %s does not exist.' % ndir)
        
    def getFile(self, fileName):
        """return a File handle for the named file."""
        fullName = os.path.join(self.name(), fileName)
        fh = self[fileName]
        if not fh.isManaged():
            self.indexFile(fileName)
        return fh
        
    def dirExists(self, dirName):
        return os.path.isdir(os.path.join(self.path, dirName))
            
    def ls(self, normcase=False, sortMode='date', useCache=False):
        """Return a list of all files in the directory.
        If normcase is True, normalize the case of all names in the list.
        sortMode may be 'date', 'alpha', or None."""
        with self.lock:
            if (not useCache) or (sortMode not in self.lsCache):
                self._updateLsCache(sortMode)
            files = self.lsCache[sortMode]
            
            if normcase:
                ret = list(map(os.path.normcase, files))
                return ret
            else:
                ret = files[:]
                return ret
    
    def _updateLsCache(self, sortMode):
        try:
            files = os.listdir(self.name())
        except:
            printExc("Error while listing files in %s:" % self.name())
            files = []
        for i in ['.index', '.log']:
            if i in files:
                files.remove(i)
        
        if sortMode == 'date':
            ## Sort files by creation time
            for f in files:
                if f not in self.cTimeCache:
                    self.cTimeCache[f] = self._getFileCTime(f)
            files.sort(key=lambda f: (self.cTimeCache[f], f))  ## sort by time first, then name.
        elif sortMode == 'alpha':
            ## show directories first when sorting alphabetically.
            files.sort(lambda a,b: 2*cmp(os.path.isdir(os.path.join(self.name(),b)), os.path.isdir(os.path.join(self.name(),a))) + cmp(a,b))
        elif sortMode == None:
            pass
        else:
            raise Exception('Unrecognized sort mode "%s"' % str(sortMode))
            
        self.lsCache[sortMode] = files
    
    def _getFileCTime(self, fileName):
        if self.isManaged():
            index = self._readIndex()
            try:
                t = index[fileName]['__timestamp__']
                return t
            except KeyError:
                pass
            
            ## try getting time directly from file
            try:
                t = self[fileName].info()['__timestamp__']
            except:
                pass
                    
        ## if the file has an obvious date in it, use that
        m = re.search(r'(20\d\d\.\d\d?\.\d\d?)', fileName)
        if m is not None:
            return time.mktime(time.strptime(m.groups()[0], "%Y.%m.%d"))
        
        ## if all else fails, just ask the file system
        return os.path.getctime(os.path.join(self.name(), fileName))
    
    def isGrandparentOf(self, child):
        """Return true if child is anywhere in the tree below this directory."""
        return child.isGrandchildOf(self)
    
    def hasChildren(self):
        return len(self.ls()) > 0
    
    def info(self):
        self._readIndex(unmanagedOk=True)  ## returns None if this directory has no index file
        return advancedTypes.ProtectedDict(self._fileInfo('.'))
    
    def _fileInfo(self, file):
        """Return a dict of the meta info stored for file"""
        with self.lock:
            if not self.isManaged():
                return {}
            index = self._readIndex()
            if file in index:
                return index[file]
            else:
                return {}
    
    def isDir(self, path=None):
        with self.lock:
            if path is None:
                return True
            else:
                return self[path].isDir()
        
    def isFile(self, fileName=None):
        if fileName is None:
            return False
        with self.lock:
            fn = os.path.abspath(os.path.join(self.path, fileName))
            return os.path.isfile(fn)
        
    def isManaged(self, fileName=None):
        with self.lock:
            if self._indexFileExists is False:
                return False
            if fileName is None:
                return True
            else:
                ind = self._readIndex(unmanagedOk=True)
                if ind is None:
                    return False
                return (fileName in ind)

    def exists(self, name=None):
        """Returns True if the file 'name' exists in this directory, False otherwise."""
        with self.lock:
            if self.path is None:
                return False
            if name is None:
                return os.path.exists(self.path)
            
            try:
                fn = os.path.abspath(os.path.join(self.path, name))
            except:
                print(self.path, name)
                raise
            return os.path.exists(fn)
        
    def _readIndex(self, lock=True, unmanagedOk=False):
        with self.lock:
            indexFile = self._indexFile()
            if self._index is None or os.path.getmtime(indexFile) != self._indexMTime:
                if not os.path.isfile(indexFile):
                    if unmanagedOk:
                        return None
                    else:
                        raise Exception("Directory '%s' is not managed!" % (self.name()))
                try:
                    self._index = readConfigFile(indexFile)
                    self._indexMTime = os.path.getmtime(indexFile)
                except:
                    print("***************Error while reading index file %s!*******************" % indexFile)
                    raise
            return self._index
        
    def checkIndex(self):
        ind = self._readIndex(unmanagedOk=True)
        if ind is None:
            return
        changed = False
        for f in ind:
            if not self.exists(f):
                print("File %s is no more, removing from index." % (os.path.join(self.name(), f)))
                del ind[f]
                changed = True
        if changed:
            self._writeIndex(ind)
        
    def _childChanged(self):
        self.lsCache = {}
        self.emitChanged('children')


dm = DataManager()
