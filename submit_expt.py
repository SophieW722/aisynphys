"""
Script used to submit completed experiment to database.
"""

import acq4.util.Canvas, acq4.util.DataManager
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore


class ExperimentSubmitUi(QtGui.QWidget):
    def __init__(self):
        self.path = None
        
        QtGui.QWidget.__init__(self)
        
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        
        self.hsplit = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.layout.addWidget(self.hsplit, 0, 0)

        self.ctrl_widget = QtGui.QWidget()
        self.ctrl_layout = QtGui.QGridLayout()
        self.ctrl_widget.setLayout(self.ctrl_layout)
        self.hsplit.addWidget(self.ctrl_widget)
        
        self.file_tree = FileTreeWidget()
        self.file_tree.itemSelectionChanged.connect(self.selection_changed)
        self.ctrl_layout.addWidget(self.file_tree, 0, 0, 1, 2)
        
        row = self.ctrl_layout.rowCount()
        self.load_btn = QtGui.QPushButton('load files')
        self.load_btn.clicked.connect(self.load_clicked)
        self.ctrl_layout.addWidget(self.load_btn, row, 0)
        
        self.submit_btn = QtGui.QPushButton('submit')
        self.submit_btn.clicked.connect(self.submit_clicked)
        self.submit_btn.setEnabled(False)
        self.ctrl_layout.addWidget(self.submit_btn, row, 1)
        
        self.canvas = acq4.util.Canvas.Canvas(allowTransforms=False)
        self.hsplit.addWidget(self.canvas)
        
        self.hsplit.setSizes([600, 700])
        
    def set_path(self, path):
        self.path = path
        self.file_tree.set_path(path)

    def load_clicked(self):
        sel = self.file_tree.selectedItems()
        for item in sel:
            fh = item.fh
            self.canvas.addFile(fh)

    def submit_clicked(self):
        pass

    def selection_changed(self):
        sel = self.file_tree.selectedItems()
        sub = len(sel) == 1 and sel[0].is_submittable
        self.submit_btn.setEnabled(sub)


class FileTreeWidget(pg.TreeWidget):
    def __init__(self):
        pg.TreeWidget.__init__(self)
        self.path = None
        self.setColumnCount(3)
        self.setHeaderLabels(['file', 'category', 'metadata'])
        self.setSelectionMode(self.ExtendedSelection)
        self.setDragDropMode(self.NoDragDrop)
        
        # attempts to retain background colors on selected items:
        #self.setAllColumnsShowFocus(False)
        #self.itemSelectionChanged.connect(self._selection_changed)
        #self.style_delegate = StyleDelegate(self)
        #self.setItemDelegateForColumn(1, self.style_delegate)

    def set_path(self, path):
        self.path = path
        self._reload_file_tree()
        
    def _reload_file_tree(self):
        self.clear()
        
        dh = acq4.util.DataManager.getDirHandle(self.path)
        root = self.invisibleRootItem()
        self._fill_tree(dh, root)
        
    def _fill_tree(self, dh, root):
        for fname in dh.ls():
            fh = dh[fname]
            item = self._make_item(fh)
            item.setExpanded(True)
            if hasattr(item, 'type_selected'):
                item.type_selected.connect(self._item_type_selected)
            
            root.addChild(item)
            item.fh = fh
            if fh.isDir():
                self._fill_tree(fh, item)
        
        for i in range(3):
            self.resizeColumnToContents(i)
        
    def _make_item(self, fh):
        info = fh.info()
        objtyp = info.get('__object_type__')
        
        if fh.isDir():
            dirtyp = info.get('dirType', None)
            dtyps = {'Experiment': ExperimentTreeItem, 'Slice': SliceTreeItem, 'Site': SiteTreeItem}
            if dirtyp in dtyps:
                return dtyps[dirtyp](fh)
        if objtyp in ['ImageFile', 'MetaArray']:
            return ImageTreeItem(fh)
        elif fh.shortName().lower().endswith('.nwb'):
            return NwbTreeItem(fh)
        
        item = TypeSelectItem(fh, ['ignore'], 'ignore')
        return item

    def _item_type_selected(self, item, typ):
        for item in self.selectedItems():
            item.set_type(typ)

    ###### attempts to retain background colors on selected items:
    #def _selection_changed(self):
        ## Only select first column
        #try:
            #self.blockSignals(True)
            #for i in self.selectionModel().selectedIndexes():
                #if i.column() != 0:
                    #self.selectionModel().select(i, QtGui.QItemSelectionModel.Deselect)
        #finally:
            #self.blockSignals(False)

    #def mousePressEvent(self, ev):
        #if ev.button() == QtCore.Qt.RightButton:
            #print('press')
            #ev.accept()
        #else:
            #pg.TreeWidget.mousePressEvent(self, ev)

    #def mouseReleaseEvent(self, ev):
        #if ev.button() == QtCore.Qt.RightButton:
            #index = self.indexAt(ev.pos())
            #item, col = self.itemFromIndex(index)
            #print('release', item, col)
            #self._itemClicked(item, col)
        #else:
            #pg.TreeWidget.mouseReleaseEvent(self, ev)


#class StyleDelegate(QtGui.QStyledItemDelegate):
    #def __init__(self, table):
        #QtGui.QStyledItemDelegate.__init__(self)
        #self.table = table
    
    #def paint(self, painter, option, index):
        ##print(index.row(), index.column())
        #QtGui.QStyledItemDelegate.paint(self, painter, option, index)


class ExperimentTreeItem(pg.TreeWidgetItem):
    def __init__(self, fh):
        self.fh = fh
        pg.TreeWidgetItem.__init__(self, [fh.shortName()])


class SliceTreeItem(pg.TreeWidgetItem):
    def __init__(self, fh):
        self.fh = fh
        pg.TreeWidgetItem.__init__(self, [fh.shortName()])


class SiteTreeItem(pg.TreeWidgetItem):
    def __init__(self, fh):
        self.fh = fh
        pg.TreeWidgetItem.__init__(self, [fh.shortName()])


class TypeSelectItem(pg.TreeWidgetItem):
    """TreeWidgetItem with a type selection menu in the second column.
    """
    class Signals(QtCore.QObject):
        type_selected = QtCore.Signal(object, object)
    
    def __init__(self, fh, types, current_type):
        self.is_submittable = False
        self.fh = fh
        self._sigprox = ImageTreeItem.Signals()
        self.type_selected = self._sigprox.type_selected
        self.types = types
        pg.TreeWidgetItem.__init__(self, [fh.shortName(), '', ''])

        self.menu = QtGui.QMenu()
        for typ in self.types:
            act = self.menu.addAction(typ, self._type_selected)
        
        self.set_type(current_type)

    def _type_selected(self):
        action = self.treeWidget().sender()
        text = str(action.text()).strip()
        self.set_type(text)
        self.type_selected.emit(self, text)
            
    def set_type(self, typ):
        self.setText(1, typ)
        if typ == 'ignore':
            self.setBackground(1, pg.mkColor(0.9))
        else:
            self.setBackground(1, pg.mkColor('w'))

    def itemClicked(self, col):
        if col != 1:
            return
        tw = self.treeWidget()
        x = tw.header().sectionPosition(col)
        y = tw.header().height() + tw.visualItemRect(self).bottom()
        self.menu.popup(tw.mapToGlobal(QtCore.QPoint(x, y)))
        return None
        

class NwbTreeItem(TypeSelectItem):
    def __init__(self, fh):
        types = ['ignore', 'MIES physiology']
        TypeSelectItem.__init__(self, fh, types, 'ignore')        
    

class ImageTreeItem(TypeSelectItem):
    def __init__(self, fh):
        info = fh.info()
        meta = info['objective']

        types = ['ignore', 'slice anatomy', 'slice quality stack', 'recording site']
        TypeSelectItem.__init__(self, fh, types, 'ignore')        
        
        self.setText(2, meta)
        colors = info.get('illumination', {}).keys()
        if len(colors) == 0:
            color = 'w'
        elif len(colors) > 1:
            color = 'y'
        else:
            color = {'infrared': (255, 200, 200), 'green': (200, 255, 200), 'blue': (200, 200, 255), 'uv': (240, 200, 255)}[colors[0]]
        self.setBackground(2, pg.mkColor(color))
            




        
if __name__ == '__main__':
    import sys
    app = pg.mkQApp()
    pg.dbg()
    
    path = sys.argv[1]
    ui = ExperimentSubmitUi()
    ui.resize(1300, 800)
    ui.show()
    ui.set_path(path)
    
    if sys.flags.interactive == 0:
        app.exec_()
