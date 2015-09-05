# main.py
import enaml
from enaml.qt.qt_application import QtApplication
from atom.api import Atom, Unicode,List,observe,Str,Typed,Signal,Bool,Int
import FileWatcher
import os.path


class Options(Atom):
    optionList = List()
    optionOrder = List()
    optionOrderTemp = List()
    optionOrderUpdate = Bool(False)
    optionOrderLen = Int(0)
    
    fileToWatch = Str()
    fileWatcher = Typed(FileWatcher.LibraryFileWatcher)
    
    somethingChanged = Signal()
    reset=Bool(False)
    
    text=Str()
    
    def __init__(self, **kwargs):
        super(Options, self).__init__(**kwargs)
        if self.fileToWatch:
            print('Watching %s'%self.fileToWatch)
            self.fileWatcher = FileWatcher.LibraryFileWatcher(self.fileToWatch, self.readFile)
            self.somethingChanged.connect(self.update_from_file)
            self.reset = False
            self.optionOrderLen = len(self.optionOrder)
    
    
    @observe('optionOrder')
    def myObserver(self,change):
        print(change)
        
    def update_from_file(self,newOptionOrder):
        print("UPDATING...")
        self.optionOrder = newOptionOrder
        self.optionOrderUpdate = True
        self.optionOrderLen = len(newOptionOrder)
        self.reset = True
        
    def readFile(self):
        print("FILE CHANGED...")
        with open(self.fileToWatch, 'r') as FID:
            try:
                temp=[]
                for line in FID:
                    temp.append([n for n in line.strip().split(',')])
            except ValueError:
                print('Failed to update instrument library from file.  Probably just half-written.')
                return
            
            print(temp[0])
            self.somethingChanged(temp[0])

if __name__ == '__main__':

    with enaml.imports():
        from myExample import OptionView

    options = Options(optionList=['red','green','blue'],optionOrder=['blue','red','green'],fileToWatch=os.path.abspath('options.txt'))
    print(options.optionList)

    app = QtApplication()

    view = OptionView(options=options)
    view.show()

    app.start()