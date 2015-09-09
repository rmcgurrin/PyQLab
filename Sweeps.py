"""
Various sweeps for scanning experiment parameters
"""

from atom.api import Atom, Str, Float, Int, Bool, Dict, List, Enum, \
    Coerced, Property, Typed, observe, cached_property, Int, Signal
import enaml
from enaml.qt.qt_application import QtApplication

from instruments.MicrowaveSources import MicrowaveSource
from instruments.Instrument import Instrument

from DictManager import DictManager

import numpy as np
import json
import floatbits
import FileWatcher
import time
import sys, os


class Sweep(Atom):
    label = Str()
    axisLabel = Str()
    enabled = Bool(True)
    order = Int(-1)

    def json_encode(self, matlabCompatible=False):
        jsonDict = self.__getstate__()
        if matlabCompatible:
            jsonDict['type'] = self.__class__.__name__
            jsonDict.pop('enabled', None)
            jsonDict.pop('name', None)
        else:
            jsonDict['x__class__'] = self.__class__.__name__
            jsonDict['x__module__'] = self.__class__.__module__
        return jsonDict
        
    def update_from_jsondict(self, jsonDict):
		jsonDict.pop('x__class__', None)
		jsonDict.pop('x__module__', None)
		for label,value in jsonDict.items():
			setattr(self, label, value)

class PointsSweep(Sweep):
    """
    A class for sweeps with floating points with one instrument.

    'step' depends on numPoints (but not the other way around) to break the dependency cycle
    """
    start = Float(0.0)
    step = Property()
    stop = Float(1.0)
    numPoints = Int(1)

    def _set_step(self, step):
        # int() will give floor() casted to an Int
        try:
            self.numPoints = int((self.stop - self.start)/floatbits.prevfloat(step)) + 1
        except ValueError, e:
            print("ERROR: Sweep named %s issue computing Num. Points: %s" % (self.label,e))

    def _get_step(self):
        return (self.stop - self.start)/max(1, self.numPoints-1)

    @observe('start', 'stop', 'numPoints')
    def update_step(self, change):
        if change['type'] == 'update':
            # update the step to keep numPoints fixed
            self.get_member('step').reset(self)

class Power(PointsSweep):
    label = Str(default='Power')
    instr = Str()
    units = Enum('dBm', 'Watts').tag(desc='Logarithmic or linear power sweep')

class Frequency(PointsSweep):
    label = Str(default='Frequency')
    instr = Str()

class HeterodyneFrequency(PointsSweep):
    label = Str(default='HeterodyneFrequency')
    instr1 = Str()
    instr2 = Str()
    diffFreq = Float(10.0e-3).tag(desc="IF frequency (GHz)")

class SegmentNum(PointsSweep):
    label = Str(default='SegmentNum')

class SegmentNumWithCals(PointsSweep):
    label = Str(default='SegmentNumWithCals')
    numCals = Int(0)

    def json_encode(self, matlabCompatible=False):
        jsonDict = super(SegmentNumWithCals, self).json_encode(matlabCompatible)
        if matlabCompatible:
            jsonDict['type'] = 'SegmentNum'
            jsonDict['stop'] = self.stop + self.step * self.numCals
            jsonDict['numPoints'] = self.numPoints + self.numCals
        return jsonDict

class Repeat(Sweep):
    label = Str(default='Repeat')
    numRepeats = Int(1).tag(desc='How many times to loop.')

class AWGChannel(PointsSweep):
    label = Str(default='AWGChannel')
    channel = Enum('1','2','3','4','1&2','3&4').tag(desc='Which channel or pair to sweep')
    mode = Enum('Amp.', 'Offset').tag(desc='Sweeping amplitude or offset')
    instr = Str()

class AWGSequence(Sweep):
    label = Str(default='AWGSequence')
    start = Int()
    stop = Int()
    step = Int(1)
    sequenceFile = Str().tag(desc='Base string for the sequence files')

class Attenuation(PointsSweep):
    label = Str(default='Attenuation (dB)')
    channel = Enum(1, 2, 3).tag(desc='Which channel to sweep')
    instr = Str()

class DC(PointsSweep):
    label = Str(default='DC')
    instr = Str()

class Threshold(PointsSweep):
    label = Str(default="Threshold")
    instr = Str()
    stream = Enum('(1,1)','(1,2)','(2,1)','(2,2)').tag(desc='which stream to set threshold')

newSweepClasses = [Power, Frequency, HeterodyneFrequency, Attenuation, SegmentNum, SegmentNumWithCals, AWGChannel, AWGSequence, DC, Repeat, Threshold]

class SweepLibrary(Atom):
    sweepDict = Coerced(dict)
    sweepList = Property()
    sweepOrder = List()
    possibleInstrs = List()
    version = Int(0)

    sweepManager = Typed(DictManager)
    updateItems = Signal()

    libFile = Str()
    fileWatcher = Typed(FileWatcher.LibraryFileWatcher)

    def __init__(self, **kwargs):
        super(SweepLibrary, self).__init__(**kwargs)
        self.load_from_library()
        self.sweepManager = DictManager(itemDict=self.sweepDict,
                                        possibleItems=newSweepClasses)
        if self.libFile:
            self.fileWatcher = FileWatcher.LibraryFileWatcher(self.libFile, self.update_from_file)

    sweepOrder
    @observe('sweepOrder')
    def foo(self,change):
        print(change)
        print('SWEEP ORDER CHANGED')
        #import pdb; pdb.set_trace()
        print(change)
    
    @observe('sweepDict')
    def foo2(self,change):
        print('SWEEP DICT CHANGED')
        print(change)

    #Overload [] to allow direct pulling of sweep info
    def __getitem__(self, sweepName):
        print(sweepName)
        return self.sweepDict[sweepName]

    def _get_sweepList(self):
        #import pdb; pdb.set_trace()
        temp = [sweep.label for sweep in self.sweepDict.values() if sweep.enabled]
        print("GETTING SWEEP LIST")
        print(temp)
        return temp

    def write_to_file(self):
        import JSONHelpers
        if self.libFile:
            #Pause the file watcher to stop circular updating insanity
            if self.fileWatcher:
                self.fileWatcher.pause()
        
            with open(self.libFile, 'w') as FID:
                json.dump(self, FID, cls=JSONHelpers.LibraryEncoder, indent=2, sort_keys=True)
                
            #delay here to allow the OS to generate the file modified event before
            #resuming the file watcher, otherwise you will have a race condition
            #causing multiple file writes
            time.sleep(.1)
            if self.fileWatcher:
                self.fileWatcher.resume()

    def load_from_library(self):
        import JSONHelpers
        if self.libFile:
            try:
                with open(self.libFile, 'r') as FID:
                    try:
                         tmpLib = json.load(FID, cls=JSONHelpers.LibraryDecoder)
                    except ValueError, e:
                         print ("WARNING: JSON object issue: %s in %s" % (e,self.libFile))
                         return

                    if isinstance(tmpLib, SweepLibrary):
                        self.sweepDict.update(tmpLib.sweepDict)
                        del self.possibleInstrs[:]
                        for instr in tmpLib.possibleInstrs:
                            self.possibleInstrs.append(instr)
                        #del self.sweepOrder[:]
                        print("BEFORE ",self.sweepOrder)
                        for sweepStr in tmpLib.sweepOrder:
                            self.sweepOrder.append(sweepStr)
                        # grab library version
                        print("AFTER ",self.sweepOrder)
                        self.version = tmpLib.version
                        #self.sweepList = self.getSweepList()

            except IOError:
                print('No sweep library found.')
                
    def update_from_file(self):
        """
        Only update relevant parameters
        Helps avoid stale references by replacing whole channel objects as in load_from_library
        and the overhead of recreating everything.
        """
        print("UPDATING FROM SWEEPS")
        if self.libFile:
            with open(self.libFile, 'r') as FID:
                try:
                    jsonDict = json.load(FID)
                except ValueError:
                    print('Failed to update instrument library from file.  Probably just half-written.')
                    return
                # update and add new items
                allParams = jsonDict['sweepDict']
                for sweepName, sweepParams in allParams.items():
                    # Re-encode the strings as ascii (this should go away in Python 3)
                    sweepParams = {k.encode('ascii'):v for k,v in sweepParams.items()}
                    # update
                    if sweepName in self.sweepDict:
                        self.sweepDict[sweepName].update_from_jsondict(sweepParams)
                    else:
                        # load class from name and update from json
                        className = sweepParams['x__class__']
                        moduleName = sweepParams['x__module__']
                        print(className,moduleName)

                        #mod = importlib.import_module(moduleName)
                        cls = getattr(sys.modules[moduleName], className)
                        print(cls)
                        self.sweepDict[sweepName]  = cls()
                        self.sweepDict[sweepName].update_from_jsondict(sweepParams)

                # delete removed items
                for sweepName in self.sweepDict.keys():
                    if sweepName not in allParams:
                        del self.sweepDict[sweepName]
                
                '''
                Update the display lists and signal that the list widget needs
                to be updated
                '''
                self.sweepManager.update_display_list_from_file(itemDict=self.sweepDict)

                #self.sweepList = self.getSweepList()
                temp = []
                for sweepStr in jsonDict['sweepOrder']:
                    temp.append(sweepStr)
                self.sweepOrder = temp
                print(self.sweepOrder)
                
                self.updateItems(self._get_sweepList())
                
                
    def json_encode(self, matlabCompatible=False):
            if matlabCompatible:
                #  re-assign based on sweepOrder
                for ct, sweep in enumerate(self.sweepOrder):
                    print(ct,sweep)    
                    self.sweepDict[sweep].order = ct+1
                return {label:sweep for label,sweep in self.sweepDict.items() if label in self.sweepOrder}
            else:
                return {
                    'sweepDict': {label:sweep for label,sweep in self.sweepDict.items()},
                    'sweepOrder': self.sweepOrder,
                    'version': self.version
                }





if __name__ == "__main__":

    from instruments.MicrowaveSources import AgilentN5183A
    testSource1 = AgilentN5183A(label='TestSource1')
    testSource2 = AgilentN5183A(label='TestSource2')
    from Sweeps import Frequency, Power, SegmentNumWithCals, SweepLibrary

    sweepDict = {
        'TestSweep1': Frequency(label='TestSweep1', start=5, step=0.1, stop=6, instr=testSource1.label),
        'TestSweep2': Power(label='TestSweep2', start=-20, stop=0, numPoints=41, instr=testSource2.label),
        'SegWithCals': SegmentNumWithCals(label='SegWithCals', start=0, stop=20, numPoints=101, numCals=4)
    }
    sweepLib = SweepLibrary(possibleInstrs=[testSource1.label, testSource2.label], sweepDict=sweepDict)
    #sweepLib = SweepLibrary(libFile='Sweeps.json')

    with enaml.imports():
        from SweepsViews import SweepManagerWindow
    app = QtApplication()
    view = SweepManagerWindow(sweepLib=sweepLib)
    view.show()

    app.start()
