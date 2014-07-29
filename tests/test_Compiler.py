import unittest
import numpy as np

from QGL import *

class CompileUtils(unittest.TestCase):
    def setUp(self):
        self.q1gate = Channels.LogicalMarkerChannel(label='q1-gate')
        self.q1 = Qubit(label='q1', gateChan=self.q1gate)
        self.q1.pulseParams['length'] = 30e-9

        self.q2gate = Channels.LogicalMarkerChannel(label='q2-gate')
        self.q2 = Qubit(label='q2', gateChan=self.q2gate)
        self.q2.pulseParams['length'] = 30e-9

        self.measq1 = Channels.Measurement(label='M-q1')
        self.trigger = Channels.LogicalMarkerChannel(label='trigger')

        Compiler.channelLib = {'q1': self.q1, 'q2': self.q2, 'M-q1': self.measq1}

    def test_add_digitizer_trigger(self):
        q1 = self.q1
        seq = [X90(q1), MEAS(q1), Y(q1), MEAS(q1)]

        PatternUtils.add_digitizer_trigger([seq], self.trigger)
        assert(self.trigger in seq[1].pulses.keys())
        assert(self.trigger in seq[3].pulses.keys())

    def test_add_gate_pulses(self):
        q1 = self.q1
        seq = [X90(q1), Y90(q1)]
        PatternUtils.add_gate_pulses([seq])
        assert([self.q1gate in entry.pulses.keys() for entry in seq] == [True, True])

        q2 = self.q2
        seq = [X90(q1), X90(q2), X(q1)*Y(q2)]
        PatternUtils.add_gate_pulses([seq])
        assert([self.q1gate in entry.pulses.keys() for entry in seq] == [True, False, True])
        assert([self.q2gate in entry.pulses.keys() for entry in seq] == [False, True, True])

if __name__ == "__main__":    
    unittest.main()