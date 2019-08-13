#!/usr/bin/env python3

from dfatool import AnalyticModel, TimingData, pta_trace_to_aggregate
import unittest

class TestModels(unittest.TestCase):
    def test_model_singlefile_rf24(self):
        raw_data = TimingData(['test-data/20190724_161440_nRF24_no-rx.json'])
        preprocessed_data = raw_data.get_preprocessed_data(verbose = False)
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = AnalyticModel(by_name, parameters, verbose = False)
        self.assertEqual(model.names, 'setAutoAck setPALevel setRetries setup startListening stopListening write'.split(' '))
        static_model = model.get_static()
        self.assertAlmostEqual(static_model('setAutoAck', 'duration'), 72, places=0)
        self.assertAlmostEqual(static_model('setPALevel', 'duration'), 145, places=0)
        self.assertAlmostEqual(static_model('setRetries', 'duration'), 72, places=0)
        self.assertAlmostEqual(static_model('setup', 'duration'), 6464, places=0)
        self.assertAlmostEqual(static_model('startListening', 'duration'), 455, places=0)
        self.assertAlmostEqual(static_model('stopListening', 'duration'), 487, places=0)
        self.assertAlmostEqual(static_model('write', 'duration'), 5877, places=0)

        for transition in 'setAutoAck setPALevel setRetries setup startListening stopListening'.split(' '):
            self.assertAlmostEqual(model.stats.param_dependence_ratio(transition, 'duration', 'channel'), 0, places=2)


if __name__ == '__main__':
    unittest.main()
