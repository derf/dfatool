#!/usr/bin/env python3

from dfatool import PTAModel, RawData, pta_trace_to_aggregate
import os
import unittest
import pytest


class TestModels(unittest.TestCase):
    def test_model_singlefile_rf24(self):
        raw_data = RawData(['test-data/20170220_164723_RF24_int_A.tar'])
        preprocessed_data = raw_data.get_preprocessed_data(verbose=False)
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = PTAModel(by_name, parameters, arg_count, verbose=False)
        self.assertEqual(model.states(), 'POWERDOWN RX STANDBY1 TX'.split(' '))
        self.assertEqual(model.transitions(), 'begin epilogue powerDown powerUp setDataRate_num setPALevel_num startListening stopListening write_nb'.split(' '))
        static_model = model.get_static()
        self.assertAlmostEqual(static_model('POWERDOWN', 'power'), 0, places=0)
        self.assertAlmostEqual(static_model('RX', 'power'), 52254, places=0)
        self.assertAlmostEqual(static_model('STANDBY1', 'power'), 7, places=0)
        self.assertAlmostEqual(static_model('TX', 'power'), 18414, places=0)
        self.assertAlmostEqual(static_model('begin', 'energy'), 1652249, places=0)
        self.assertAlmostEqual(static_model('epilogue', 'energy'), 15449, places=0)
        self.assertAlmostEqual(static_model('powerDown', 'energy'), 4547, places=0)
        self.assertAlmostEqual(static_model('powerUp', 'energy'), 1641765, places=0)
        self.assertAlmostEqual(static_model('setDataRate_num', 'energy'), 7749, places=0)
        self.assertAlmostEqual(static_model('setPALevel_num', 'energy'), 4700, places=0)
        self.assertAlmostEqual(static_model('startListening', 'energy'), 4309602, places=0)
        self.assertAlmostEqual(static_model('stopListening', 'energy'), 193775, places=0)
        self.assertAlmostEqual(static_model('write_nb', 'energy'), 218339, places=0)
        self.assertAlmostEqual(static_model('begin', 'rel_energy_prev'), 1649571, places=0)
        self.assertAlmostEqual(static_model('epilogue', 'rel_energy_prev'), -744114, places=0)
        self.assertAlmostEqual(static_model('powerDown', 'rel_energy_prev'), 3854, places=0)
        self.assertAlmostEqual(static_model('powerUp', 'rel_energy_prev'), 1641381, places=0)
        self.assertAlmostEqual(static_model('setDataRate_num', 'rel_energy_prev'), 6777, places=0)
        self.assertAlmostEqual(static_model('setPALevel_num', 'rel_energy_prev'), 3728, places=0)
        self.assertAlmostEqual(static_model('startListening', 'rel_energy_prev'), 4307769, places=0)
        self.assertAlmostEqual(static_model('stopListening', 'rel_energy_prev'), -13533693, places=0)
        self.assertAlmostEqual(static_model('write_nb', 'rel_energy_prev'), 214618, places=0)
        self.assertAlmostEqual(static_model('begin', 'duration'), 19830, places=0)
        self.assertAlmostEqual(static_model('epilogue', 'duration'), 40, places=0)
        self.assertAlmostEqual(static_model('powerDown', 'duration'), 90, places=0)
        self.assertAlmostEqual(static_model('powerUp', 'duration'), 10030, places=0)
        self.assertAlmostEqual(static_model('setDataRate_num', 'duration'), 140, places=0)
        self.assertAlmostEqual(static_model('setPALevel_num', 'duration'), 90, places=0)
        self.assertAlmostEqual(static_model('startListening', 'duration'), 260, places=0)
        self.assertAlmostEqual(static_model('stopListening', 'duration'), 260, places=0)
        self.assertAlmostEqual(static_model('write_nb', 'duration'), 510, places=0)

        self.assertAlmostEqual(model.stats.param_dependence_ratio('POWERDOWN', 'power', 'datarate'), 0, places=2)
        self.assertAlmostEqual(model.stats.param_dependence_ratio('POWERDOWN', 'power', 'txbytes'), 0, places=2)
        self.assertAlmostEqual(model.stats.param_dependence_ratio('POWERDOWN', 'power', 'txpower'), 0, places=2)
        self.assertAlmostEqual(model.stats.param_dependence_ratio('RX', 'power', 'datarate'), 0.99, places=2)
        self.assertAlmostEqual(model.stats.param_dependence_ratio('RX', 'power', 'txbytes'), 0, places=2)
        self.assertAlmostEqual(model.stats.param_dependence_ratio('RX', 'power', 'txpower'), 0.01, places=2)
        self.assertAlmostEqual(model.stats.param_dependence_ratio('STANDBY1', 'power', 'datarate'), 0.04, places=2)
        self.assertAlmostEqual(model.stats.param_dependence_ratio('STANDBY1', 'power', 'txbytes'), 0.35, places=2)
        self.assertAlmostEqual(model.stats.param_dependence_ratio('STANDBY1', 'power', 'txpower'), 0.32, places=2)
        self.assertAlmostEqual(model.stats.param_dependence_ratio('TX', 'power', 'datarate'), 1, places=2)
        self.assertAlmostEqual(model.stats.param_dependence_ratio('TX', 'power', 'txbytes'), 0.09, places=2)
        self.assertAlmostEqual(model.stats.param_dependence_ratio('TX', 'power', 'txpower'), 1, places=2)

        param_model, param_info = model.get_fitted()
        self.assertEqual(param_info('POWERDOWN', 'power'), None)
        self.assertEqual(param_info('RX', 'power')['function']._model_str,
                         '0 + regression_arg(0) + regression_arg(1) * np.sqrt(parameter(datarate))')
        self.assertAlmostEqual(param_info('RX', 'power')['function']._regression_args[0], 48530.7, places=0)
        self.assertAlmostEqual(param_info('RX', 'power')['function']._regression_args[1], 117, places=0)
        self.assertEqual(param_info('STANDBY1', 'power'), None)
        self.assertEqual(param_info('TX', 'power')['function']._model_str,
                         '0 + regression_arg(0) + regression_arg(1) * 1/(parameter(datarate)) + regression_arg(2) * parameter(txpower) + regression_arg(3) * 1/(parameter(datarate)) * parameter(txpower)')
        self.assertEqual(param_info('epilogue', 'timeout')['function']._model_str,
                         '0 + regression_arg(0) + regression_arg(1) * 1/(parameter(datarate))')
        self.assertEqual(param_info('stopListening', 'duration')['function']._model_str,
                         '0 + regression_arg(0) + regression_arg(1) * 1/(parameter(datarate))')

        self.assertAlmostEqual(param_model('RX', 'power', param=[1, None, None]), 48647, places=-1)

    def test_model_singlefile_mmparam(self):
        raw_data = RawData(['test-data/20161221_123347_mmparam.tar'])
        preprocessed_data = raw_data.get_preprocessed_data(verbose=False)
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = PTAModel(by_name, parameters, arg_count, verbose=False)
        self.assertEqual(model.states(), 'OFF ON'.split(' '))
        self.assertEqual(model.transitions(), 'off setBrightness'.split(' '))
        static_model = model.get_static()
        self.assertAlmostEqual(static_model('OFF', 'power'), 7124, places=0)
        self.assertAlmostEqual(static_model('ON', 'power'), 17866, places=0)
        self.assertAlmostEqual(static_model('off', 'energy'), 268079197, places=0)
        self.assertAlmostEqual(static_model('setBrightness', 'energy'), 168912773, places=0)
        self.assertAlmostEqual(static_model('off', 'rel_energy_prev'), 105040198, places=0)
        self.assertAlmostEqual(static_model('setBrightness', 'rel_energy_prev'), 103745586, places=0)
        self.assertAlmostEqual(static_model('off', 'duration'), 9130, places=0)
        self.assertAlmostEqual(static_model('setBrightness', 'duration'), 9130, places=0)

        param_lut_model = model.get_param_lut()
        self.assertAlmostEqual(param_lut_model('OFF', 'power', param=[None, None]), 7124, places=0)
        with self.assertRaises(KeyError):
            param_lut_model('ON', 'power', param=[None, None])
            param_lut_model('ON', 'power', param=['a'])
            param_lut_model('ON', 'power', param=[0])
        self.assertTrue(param_lut_model('ON', 'power', param=[0, 0]))
        param_lut_model = model.get_param_lut(fallback=True)
        self.assertAlmostEqual(param_lut_model('ON', 'power', param=[None, None]), 17866, places=0)

    def test_model_multifile_lm75x(self):
        testfiles = [
            'test-data/20170116_124500_LM75x.tar',
            'test-data/20170116_131306_LM75x.tar',
        ]
        raw_data = RawData(testfiles)
        preprocessed_data = raw_data.get_preprocessed_data(verbose=False)
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = PTAModel(by_name, parameters, arg_count, verbose=False)
        self.assertEqual(model.states(), 'ACTIVE POWEROFF'.split(' '))
        self.assertEqual(model.transitions(), 'getTemp setHyst setOS shutdown start'.split(' '))
        static_model = model.get_static()
        self.assertAlmostEqual(static_model('ACTIVE', 'power'), 332, places=0)
        self.assertAlmostEqual(static_model('POWEROFF', 'power'), 7, places=0)
        self.assertAlmostEqual(static_model('getTemp', 'energy'), 26016748, places=0)
        self.assertAlmostEqual(static_model('setHyst', 'energy'), 22082226, places=0)
        self.assertAlmostEqual(static_model('setOS', 'energy'), 21774238, places=0)
        self.assertAlmostEqual(static_model('shutdown', 'energy'), 11808160, places=0)
        self.assertAlmostEqual(static_model('start', 'energy'), 12445302, places=0)
        self.assertAlmostEqual(static_model('getTemp', 'rel_energy_prev'), 21722720, places=0)
        self.assertAlmostEqual(static_model('setHyst', 'rel_energy_prev'), 19001499, places=0)
        self.assertAlmostEqual(static_model('setOS', 'rel_energy_prev'), 18693283, places=0)
        self.assertAlmostEqual(static_model('shutdown', 'rel_energy_prev'), 11746224, places=0)
        self.assertAlmostEqual(static_model('start', 'rel_energy_prev'), 12391462, places=0)
        self.assertAlmostEqual(static_model('getTemp', 'duration'), 12740, places=0)
        self.assertAlmostEqual(static_model('setHyst', 'duration'), 9140, places=0)
        self.assertAlmostEqual(static_model('setOS', 'duration'), 9140, places=0)
        self.assertAlmostEqual(static_model('shutdown', 'duration'), 6980, places=0)
        self.assertAlmostEqual(static_model('start', 'duration'), 6980, places=0)

    def test_model_multifile_sharp(self):
        testfiles = [
            'test-data/20170116_145420_sharpLS013B4DN.tar',
            'test-data/20170116_151348_sharpLS013B4DN.tar',
        ]
        raw_data = RawData(testfiles)
        preprocessed_data = raw_data.get_preprocessed_data(verbose=False)
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = PTAModel(by_name, parameters, arg_count, verbose=False)
        self.assertEqual(model.states(), 'DISABLED ENABLED'.split(' '))
        self.assertEqual(model.transitions(), 'clear disable enable ioInit sendLine toggleVCOM'.split(' '))
        static_model = model.get_static()
        self.assertAlmostEqual(static_model('DISABLED', 'power'), 22, places=0)
        self.assertAlmostEqual(static_model('ENABLED', 'power'), 24, places=0)
        self.assertAlmostEqual(static_model('clear', 'energy'), 14059, places=0)
        self.assertAlmostEqual(static_model('disable', 'energy'), 0, places=0)
        self.assertAlmostEqual(static_model('enable', 'energy'), 0, places=0)
        self.assertAlmostEqual(static_model('ioInit', 'energy'), 0, places=0)
        self.assertAlmostEqual(static_model('sendLine', 'energy'), 37874, places=0)
        self.assertAlmostEqual(static_model('toggleVCOM', 'energy'), 30991, places=0)
        self.assertAlmostEqual(static_model('clear', 'rel_energy_prev'), 13329, places=0)
        self.assertAlmostEqual(static_model('disable', 'rel_energy_prev'), 0, places=0)
        self.assertAlmostEqual(static_model('enable', 'rel_energy_prev'), 0, places=0)
        self.assertAlmostEqual(static_model('ioInit', 'rel_energy_prev'), 0, places=0)
        self.assertAlmostEqual(static_model('sendLine', 'rel_energy_prev'), 33447, places=0)
        self.assertAlmostEqual(static_model('toggleVCOM', 'rel_energy_prev'), 30242, places=0)
        self.assertAlmostEqual(static_model('clear', 'duration'), 30, places=0)
        self.assertAlmostEqual(static_model('disable', 'duration'), 0, places=0)
        self.assertAlmostEqual(static_model('enable', 'duration'), 0, places=0)
        self.assertAlmostEqual(static_model('ioInit', 'duration'), 0, places=0)
        self.assertAlmostEqual(static_model('sendLine', 'duration'), 180, places=0)
        self.assertAlmostEqual(static_model('toggleVCOM', 'duration'), 30, places=0)

    def test_model_multifile_mmstatic(self):
        testfiles = [
            'test-data/20170116_143516_mmstatic.tar',
            'test-data/20170116_142654_mmstatic.tar',
        ]
        raw_data = RawData(testfiles)
        preprocessed_data = raw_data.get_preprocessed_data(verbose=False)
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = PTAModel(by_name, parameters, arg_count, verbose=False)
        self.assertEqual(model.states(), 'B G OFF R'.split(' '))
        self.assertEqual(model.transitions(), 'blue green off red'.split(' '))
        static_model = model.get_static()
        self.assertAlmostEqual(static_model('B', 'power'), 29443, places=0)
        self.assertAlmostEqual(static_model('G', 'power'), 29432, places=0)
        self.assertAlmostEqual(static_model('OFF', 'power'), 7057, places=0)
        self.assertAlmostEqual(static_model('R', 'power'), 49068, places=0)
        self.assertAlmostEqual(static_model('blue', 'energy'), 374440955, places=0)
        self.assertAlmostEqual(static_model('green', 'energy'), 372026027, places=0)
        self.assertAlmostEqual(static_model('off', 'energy'), 372999554, places=0)
        self.assertAlmostEqual(static_model('red', 'energy'), 378936634, places=0)
        self.assertAlmostEqual(static_model('blue', 'rel_energy_prev'), 105535587, places=0)
        self.assertAlmostEqual(static_model('green', 'rel_energy_prev'), 102999371, places=0)
        self.assertAlmostEqual(static_model('off', 'rel_energy_prev'), 103613698, places=0)
        self.assertAlmostEqual(static_model('red', 'rel_energy_prev'), 110474331, places=0)
        self.assertAlmostEqual(static_model('blue', 'duration'), 9140, places=0)
        self.assertAlmostEqual(static_model('green', 'duration'), 9140, places=0)
        self.assertAlmostEqual(static_model('off', 'duration'), 9140, places=0)
        self.assertAlmostEqual(static_model('red', 'duration'), 9140, places=0)

    @pytest.mark.skipif('TEST_SLOW' not in os.environ, reason="slow test, set TEST_SLOW=1 to run")
    def test_model_multifile_cc1200(self):
        testfiles = [
            'test-data/20170125_125433_cc1200.tar',
            'test-data/20170125_142420_cc1200.tar',
            'test-data/20170125_144957_cc1200.tar',
            'test-data/20170125_151149_cc1200.tar',
            'test-data/20170125_151824_cc1200.tar',
            'test-data/20170125_154019_cc1200.tar',
        ]
        raw_data = RawData(testfiles)
        preprocessed_data = raw_data.get_preprocessed_data(verbose=False)
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = PTAModel(by_name, parameters, arg_count, verbose=False)
        self.assertEqual(model.states(), 'IDLE RX SLEEP SLEEP_EWOR SYNTH_ON TX XOFF'.split(' '))
        self.assertEqual(model.transitions(), 'crystal_off eWOR idle init prepare_xmit receive send setSymbolRate setTxPower sleep txDone'.split(' '))
        static_model = model.get_static()
        self.assertAlmostEqual(static_model('IDLE', 'power'), 9500, places=0)
        self.assertAlmostEqual(static_model('RX', 'power'), 85177, places=0)
        self.assertAlmostEqual(static_model('SLEEP', 'power'), 143, places=0)
        self.assertAlmostEqual(static_model('SLEEP_EWOR', 'power'), 81801, places=0)
        self.assertAlmostEqual(static_model('SYNTH_ON', 'power'), 60036, places=0)
        self.assertAlmostEqual(static_model('TX', 'power'), 92461, places=0)
        self.assertAlmostEqual(static_model('XOFF', 'power'), 780, places=0)
        self.assertAlmostEqual(static_model('crystal_off', 'energy'), 114658, places=0)
        self.assertAlmostEqual(static_model('eWOR', 'energy'), 317556, places=0)
        self.assertAlmostEqual(static_model('idle', 'energy'), 717713, places=0)
        self.assertAlmostEqual(static_model('init', 'energy'), 23028941, places=0)
        self.assertAlmostEqual(static_model('prepare_xmit', 'energy'), 378552, places=0)
        self.assertAlmostEqual(static_model('receive', 'energy'), 380335, places=0)
        self.assertAlmostEqual(static_model('send', 'energy'), 4282597, places=0)
        self.assertAlmostEqual(static_model('setSymbolRate', 'energy'), 962060, places=0)
        self.assertAlmostEqual(static_model('setTxPower', 'energy'), 288701, places=0)
        self.assertAlmostEqual(static_model('sleep', 'energy'), 104445, places=0)
        self.assertEqual(static_model('txDone', 'energy'), 0)

        param_model, param_info = model.get_fitted()
        self.assertEqual(param_info('IDLE', 'power'), None)
        self.assertEqual(param_info('RX', 'power')['function']._model_str,
                         '0 + regression_arg(0) + regression_arg(1) * np.log(parameter(symbolrate) + 1)')
        self.assertEqual(param_info('SLEEP', 'power'), None)
        self.assertEqual(param_info('SLEEP_EWOR', 'power'), None)
        self.assertEqual(param_info('SYNTH_ON', 'power'), None)
        self.assertEqual(param_info('XOFF', 'power'), None)

        self.assertAlmostEqual(param_info('RX', 'power')['function']._regression_args[0], 84415, places=0)
        self.assertAlmostEqual(param_info('RX', 'power')['function']._regression_args[1], 206, places=0)


if __name__ == '__main__':
    unittest.main()
