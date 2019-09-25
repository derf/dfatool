#!/usr/bin/env python3

from automata import PTA
import unittest
import yaml

example_json_1 = {
    'parameters' : ['datarate', 'txbytes', 'txpower'],
    'initial_param_values' : [None, None, None],
    'state' : {
        'IDLE' : {
            'power' : {
                'static' : 5,
            }
        },
        'TX' : {
            'power' : {
                'static' : 100,
                'function' : {
                    'raw' : 'regression_arg(0) + regression_arg(1)'
                        ' * parameter(txpower)',
                    'regression_args' : [ 100, 2 ]
                },
            }
        },
    },
    'transitions' : [
        {
            'name' : 'init',
            'origin' : ['UNINITIALIZED', 'IDLE'],
            'destination' : 'IDLE',
            'duration' : {
                'static' : 50000,
            },
            'set_param' : {
                'txpower' : 10
            },
        },
        {
            'name' : 'setTxPower',
            'origin' : 'IDLE',
            'destination' : 'IDLE',
            'duration' : { 'static' : 120 },
            'energy ' : { 'static' : 10000 },
            'arg_to_param_map' : { 'txpower' : 0 },
            'argument_values' : [ [10, 20, 30] ],
        },
        {
            'name' : 'send',
            'origin' : 'IDLE',
            'destination' : 'TX',
            'duration' : {
                'static' : 10,
                'function' : {
                    'raw' : 'regression_arg(0) + regression_arg(1)'
                        ' * function_arg(1)',
                    'regression_args' : [48, 8],
                },
            },
            'energy' : {
                'static' : 3,
                'function' : {
                    'raw' : 'regression_arg(0) + regression_arg(1)'
                        ' * function_arg(1)',
                    'regression_args' : [3, 5],
                },
            },
            'arg_to_param_map' : { 'txbytes' : 1 },
            'argument_values' : [ ['"foo"', '"hodor"'], [3, 5] ],
            'argument_combination' : 'zip',
        },
        {
            'name' : 'txComplete',
            'origin' : 'TX',
            'destination' : 'IDLE',
            'is_interrupt' : 1,
            'timeout' : {
                'static' : 2000,
                'function' : {
                    'raw' : 'regression_arg(0) + regression_arg(1)'
                        ' * parameter(txbytes)',
                    'regression_args' : [ 500, 16 ],
                },
            },
        }
    ],
}

example_yaml_1 = yaml.safe_load("""
codegen:
  instance: cc1200

states:
  - IDLE
  - TX

parameters:
  - datarate
  - txbytes
  - txpower

transition:
  init:
    src: [UNINITIALIZED, IDLE]
    dst: IDLE
  setTxPower:
    src: [IDLE]
    dst: IDLE
    arguments:
      - name: txpower
        values: [10, 20, 30]
        parameter: txpower
  send:
    src: [IDLE]
    dst: TX
    arguments:
      - name: data
        values: ['"foo"', '"hodor"']
      - name: length
        values: [3, 5]
        parameter: txbytes
    argument_combination: zip
  txComplete:
    src: [TX]
    dst: IDLE
    is_interrupt: true
""")

example_yaml_2 = yaml.safe_load("""
codegen:
  instance: cc1200

states:
  - IDLE
  - TX

parameters:
  - datarate
  - txbytes
  - txpower

parameter_normalization:
  txbytes:
    enum:
      FOO: 3
      LONGER: 6
  txpower:
    formula: 'param - 16'

transition:
  init:
    src: [UNINITIALIZED, IDLE]
    dst: IDLE
  setTxPower:
    src: [IDLE]
    dst: IDLE
    arguments:
      - name: txpower
        values: [10, 20, 30]
        parameter: txpower
  send:
    src: [IDLE]
    dst: TX
    arguments:
      - name: data
        values: ['FOO', 'LONGER']
        parameter: txbytes
  txComplete:
    src: [TX]
    dst: IDLE
    is_interrupt: true
""")

def dfs_tran_to_name(runs: list, with_args: bool = False, with_param: bool = False) -> list:
    if with_param:
        return list(map(lambda run: list(map(lambda x: (x[0].name, x[1], x[2]), run)), runs))
    if with_args:
        return list(map(lambda run: list(map(lambda x: (x[0].name, x[1]), run)), runs))
    return list(map(lambda run: list(map(lambda x: (x[0].name), run)), runs))

class TestPTA(unittest.TestCase):
    def test_dfs(self):
        pta = PTA(['IDLE', 'TX'])
        pta.add_transition('UNINITIALIZED', 'IDLE', 'init')
        pta.add_transition('IDLE', 'TX', 'send')
        pta.add_transition('TX', 'IDLE', 'txComplete')
        self.assertEqual(dfs_tran_to_name(pta.dfs(0), False), [['init']])
        self.assertEqual(dfs_tran_to_name(pta.dfs(1), False), [['init', 'send']])
        self.assertEqual(dfs_tran_to_name(pta.dfs(2), False), [['init', 'send', 'txComplete']])
        self.assertEqual(dfs_tran_to_name(pta.dfs(3), False), [['init', 'send', 'txComplete', 'send']])

        pta = PTA(['IDLE'])
        pta.add_transition('UNINITIALIZED', 'IDLE', 'init')
        pta.add_transition('IDLE', 'IDLE', 'set1')
        pta.add_transition('IDLE', 'IDLE', 'set2')
        self.assertEqual(dfs_tran_to_name(pta.dfs(0), False), [['init']])
        self.assertEqual(sorted(dfs_tran_to_name(pta.dfs(1), False)), [['init', 'set1'], ['init', 'set2']])
        self.assertEqual(sorted(dfs_tran_to_name(pta.dfs(2), False)), [['init', 'set1', 'set1'],
            ['init', 'set1', 'set2'],
            ['init', 'set2', 'set1'],
            ['init', 'set2', 'set2']])

    def test_dfs_trace_filter(self):
        pta = PTA(['IDLE'])
        pta.add_transition('UNINITIALIZED', 'IDLE', 'init')
        pta.add_transition('IDLE', 'IDLE', 'set1')
        pta.add_transition('IDLE', 'IDLE', 'set2')
        self.assertEqual(sorted(dfs_tran_to_name(pta.dfs(2, trace_filter=[['init', 'set1', 'set2'], ['init', 'set2', 'set1']]), False)),
            [['init', 'set1', 'set2'], ['init', 'set2', 'set1']])
        self.assertEqual(sorted(dfs_tran_to_name(pta.dfs(2, trace_filter=[['init', 'set1', '$'], ['init', 'set2', '$']]), False)),
            [['init', 'set1'], ['init', 'set2']])

    def test_dfs_accepting(self):
        pta = PTA(['IDLE', 'TX'], accepting_states = ['IDLE'])
        pta.add_transition('UNINITIALIZED', 'IDLE', 'init')
        pta.add_transition('IDLE', 'TX', 'send')
        pta.add_transition('TX', 'IDLE', 'txComplete')
        self.assertEqual(dfs_tran_to_name(pta.dfs(0), False), [['init']])
        self.assertEqual(dfs_tran_to_name(pta.dfs(1), False), [])
        self.assertEqual(dfs_tran_to_name(pta.dfs(2), False), [['init', 'send', 'txComplete']])
        self.assertEqual(dfs_tran_to_name(pta.dfs(3), False), [])

    def test_dfs_objects(self):
        pta = PTA(['IDLE', 'TX'])
        pta.add_transition('UNINITIALIZED', 'IDLE', 'init')
        pta.add_transition('IDLE', 'TX', 'send')
        pta.add_transition('TX', 'IDLE', 'txComplete')
        traces = list(pta.dfs(2))
        self.assertEqual(len(traces), 1)
        trace = traces[0]
        self.assertEqual(len(trace), 3)
        self.assertEqual(trace[0][0].name, 'init')
        self.assertEqual(trace[1][0].name, 'send')
        self.assertEqual(trace[2][0].name, 'txComplete')
        self.assertEqual(pta.get_transition_id(trace[0][0]), 0)
        self.assertEqual(pta.get_transition_id(trace[1][0]), 1)
        self.assertEqual(pta.get_transition_id(trace[2][0]), 2)

    def test_dfs_with_sleep(self):
        pta = PTA(['IDLE', 'TX'])
        pta.add_transition('UNINITIALIZED', 'IDLE', 'init')
        pta.add_transition('IDLE', 'TX', 'send')
        pta.add_transition('TX', 'IDLE', 'txComplete')
        traces = list(pta.dfs(2, sleep = 10))
        self.assertEqual(len(traces), 1)
        trace = traces[0]
        self.assertEqual(len(trace), 6)
        self.assertIsNone(trace[0][0])
        self.assertEqual(trace[1][0].name, 'init')
        self.assertIsNone(trace[2][0])
        self.assertEqual(trace[3][0].name, 'send')
        self.assertIsNone(trace[4][0])
        self.assertEqual(trace[5][0].name, 'txComplete')
        self.assertEqual(pta.get_transition_id(trace[1][0]), 0)
        self.assertEqual(pta.get_transition_id(trace[3][0]), 1)
        self.assertEqual(pta.get_transition_id(trace[5][0]), 2)

    def test_from_json(self):
        pta = PTA.from_json(example_json_1)
        self.assertEqual(pta.parameters, ['datarate', 'txbytes', 'txpower'])
        self.assertEqual(pta.state['UNINITIALIZED'].name, 'UNINITIALIZED')
        self.assertEqual(pta.state['IDLE'].name, 'IDLE')
        self.assertEqual(pta.state['TX'].name, 'TX')
        self.assertEqual(len(pta.transitions), 5)
        self.assertEqual(pta.transitions[0].name, 'init')
        self.assertEqual(pta.transitions[1].name, 'init')
        self.assertEqual(pta.transitions[2].name, 'setTxPower')
        self.assertEqual(pta.transitions[3].name, 'send')
        self.assertEqual(pta.transitions[4].name, 'txComplete')

    #def test_to_json(self):
    #    pta = PTA.from_json(example_json_1)
    #    json = pta.to_json()
    #    json['state'].pop('UNINITIALIZED')
    #    print(json)
    #    self.assertDictEqual(json, example_json_1)

    def test_from_json_dfs_arg(self):
        pta = PTA.from_json(example_json_1)
        self.assertEqual(sorted(dfs_tran_to_name(pta.dfs(1), False)), [['init', 'init'], ['init', 'send'], ['init', 'setTxPower']])
        self.assertEqual(sorted(dfs_tran_to_name(pta.dfs(1, with_arguments = True), True)),
            [
                [('init', ()), ('init', ())],
                [('init', ()), ('send', ('"foo"', 3))],
                [('init', ()), ('send', ('"hodor"', 5))],
                [('init', ()), ('setTxPower', (10,))],
                [('init', ()), ('setTxPower', (20,))],
                [('init', ()), ('setTxPower', (30,))],
            ]
        )

    def test_from_json_dfs_param(self):
        pta = PTA.from_json(example_json_1)
        no_param = {
            'datarate' : None,
            'txbytes' : None,
            'txpower' : None,
        }
        param_tx3 = {
            'datarate' : None,
            'txbytes' : 3,
            'txpower' : None,
        }
        param_tx5 = {
            'datarate' : None,
            'txbytes' : 5,
            'txpower' : None,
        }
        param_txp10 = {
            'datarate' : None,
            'txbytes' : None,
            'txpower' : 10,
        }
        param_txp20 = {
            'datarate' : None,
            'txbytes' : None,
            'txpower' : 20,
        }
        param_txp30 = {
            'datarate' : None,
            'txbytes' : None,
            'txpower' : 30,
        }
        self.assertEqual(sorted(dfs_tran_to_name(pta.dfs(1, with_arguments = True, with_parameters = True), True, True)),
            [
                [('init', (), no_param), ('init', (), no_param)],
                [('init', (), no_param), ('send', ('"foo"', 3), param_tx3)],
                [('init', (), no_param), ('send', ('"hodor"', 5), param_tx5)],
                [('init', (), no_param), ('setTxPower', (10,), param_txp10)],
                [('init', (), no_param), ('setTxPower', (20,), param_txp20)],
                [('init', (), no_param), ('setTxPower', (30,), param_txp30)],
            ]
        )

    def test_from_json_function(self):
        pta = PTA.from_json(example_json_1)
        self.assertEqual(pta.state['TX'].get_energy(1000, {'datarate' : 10, 'txbytes' : 6, 'txpower' : 10 }), 1000 * (100 + 2 * 10))
        self.assertEqual(pta.transitions[4].get_timeout({'datarate' : 10, 'txbytes' : 6, 'txpower' : 10 }), 500 + 16 * 6)

    def test_from_yaml(self):
        pta = PTA.from_yaml(example_yaml_1)

    def test_from_yaml_dfs_param(self):
        pta = PTA.from_yaml(example_yaml_1)
        no_param = {
            'datarate' : None,
            'txbytes' : None,
            'txpower' : None,
        }
        param_tx3 = {
            'datarate' : None,
            'txbytes' : 3,
            'txpower' : None,
        }
        param_tx5 = {
            'datarate' : None,
            'txbytes' : 5,
            'txpower' : None,
        }
        param_txp10 = {
            'datarate' : None,
            'txbytes' : None,
            'txpower' : 10,
        }
        param_txp20 = {
            'datarate' : None,
            'txbytes' : None,
            'txpower' : 20,
        }
        param_txp30 = {
            'datarate' : None,
            'txbytes' : None,
            'txpower' : 30,
        }
        self.assertEqual(sorted(dfs_tran_to_name(pta.dfs(1, with_arguments = True, with_parameters = True), True, True)),
            [
                [('init', (), no_param), ('init', (), no_param)],
                [('init', (), no_param), ('send', ('"foo"', 3), param_tx3)],
                [('init', (), no_param), ('send', ('"hodor"', 5), param_tx5)],
                [('init', (), no_param), ('setTxPower', (10,), param_txp10)],
                [('init', (), no_param), ('setTxPower', (20,), param_txp20)],
                [('init', (), no_param), ('setTxPower', (30,), param_txp30)],
            ]
        )

    def test_normalization(self):
        pta = PTA.from_yaml(example_yaml_2)
        no_param = {
            'datarate' : None,
            'txbytes' : None,
            'txpower' : None,
        }
        param_tx3 = {
            'datarate' : None,
            'txbytes' : 3,
            'txpower' : None,
        }
        param_tx6 = {
            'datarate' : None,
            'txbytes' : 6,
            'txpower' : None,
        }
        param_txp10 = {
            'datarate' : None,
            'txbytes' : None,
            'txpower' : -6,
        }
        param_txp20 = {
            'datarate' : None,
            'txbytes' : None,
            'txpower' : 4,
        }
        param_txp30 = {
            'datarate' : None,
            'txbytes' : None,
            'txpower' : 14,
        }
        self.assertEqual(sorted(dfs_tran_to_name(pta.dfs(1, with_arguments = True, with_parameters = True), True, True)),
            [
                [('init', (), no_param), ('init', (), no_param)],
                [('init', (), no_param), ('send', ('FOO',), param_tx3)],
                [('init', (), no_param), ('send', ('LONGER',), param_tx6)],
                [('init', (), no_param), ('setTxPower', (10,), param_txp10)],
                [('init', (), no_param), ('setTxPower', (20,), param_txp20)],
                [('init', (), no_param), ('setTxPower', (30,), param_txp30)],
            ]
        )

    def test_simulation(self):
        pta = PTA()
        pta.add_state('IDLE', power = 5)
        pta.add_state('TX', power = 100)
        pta.add_transition('UNINITIALIZED', 'IDLE', 'init', duration = 50000)
        pta.add_transition('IDLE', 'TX', 'send', energy = 3, duration = 10)
        pta.add_transition('TX', 'IDLE', 'txComplete', timeout = 2000, is_interrupt = True)
        trace = [
            ['init'],
            [None, 10000000],
            ['send', 'foo', 3],
            [None, 5000000],
            ['send', 'foo', 3]
        ]
        expected_energy = 5. * 10000000 + 3 + 100 * 2000 + 5 * 5000000 + 3 + 100 * 2000
        expected_duration = 50000 + 10000000 + 10 + 2000 + 5000000 + 10 + 2000
        power, duration, state, parameters = pta.simulate(trace)
        self.assertEqual(power, expected_energy)
        self.assertEqual(duration, expected_duration)
        self.assertEqual(state.name, 'IDLE')
        self.assertEqual(parameters, {})

    def test_simulation_param_none(self):
        pta = PTA(parameters = ['txpower', 'length'])
        pta.add_state('IDLE', power = 5)
        pta.add_state('TX', power = 100)
        pta.add_transition('UNINITIALIZED', 'IDLE', 'init', energy = 500000, duration = 50000)
        pta.add_transition('IDLE', 'TX', 'send', energy = 3, duration = 10)
        pta.add_transition('TX', 'IDLE', 'txComplete', timeout = 2000, is_interrupt = True)
        trace = [
            ['init'],
        ]
        expected_energy = 500000
        expected_duration = 50000
        power, duration, state, parameters = pta.simulate(trace)
        self.assertEqual(power, expected_energy)
        self.assertEqual(duration, expected_duration)
        self.assertEqual(state.name, 'IDLE')
        self.assertEqual(parameters, {
            'txpower' : None,
            'length' : None
        })

    def test_simulation_param_update_function(self):
        pta = PTA(parameters = ['txpower', 'length'])
        pta.add_state('IDLE', power = 5)
        pta.add_state('TX', power = 100)
        pta.add_transition('UNINITIALIZED', 'IDLE', 'init', energy = 500000, duration = 50000)
        pta.add_transition('IDLE', 'IDLE', 'setTxPower', energy = 10000, duration = 120,
            param_update_function = lambda param, arg: {**param, 'txpower' : arg[0]})
        pta.add_transition('IDLE', 'TX', 'send', energy = 3, duration = 10)
        pta.add_transition('TX', 'IDLE', 'txComplete', timeout = 2000, is_interrupt = True)
        trace = [
            ['init'],
            ['setTxPower', 10]
        ]
        expected_energy = 510000
        expected_duration = 50120
        power, duration, state, parameters = pta.simulate(trace)
        self.assertEqual(power, expected_energy)
        self.assertEqual(duration, expected_duration)
        self.assertEqual(state.name, 'IDLE')
        self.assertEqual(parameters, {
            'txpower' : 10,
            'length' : None
        })

    def test_simulation_arg_to_param_map(self):
        pta = PTA(parameters = ['txpower', 'length'])
        pta.add_state('IDLE', power = 5)
        pta.add_state('TX', power = 100)
        pta.add_transition('UNINITIALIZED', 'IDLE', 'init', energy = 500000, duration = 50000)
        pta.add_transition('IDLE', 'IDLE', 'setTxPower', energy = 10000, duration = 120,
            arg_to_param_map = {'txpower' : 0})
        pta.add_transition('IDLE', 'TX', 'send', energy = 3, duration = 10)
        pta.add_transition('TX', 'IDLE', 'txComplete', timeout = 2000, is_interrupt = True)
        trace = [
            ['init'],
            ['setTxPower', 10]
        ]
        expected_energy = 510000
        expected_duration = 50120
        power, duration, state, parameters = pta.simulate(trace)
        self.assertEqual(power, expected_energy)
        self.assertEqual(duration, expected_duration)
        self.assertEqual(state.name, 'IDLE')
        self.assertEqual(parameters, {
            'txpower' : 10,
            'length' : None
        })

    def test_simulation_set_param(self):
        pta = PTA(parameters = ['txpower', 'length'])
        pta.add_state('IDLE', power = 5)
        pta.add_state('TX', power = 100)
        pta.add_transition('UNINITIALIZED', 'IDLE', 'init', energy = 500000, duration = 50000, set_param = {'txpower' : 10})
        trace = [
            ['init'],
        ]
        expected_energy = 500000
        expected_duration = 50000
        power, duration, state, parameters = pta.simulate(trace)
        self.assertEqual(power, expected_energy)
        self.assertEqual(duration, expected_duration)
        self.assertEqual(state.name, 'IDLE')
        self.assertEqual(parameters, {
            'txpower' : 10,
            'length' : None
        })

    def test_simulation_arg_function(self):
        pta = PTA(parameters = ['txpower', 'length'])
        pta.add_state('IDLE', power = 5)
        pta.add_state('TX', power = 100)
        pta.add_transition('UNINITIALIZED', 'IDLE', 'init', energy = 500000, duration = 50000)
        pta.add_transition('IDLE', 'IDLE', 'setTxPower', energy = 10000, duration = 120,
            param_update_function = lambda param, arg: {**param, 'txpower' : arg[0]})
        pta.add_transition('IDLE', 'TX', 'send', energy = 3, duration = 10,
            energy_function = lambda param, arg: 3 + 5 * arg[1],
            duration_function = lambda param, arg: 48 + 8 * arg[1])
        pta.add_transition('TX', 'IDLE', 'txComplete', timeout = 2000, is_interrupt = True)
        trace = [
            ['init'],
            ['setTxPower', 10],
            ['send', 'foo', 3],
        ]
        expected_energy = 500000 + 10000 + (3 + 5 * 3) + (2000 * 100)
        expected_duration = 50000 + 120 + (48 + 8 * 3) + 2000
        power, duration, state, parameters = pta.simulate(trace)
        self.assertEqual(power, expected_energy)
        self.assertEqual(duration, expected_duration)
        self.assertEqual(state.name, 'IDLE')
        self.assertEqual(parameters, {
            'txpower' : 10,
            'length' : None
        })

        pta = PTA(parameters = ['txpower', 'length'])
        pta.add_state('IDLE', power = 5)
        pta.add_state('TX', power = 100)
        pta.add_transition('UNINITIALIZED', 'IDLE', 'init', energy = 500000, duration = 50000)
        pta.add_transition('IDLE', 'IDLE', 'setTxPower', energy = 10000, duration = 120,
            param_update_function = lambda param, arg: {**param, 'txpower' : arg[0]})
        pta.add_transition('IDLE', 'TX', 'send', energy = 3, duration = 10,
            energy_function = lambda param, arg: 3 + 5 * arg[1],
            duration_function = lambda param, arg: 48 + 8 * arg[1])
        pta.add_transition('TX', 'IDLE', 'txComplete', timeout = 2000, is_interrupt = True)
        trace = [
            ['init'],
            ['setTxPower', 10],
            ['send', 'foobar', 6],
        ]
        expected_energy = 500000 + 10000 + (3 + 5 * 6) + (2000 * 100)
        expected_duration = 50000 + 120 + (48 + 8 * 6) + 2000
        power, duration, state, parameters = pta.simulate(trace)
        self.assertEqual(power, expected_energy)
        self.assertEqual(duration, expected_duration)
        self.assertEqual(state.name, 'IDLE')
        self.assertEqual(parameters, {
            'txpower' : 10,
            'length' : None
        })


    def test_simulation_param_function(self):
        pta = PTA(parameters = ['length', 'txpower'])
        pta.add_state('IDLE', power = 5)
        pta.add_state('TX', power = 100,
            power_function = lambda param, arg: 1000 + 2 * param[1])
        pta.add_transition('UNINITIALIZED', 'IDLE', 'init', energy = 500000, duration = 50000)
        pta.add_transition('IDLE', 'IDLE', 'setTxPower', energy = 10000, duration = 120,
            param_update_function = lambda param, arg: {**param, 'txpower' : arg[0]})
        pta.add_transition('IDLE', 'TX', 'send', energy = 3, duration = 10,
            energy_function = lambda param, arg: 3 + 5 * arg[1],
            param_update_function = lambda param, arg: {**param, 'length' : arg[1]})
        pta.add_transition('TX', 'IDLE', 'txComplete', timeout = 2000, is_interrupt = True,
            timeout_function = lambda param, arg: 500 + 16 * param[0])
        trace = [
            ['init'],
            ['setTxPower', 10],
            ['send', 'foo', 3],
        ]
        expected_energy = 500000 + 10000 + (3 + 5 * 3) + (1000 + 2 * 10) * (500 + 16 * 3)
        expected_duration = 50000 + 120 + 10 + (500 + 16 * 3)
        power, duration, state, parameters = pta.simulate(trace)
        self.assertEqual(power, expected_energy)
        self.assertEqual(duration, expected_duration)
        self.assertEqual(state.name, 'IDLE')
        self.assertEqual(parameters, {
            'txpower' : 10,
            'length' : 3
        })



if __name__ == '__main__':
    unittest.main()
