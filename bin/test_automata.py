#!/usr/bin/env python3

from automata import PTA
import unittest

example_json_1 = {
    'parameters' : ['datarate', 'txbytes', 'txpower'],
    'initial_param_values' : [None, None],
    'states' : {
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

class TestPTA(unittest.TestCase):
    def test_dfs(self):
        pta = PTA(['IDLE', 'TX'])
        pta.add_transition('UNINITIALIZED', 'IDLE', 'init')
        pta.add_transition('IDLE', 'TX', 'send')
        pta.add_transition('TX', 'IDLE', 'txComplete')
        self.assertEqual(list(pta.dfs(0)), [['init']])
        self.assertEqual(list(pta.dfs(1)), [['init', 'send']])
        self.assertEqual(list(pta.dfs(2)), [['init', 'send', 'txComplete']])
        self.assertEqual(list(pta.dfs(3)), [['init', 'send', 'txComplete', 'send']])

        pta = PTA(['IDLE'])
        pta.add_transition('UNINITIALIZED', 'IDLE', 'init')
        pta.add_transition('IDLE', 'IDLE', 'set1')
        pta.add_transition('IDLE', 'IDLE', 'set2')
        self.assertEqual(list(pta.dfs(0)), [['init']])
        self.assertEqual(sorted(pta.dfs(1)), [['init', 'set1'], ['init', 'set2']])
        self.assertEqual(sorted(pta.dfs(2)), [['init', 'set1', 'set1'],
            ['init', 'set1', 'set2'],
            ['init', 'set2', 'set1'],
            ['init', 'set2', 'set2']])

    def test_from_json(self):
        pta = PTA.from_json(example_json_1)
        self.assertEqual(pta.parameters, ['datarate', 'txbytes', 'txpower'])
        self.assertEqual(pta.states['UNINITIALIZED'].name, 'UNINITIALIZED')
        self.assertEqual(pta.states['IDLE'].name, 'IDLE')
        self.assertEqual(pta.states['TX'].name, 'TX')
        self.assertEqual(len(pta.transitions), 5)
        self.assertEqual(pta.transitions[0].name, 'init')
        self.assertEqual(pta.transitions[1].name, 'init')
        self.assertEqual(pta.transitions[2].name, 'setTxPower')
        self.assertEqual(pta.transitions[3].name, 'send')
        self.assertEqual(pta.transitions[4].name, 'txComplete')

    def test_from_json_dfs(self):
        pta = PTA.from_json(example_json_1)
        self.assertEqual(sorted(pta.dfs(1)), [['init', 'init'], ['init', 'send'], ['init', 'setTxPower']])

    def test_from_json_function(self):
        pta = PTA.from_json(example_json_1)
        self.assertEqual(pta.states['TX'].get_energy(1000, {'datarate' : 10, 'txbytes' : 6, 'txpower' : 10 }), 1000 * (100 + 2 * 10))
        self.assertEqual(pta.transitions[4].get_timeout({'datarate' : 10, 'txbytes' : 6, 'txpower' : 10 }), 500 + 16 * 6)

    def test_simulation(self):
        pta = PTA()
        pta.add_state('IDLE', power = 5)
        pta.add_state('TX', power = 100)
        pta.add_transition('UNINITIALIZED', 'IDLE', 'init', duration = 50000)
        pta.add_transition('IDLE', 'TX', 'send', energy = 3, duration = 10)
        pta.add_transition('TX', 'IDLE', 'txComplete', timeout = 2000, is_interrupt = True)
        trace = [
            ['init'],
            ['sleep', 10000000],
            ['send', 'foo', 3],
            ['sleep', 5000000],
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
