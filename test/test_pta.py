#!/usr/bin/env python3

from dfatool.automata import PTA
from dfatool.functions import AnalyticFunction
import unittest
import yaml

example_json_1 = {
    "parameters": ["datarate", "txbytes", "txpower"],
    "initial_param_values": [None, None, None],
    "state": {
        "IDLE": {"power": {"type": "static", "value": 5}},
        "TX": {
            "power": {
                "type": "analytic",
                "value": 100,
                "functionStr": "regression_arg(0) + regression_arg(1) * parameter(txpower)",
                "parameterNames": ["datarate", "txbytes", "txpower"],
                "argCount": 0,
                "regressionModel": [10000, 2],
            }
        },
    },
    "transitions": [
        {
            "name": "init",
            "origin": ["UNINITIALIZED", "IDLE"],
            "destination": "IDLE",
            "duration": {"type": "static", "value": 50000},
            "set_param": {"txpower": 10},
        },
        {
            "name": "setTxPower",
            "origin": "IDLE",
            "destination": "IDLE",
            "duration": {"type": "static", "value": 120},
            "energy ": {"type": "static", "value": 10000},
            "arg_to_param_map": {0: "txpower"},
            "argument_values": [[10, 20, 30]],
        },
        {
            "name": "send",
            "origin": "IDLE",
            "destination": "TX",
            "duration": {
                "type": "analytic",
                "value": 10,
                "functionStr": "regression_arg(0) + regression_arg(1)"
                " * function_arg(1)",
                "parameterNames": ["datarate", "txbytes", "txpower"],
                "argCount": 0,
                "regressionModel": [48, 8],
            },
            "energy": {
                "type": "analytic",
                "value": 3,
                "functionStr": "regression_arg(0) + regression_arg(1)"
                " * function_arg(1)",
                "parameterNames": ["datarate", "txbytes", "txpower"],
                "argCount": 0,
                "regressionModel": [3, 5],
            },
            "arg_to_param_map": {1: "txbytes"},
            "argument_values": [['"foo"', '"hodor"'], [3, 5]],
            "argument_combination": "zip",
        },
        {
            "name": "txComplete",
            "origin": "TX",
            "destination": "IDLE",
            "is_interrupt": 1,
            "timeout": {
                "type": "analytic",
                "value": 2000,
                "functionStr": "regression_arg(0) + regression_arg(1)"
                " * parameter(txbytes)",
                "parameterNames": ["datarate", "txbytes", "txpower"],
                "argCount": 0,
                "regressionModel": [500, 16],
            },
        },
    ],
}

example_yaml_1 = yaml.safe_load(
    """
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
"""
)

example_yaml_2 = yaml.safe_load(
    """
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
"""
)

example_yaml_3 = yaml.safe_load(
    """
codegen:
  instance: nrf24l01
  includes: ['driver/nrf24l01.h']
  flags: ['drivers=nrf24l01', 'arch_drivers=spi_b']

parameters:
  - auto_ack!
  - payload_size
  - dynamic_payloads_enabled!
  - max_retry_count
  - retry_count
  - retry_delay
  - tx_power
  - datarate
  - channel

parameter_normalization:
  tx_power:
    unit: dBm
    enum:
      Nrf24l01::RF24_PA_MIN: 0
      Nrf24l01::RF24_PA_LOW: 6
      Nrf24l01::RF24_PA_HIGH: 12
      Nrf24l01::RF24_PA_MAX: 18
  datarate:
    unit: 'kbit/s'
    enum:
      Nrf24l01::RF24_1MBPS: 1000
      Nrf24l01::RF24_2MBPS: 2000
      Nrf24l01::RF24_250KBPS: 250
  retry_delay:
    unit: us
    formula: '250 * param + 250'

states:
  - UNINITIALIZED
  - STANDBY1

transition:
  setup:
    src: [UNINITIALIZED, STANDBY1]
    dst: STANDBY1
    set_param:
      'auto_ack!': 1
      'dynamic_payloads_enabled!': 0
      max_retry_count: 10
      retry_delay: 5
      datarate: 'Nrf24l01::RF24_1MBPS'
      tx_power: 'Nrf24l01::RF24_PA_MAX'
      channel: 76
  setAutoAck:
    src: [STANDBY1]
    dst: STANDBY1
    arguments:
      - name: enable
        values: [0, 1]
        parameter: 'auto_ack!'
  setPALevel:
    src: [STANDBY1]
    dst: STANDBY1
    arguments:
      - name: palevel
        values: ['Nrf24l01::RF24_PA_MIN', 'Nrf24l01::RF24_PA_LOW', 'Nrf24l01::RF24_PA_HIGH', 'Nrf24l01::RF24_PA_MAX']
        parameter: tx_power
  setRetries:
    src: [STANDBY1]
    dst: STANDBY1
    arguments:
      - name: delay
        values: [0, 5, 10, 15]
        parameter: retry_delay
      - name: count
        values: [0, 5, 10, 15]
        parameter: max_retry_count
  write:
    src: [STANDBY1]
    dst: STANDBY1
    arguments:
      - name: buf
        values: ['"foo"', '"foo"', '"foofoofoo"', '"foofoofoo"', '"123456789012345678901234567890"', '"123456789012345678901234567890"']
      - name: len
        values: [3, 3, 9, 9, 30, 30]
        parameter: payload_size
      - name: await_ack
        values: [0, 1, 0, 1, 0, 1]
      - name: blocking
        values: [1, 1, 1, 1, 1, 1]
    argument_combination: zip
"""
)


def dfs_tran_to_name(
    runs: list, with_args: bool = False, with_param: bool = False
) -> list:
    if with_param:
        return list(
            map(lambda run: list(map(lambda x: (x[0].name, x[1], x[2]), run)), runs)
        )
    if with_args:
        return list(map(lambda run: list(map(lambda x: (x[0].name, x[1]), run)), runs))
    return list(map(lambda run: list(map(lambda x: (x[0].name), run)), runs))


class TestPTA(unittest.TestCase):
    def test_dfs(self):
        pta = PTA(["IDLE", "TX"])
        pta.add_transition("UNINITIALIZED", "IDLE", "init")
        pta.add_transition("IDLE", "TX", "send")
        pta.add_transition("TX", "IDLE", "txComplete")
        self.assertEqual(dfs_tran_to_name(pta.dfs(0), False), [["init"]])
        self.assertEqual(dfs_tran_to_name(pta.dfs(1), False), [["init", "send"]])
        self.assertEqual(
            dfs_tran_to_name(pta.dfs(2), False), [["init", "send", "txComplete"]]
        )
        self.assertEqual(
            dfs_tran_to_name(pta.dfs(3), False),
            [["init", "send", "txComplete", "send"]],
        )

        pta = PTA(["IDLE"])
        pta.add_transition("UNINITIALIZED", "IDLE", "init")
        pta.add_transition("IDLE", "IDLE", "set1")
        pta.add_transition("IDLE", "IDLE", "set2")
        self.assertEqual(dfs_tran_to_name(pta.dfs(0), False), [["init"]])
        self.assertEqual(
            sorted(dfs_tran_to_name(pta.dfs(1), False)),
            [["init", "set1"], ["init", "set2"]],
        )
        self.assertEqual(
            sorted(dfs_tran_to_name(pta.dfs(2), False)),
            [
                ["init", "set1", "set1"],
                ["init", "set1", "set2"],
                ["init", "set2", "set1"],
                ["init", "set2", "set2"],
            ],
        )

    def test_dfs_trace_filter(self):
        pta = PTA(["IDLE"])
        pta.add_transition("UNINITIALIZED", "IDLE", "init")
        pta.add_transition("IDLE", "IDLE", "set1")
        pta.add_transition("IDLE", "IDLE", "set2")
        self.assertEqual(
            sorted(
                dfs_tran_to_name(
                    pta.dfs(
                        2,
                        trace_filter=[
                            ["init", "set1", "set2"],
                            ["init", "set2", "set1"],
                        ],
                    ),
                    False,
                )
            ),
            [["init", "set1", "set2"], ["init", "set2", "set1"]],
        )
        self.assertEqual(
            sorted(
                dfs_tran_to_name(
                    pta.dfs(
                        2, trace_filter=[["init", "set1", "$"], ["init", "set2", "$"]]
                    ),
                    False,
                )
            ),
            [["init", "set1"], ["init", "set2"]],
        )

    def test_dfs_accepting(self):
        pta = PTA(["IDLE", "TX"], accepting_states=["IDLE"])
        pta.add_transition("UNINITIALIZED", "IDLE", "init")
        pta.add_transition("IDLE", "TX", "send")
        pta.add_transition("TX", "IDLE", "txComplete")
        self.assertEqual(dfs_tran_to_name(pta.dfs(0), False), [["init"]])
        self.assertEqual(dfs_tran_to_name(pta.dfs(1), False), [])
        self.assertEqual(
            dfs_tran_to_name(pta.dfs(2), False), [["init", "send", "txComplete"]]
        )
        self.assertEqual(dfs_tran_to_name(pta.dfs(3), False), [])

    def test_dfs_objects(self):
        pta = PTA(["IDLE", "TX"])
        pta.add_transition("UNINITIALIZED", "IDLE", "init")
        pta.add_transition("IDLE", "TX", "send")
        pta.add_transition("TX", "IDLE", "txComplete")
        traces = list(pta.dfs(2))
        self.assertEqual(len(traces), 1)
        trace = traces[0]
        self.assertEqual(len(trace), 3)
        self.assertEqual(trace[0][0].name, "init")
        self.assertEqual(trace[1][0].name, "send")
        self.assertEqual(trace[2][0].name, "txComplete")
        self.assertEqual(pta.get_transition_id(trace[0][0]), 0)
        self.assertEqual(pta.get_transition_id(trace[1][0]), 1)
        self.assertEqual(pta.get_transition_id(trace[2][0]), 2)

    def test_dfs_with_sleep(self):
        pta = PTA(["IDLE", "TX"])
        pta.add_transition("UNINITIALIZED", "IDLE", "init")
        pta.add_transition("IDLE", "TX", "send")
        pta.add_transition("TX", "IDLE", "txComplete")
        traces = list(pta.dfs(2, sleep=10))
        self.assertEqual(len(traces), 1)
        trace = traces[0]
        self.assertEqual(len(trace), 6)
        self.assertIsNone(trace[0][0])
        self.assertEqual(trace[1][0].name, "init")
        self.assertIsNone(trace[2][0])
        self.assertEqual(trace[3][0].name, "send")
        self.assertIsNone(trace[4][0])
        self.assertEqual(trace[5][0].name, "txComplete")
        self.assertEqual(pta.get_transition_id(trace[1][0]), 0)
        self.assertEqual(pta.get_transition_id(trace[3][0]), 1)
        self.assertEqual(pta.get_transition_id(trace[5][0]), 2)

    def test_bfs(self):
        pta = PTA(["IDLE", "TX"])
        pta.add_transition("UNINITIALIZED", "IDLE", "init")
        pta.add_transition("IDLE", "TX", "send")
        pta.add_transition("TX", "IDLE", "txComplete")
        self.assertEqual(dfs_tran_to_name(pta.bfs(0), False), [["init"]])
        self.assertEqual(
            dfs_tran_to_name(pta.bfs(1), False), [["init"], ["init", "send"]]
        )
        self.assertEqual(
            dfs_tran_to_name(pta.bfs(2), False),
            [["init"], ["init", "send"], ["init", "send", "txComplete"]],
        )
        self.assertEqual(
            dfs_tran_to_name(pta.bfs(3), False),
            [
                ["init"],
                ["init", "send"],
                ["init", "send", "txComplete"],
                ["init", "send", "txComplete", "send"],
            ],
        )

        pta = PTA(["IDLE"])
        pta.add_transition("UNINITIALIZED", "IDLE", "init")
        pta.add_transition("IDLE", "IDLE", "set1")
        pta.add_transition("IDLE", "IDLE", "set2")
        self.assertEqual(dfs_tran_to_name(pta.bfs(0), False), [["init"]])
        self.assertEqual(
            sorted(dfs_tran_to_name(pta.bfs(1), False)),
            [["init"], ["init", "set1"], ["init", "set2"]],
        )
        self.assertEqual(
            sorted(dfs_tran_to_name(pta.bfs(2), False)),
            [
                ["init"],
                ["init", "set1"],
                ["init", "set1", "set1"],
                ["init", "set1", "set2"],
                ["init", "set2"],
                ["init", "set2", "set1"],
                ["init", "set2", "set2"],
            ],
        )

    def test_from_json(self):
        pta = PTA.from_json(example_json_1)
        self.assertEqual(pta.parameters, ["datarate", "txbytes", "txpower"])
        self.assertEqual(pta.state["UNINITIALIZED"].name, "UNINITIALIZED")
        self.assertEqual(pta.state["IDLE"].name, "IDLE")
        self.assertEqual(pta.state["TX"].name, "TX")
        self.assertEqual(len(pta.transitions), 5)
        self.assertEqual(pta.transitions[0].name, "init")
        self.assertEqual(pta.transitions[1].name, "init")
        self.assertEqual(pta.transitions[2].name, "setTxPower")
        self.assertEqual(pta.transitions[3].name, "send")
        self.assertEqual(pta.transitions[4].name, "txComplete")

    # def test_to_json(self):
    #    pta = PTA.from_json(example_json_1)
    #    json = pta.to_json()
    #    json['state'].pop('UNINITIALIZED')
    #    print(json)
    #    self.assertDictEqual(json, example_json_1)

    def test_from_json_dfs_arg(self):
        pta = PTA.from_json(example_json_1)
        self.assertEqual(
            sorted(dfs_tran_to_name(pta.dfs(1), False)),
            [["init", "init"], ["init", "send"], ["init", "setTxPower"]],
        )
        self.assertEqual(
            sorted(dfs_tran_to_name(pta.dfs(1, with_arguments=True), True)),
            [
                [("init", ()), ("init", ())],
                [("init", ()), ("send", ('"foo"', 3))],
                [("init", ()), ("send", ('"hodor"', 5))],
                [("init", ()), ("setTxPower", (10,))],
                [("init", ()), ("setTxPower", (20,))],
                [("init", ()), ("setTxPower", (30,))],
            ],
        )

    def test_from_json_dfs_param(self):
        pta = PTA.from_json(example_json_1)
        no_param = {"datarate": None, "txbytes": None, "txpower": 10}
        param_tx3 = {"datarate": None, "txbytes": 3, "txpower": 10}
        param_tx5 = {"datarate": None, "txbytes": 5, "txpower": 10}
        param_txp10 = {"datarate": None, "txbytes": None, "txpower": 10}
        param_txp20 = {"datarate": None, "txbytes": None, "txpower": 20}
        param_txp30 = {"datarate": None, "txbytes": None, "txpower": 30}
        self.assertEqual(
            sorted(
                dfs_tran_to_name(
                    pta.dfs(1, with_arguments=True, with_parameters=True), True, True
                )
            ),
            [
                [("init", (), no_param), ("init", (), no_param)],
                [("init", (), no_param), ("send", ('"foo"', 3), param_tx3)],
                [("init", (), no_param), ("send", ('"hodor"', 5), param_tx5)],
                [("init", (), no_param), ("setTxPower", (10,), param_txp10)],
                [("init", (), no_param), ("setTxPower", (20,), param_txp20)],
                [("init", (), no_param), ("setTxPower", (30,), param_txp30)],
            ],
        )

    def test_from_json_function(self):
        pta = PTA.from_json(example_json_1)
        self.assertEqual(
            pta.state["TX"].get_energy(
                1000, {"datarate": 10, "txbytes": 6, "txpower": 10}
            ),
            1000 * (10000 + 2 * 10),
        )
        self.assertEqual(
            pta.transitions[4].get_timeout(
                {"datarate": 10, "txbytes": 6, "txpower": 10}
            ),
            500 + 16 * 6,
        )

    def test_from_yaml_dfs_param(self):
        pta = PTA.from_yaml(example_yaml_1)
        no_param = {"datarate": None, "txbytes": None, "txpower": None}
        param_tx3 = {"datarate": None, "txbytes": 3, "txpower": None}
        param_tx5 = {"datarate": None, "txbytes": 5, "txpower": None}
        param_txp10 = {"datarate": None, "txbytes": None, "txpower": 10}
        param_txp20 = {"datarate": None, "txbytes": None, "txpower": 20}
        param_txp30 = {"datarate": None, "txbytes": None, "txpower": 30}
        self.assertEqual(
            sorted(
                dfs_tran_to_name(
                    pta.dfs(1, with_arguments=True, with_parameters=True), True, True
                )
            ),
            [
                [("init", (), no_param), ("init", (), no_param)],
                [("init", (), no_param), ("send", ('"foo"', 3), param_tx3)],
                [("init", (), no_param), ("send", ('"hodor"', 5), param_tx5)],
                [("init", (), no_param), ("setTxPower", (10,), param_txp10)],
                [("init", (), no_param), ("setTxPower", (20,), param_txp20)],
                [("init", (), no_param), ("setTxPower", (30,), param_txp30)],
            ],
        )

    def test_normalization(self):
        pta = PTA.from_yaml(example_yaml_2)
        no_param = {"datarate": None, "txbytes": None, "txpower": None}
        param_tx3 = {"datarate": None, "txbytes": 3, "txpower": None}
        param_tx6 = {"datarate": None, "txbytes": 6, "txpower": None}
        param_txp10 = {"datarate": None, "txbytes": None, "txpower": -6}
        param_txp20 = {"datarate": None, "txbytes": None, "txpower": 4}
        param_txp30 = {"datarate": None, "txbytes": None, "txpower": 14}
        self.assertEqual(
            sorted(
                dfs_tran_to_name(
                    pta.dfs(1, with_arguments=True, with_parameters=True), True, True
                )
            ),
            [
                [("init", (), no_param), ("init", (), no_param)],
                [("init", (), no_param), ("send", ("FOO",), param_tx3)],
                [("init", (), no_param), ("send", ("LONGER",), param_tx6)],
                [("init", (), no_param), ("setTxPower", (10,), param_txp10)],
                [("init", (), no_param), ("setTxPower", (20,), param_txp20)],
                [("init", (), no_param), ("setTxPower", (30,), param_txp30)],
            ],
        )

    def test_shrink(self):
        pta = PTA.from_yaml(example_yaml_3)
        pta.shrink_argument_values()
        self.assertEqual(pta.transitions[0].name, "setAutoAck")
        self.assertEqual(pta.transitions[1].name, "setPALevel")
        self.assertEqual(pta.transitions[2].name, "setRetries")
        self.assertEqual(pta.transitions[3].name, "setup")
        self.assertEqual(pta.transitions[4].name, "setup")
        self.assertEqual(pta.transitions[5].name, "write")
        self.assertEqual(pta.transitions[0].argument_values, [[0, 1]])
        self.assertEqual(
            pta.transitions[1].argument_values,
            [["Nrf24l01::RF24_PA_MIN", "Nrf24l01::RF24_PA_MAX"]],
        )
        self.assertEqual(pta.transitions[2].argument_values, [[0, 15], [0, 15]])
        self.assertEqual(
            pta.transitions[5].argument_values,
            [
                [
                    '"foo"',
                    '"foo"',
                    '"foofoofoo"',
                    '"foofoofoo"',
                    '"123456789012345678901234567890"',
                    '"123456789012345678901234567890"',
                ],
                [3, 3, 9, 9, 30, 30],
                [0, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ],
        )

    def test_simulation(self):
        pta = PTA()
        pta.add_state("IDLE", power=5)
        pta.add_state("TX", power=100)
        pta.add_transition("UNINITIALIZED", "IDLE", "init", duration=50000)
        pta.add_transition("IDLE", "TX", "send", energy=3, duration=10)
        pta.add_transition("TX", "IDLE", "txComplete", timeout=2000, is_interrupt=True)
        trace = [
            ["init"],
            [None, 10000000],
            ["send", "foo", 3],
            [None, 5000000],
            ["send", "foo", 3],
        ]
        expected_energy = 5.0 * 10000000 + 3 + 100 * 2000 + 5 * 5000000 + 3 + 100 * 2000
        expected_duration = 50000 + 10000000 + 10 + 2000 + 5000000 + 10 + 2000
        result = pta.simulate(trace)
        self.assertAlmostEqual(result.energy, expected_energy * 1e-12, places=12)
        self.assertAlmostEqual(result.duration, expected_duration * 1e-6, places=6)
        self.assertEqual(result.end_state.name, "IDLE")
        self.assertEqual(result.parameters, {})

    def test_simulation_param_none(self):
        pta = PTA(parameters=["txpower", "length"])
        pta.add_state("IDLE", power=5)
        pta.add_state("TX", power=100)
        pta.add_transition(
            "UNINITIALIZED", "IDLE", "init", energy=500000, duration=50000
        )
        pta.add_transition("IDLE", "TX", "send", energy=3, duration=10)
        pta.add_transition("TX", "IDLE", "txComplete", timeout=2000, is_interrupt=True)
        trace = [["init"]]
        expected_energy = 500000
        expected_duration = 50000
        result = pta.simulate(trace)
        self.assertAlmostEqual(result.energy, expected_energy * 1e-12, places=12)
        self.assertAlmostEqual(result.duration, expected_duration * 1e-6, places=6)
        self.assertEqual(result.end_state.name, "IDLE")
        self.assertEqual(result.parameters, {"txpower": None, "length": None})

    def test_simulation_param_update_function(self):
        pta = PTA(parameters=["txpower", "length"])
        pta.add_state("IDLE", power=5)
        pta.add_state("TX", power=100)
        pta.add_transition(
            "UNINITIALIZED", "IDLE", "init", energy=500000, duration=50000
        )
        pta.add_transition(
            "IDLE",
            "IDLE",
            "setTxPower",
            energy=10000,
            duration=120,
            param_update_function=lambda param, arg: {**param, "txpower": arg[0]},
        )
        pta.add_transition("IDLE", "TX", "send", energy=3, duration=10)
        pta.add_transition("TX", "IDLE", "txComplete", timeout=2000, is_interrupt=True)
        trace = [["init"], ["setTxPower", 10]]
        expected_energy = 510000
        expected_duration = 50120
        result = pta.simulate(trace)
        self.assertAlmostEqual(result.energy, expected_energy * 1e-12, places=12)
        self.assertAlmostEqual(result.duration, expected_duration * 1e-6, places=6)
        self.assertEqual(result.end_state.name, "IDLE")
        self.assertEqual(result.parameters, {"txpower": 10, "length": None})

    def test_simulation_arg_to_param_map(self):
        pta = PTA(parameters=["txpower", "length"])
        pta.add_state("IDLE", power=5)
        pta.add_state("TX", power=100)
        pta.add_transition(
            "UNINITIALIZED", "IDLE", "init", energy=500000, duration=50000
        )
        pta.add_transition(
            "IDLE",
            "IDLE",
            "setTxPower",
            energy=10000,
            duration=120,
            arg_to_param_map={0: "txpower"},
        )
        pta.add_transition("IDLE", "TX", "send", energy=3, duration=10)
        pta.add_transition("TX", "IDLE", "txComplete", timeout=2000, is_interrupt=True)
        trace = [["init"], ["setTxPower", 10]]
        expected_energy = 510000
        expected_duration = 50120
        result = pta.simulate(trace)
        self.assertAlmostEqual(result.energy, expected_energy * 1e-12, places=12)
        self.assertAlmostEqual(result.duration, expected_duration * 1e-6, places=6)
        self.assertEqual(result.end_state.name, "IDLE")
        self.assertEqual(result.parameters, {"txpower": 10, "length": None})

    def test_simulation_set_param(self):
        pta = PTA(parameters=["txpower", "length"])
        pta.add_state("IDLE", power=5)
        pta.add_state("TX", power=100)
        pta.add_transition(
            "UNINITIALIZED",
            "IDLE",
            "init",
            energy=500000,
            duration=50000,
            set_param={"txpower": 10},
        )
        trace = [["init"]]
        expected_energy = 500000
        expected_duration = 50000
        result = pta.simulate(trace)
        self.assertAlmostEqual(result.energy, expected_energy * 1e-12, places=12)
        self.assertAlmostEqual(result.duration, expected_duration * 1e-6, places=6)
        self.assertEqual(result.end_state.name, "IDLE")
        self.assertEqual(result.parameters, {"txpower": 10, "length": None})

    def test_simulation_arg_function(self):
        pta = PTA(parameters=["txpower", "length"])
        pta.add_state("IDLE", power=5)
        pta.add_state("TX", power=100)
        pta.add_transition(
            "UNINITIALIZED", "IDLE", "init", energy=500000, duration=50000
        )
        pta.add_transition(
            "IDLE",
            "IDLE",
            "setTxPower",
            energy=10000,
            duration=120,
            param_update_function=lambda param, arg: {**param, "txpower": arg[0]},
        )
        pta.add_transition(
            "IDLE",
            "TX",
            "send",
            energy=AnalyticFunction(
                3,
                "regression_arg(0) + regression_arg(1) * function_arg(1)",
                ["txpower", "length"],
                regression_args=[3.0, 5],
                num_args=2,
            ),
            duration=AnalyticFunction(
                10,
                "regression_arg(0) + regression_arg(1) * function_arg(1)",
                ["txpower", "length"],
                regression_args=[48, 8],
                num_args=2,
            ),
        )
        pta.add_transition("TX", "IDLE", "txComplete", timeout=2000, is_interrupt=True)
        trace = [["init"], ["setTxPower", 10], ["send", "foo", 3]]
        expected_energy = 500000 + 10000 + (3 + 5 * 3) + (2000 * 100)
        expected_duration = 50000 + 120 + (48 + 8 * 3) + 2000
        result = pta.simulate(trace)
        self.assertAlmostEqual(result.energy, expected_energy * 1e-12, places=12)
        self.assertAlmostEqual(result.duration, expected_duration * 1e-6, places=6)
        self.assertEqual(result.end_state.name, "IDLE")
        self.assertEqual(result.parameters, {"txpower": 10, "length": None})

        pta = PTA(parameters=["txpower", "length"])
        pta.add_state("IDLE", power=5)
        pta.add_state("TX", power=100)
        pta.add_transition(
            "UNINITIALIZED", "IDLE", "init", energy=500000, duration=50000
        )
        pta.add_transition(
            "IDLE",
            "IDLE",
            "setTxPower",
            energy=10000,
            duration=120,
            param_update_function=lambda param, arg: {**param, "txpower": arg[0]},
        )
        pta.add_transition(
            "IDLE",
            "TX",
            "send",
            energy=AnalyticFunction(
                3,
                "regression_arg(0) + regression_arg(1) * function_arg(1)",
                ["txpower", "length"],
                regression_args=[3, 5],
                num_args=2,
            ),
            duration=AnalyticFunction(
                10,
                "regression_arg(0) + regression_arg(1) * function_arg(1)",
                ["txpower", "length"],
                regression_args=[48, 8],
                num_args=2,
            ),
        )
        pta.add_transition("TX", "IDLE", "txComplete", timeout=2000, is_interrupt=True)
        trace = [["init"], ["setTxPower", 10], ["send", "foobar", 6]]
        expected_energy = 500000 + 10000 + (3 + 5 * 6) + (2000 * 100)
        expected_duration = 50000 + 120 + (48 + 8 * 6) + 2000
        result = pta.simulate(trace)
        self.assertAlmostEqual(result.energy, expected_energy * 1e-12, places=12)
        self.assertAlmostEqual(result.duration, expected_duration * 1e-6, places=6)
        self.assertEqual(result.end_state.name, "IDLE")
        self.assertEqual(result.parameters, {"txpower": 10, "length": None})

    def test_simulation_param_function(self):
        pta = PTA(parameters=["length", "txpower"])
        pta.add_state("IDLE", power=5)
        pta.add_state(
            "TX",
            power=AnalyticFunction(
                100,
                "regression_arg(0) + regression_arg(1) * parameter(txpower)",
                ["length", "txpower"],
                regression_args=[1000, 2],
            ),
        )
        pta.add_transition(
            "UNINITIALIZED", "IDLE", "init", energy=500000, duration=50000
        )
        pta.add_transition(
            "IDLE",
            "IDLE",
            "setTxPower",
            energy=10000,
            duration=120,
            param_update_function=lambda param, arg: {**param, "txpower": arg[0]},
        )
        pta.add_transition(
            "IDLE",
            "TX",
            "send",
            energy=AnalyticFunction(
                3,
                "regression_arg(0) + regression_arg(1) * function_arg(1)",
                ["length", "txpower"],
                regression_args=[3, 5],
                num_args=2,
            ),
            duration=10,
            param_update_function=lambda param, arg: {**param, "length": arg[1]},
        )
        pta.add_transition(
            "TX",
            "IDLE",
            "txComplete",
            is_interrupt=True,
            timeout=AnalyticFunction(
                2000,
                "regression_arg(0) + regression_arg(1) * parameter(length)",
                ["length", "txpower"],
                regression_args=[500, 16],
            ),
        )
        trace = [["init"], ["setTxPower", 10], ["send", "foo", 3]]
        expected_energy = (
            500000 + 10000 + (3 + 5 * 3) + (1000 + 2 * 10) * (500 + 16 * 3)
        )
        expected_duration = 50000 + 120 + 10 + (500 + 16 * 3)
        result = pta.simulate(trace)
        self.assertAlmostEqual(result.energy, expected_energy * 1e-12, places=12)
        self.assertAlmostEqual(result.duration, expected_duration * 1e-6, places=6)
        self.assertEqual(result.end_state.name, "IDLE")
        self.assertEqual(result.parameters, {"txpower": 10, "length": 3})

    def test_get_X_expensive_state(self):
        pta = PTA.from_json(example_json_1)
        self.assertEqual(pta.get_least_expensive_state(), pta.state["IDLE"])
        self.assertEqual(pta.get_most_expensive_state(), pta.state["TX"])
        # self.assertAlmostEqual(pta.min_duration_until_energy_overflow(), (2**32 - 1) * 1e-12 / 10e-3, places=9)
        # self.assertAlmostEqual(pta.min_duration_until_energy_overflow(energy_granularity=1e-9), (2**32 - 1) * 1e-9 / 10e-3, places=9)
        self.assertAlmostEqual(
            pta.max_duration_until_energy_overflow(),
            (2 ** 32 - 1) * 1e-12 / 5e-6,
            places=9,
        )
        self.assertAlmostEqual(
            pta.max_duration_until_energy_overflow(energy_granularity=1e-9),
            (2 ** 32 - 1) * 1e-9 / 5e-6,
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
