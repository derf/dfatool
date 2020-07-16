#!/usr/bin/env python3

from dfatool.automata import PTA
from dfatool.codegen import get_simulated_accountingmethod
import unittest

example_json_1 = {
    "parameters": ["datarate", "txbytes", "txpower"],
    "initial_param_values": [None, None, None],
    "state": {
        "IDLE": {"power": {"static": 5,}},
        "TX": {
            "power": {
                "static": 100,
                "function": {
                    "raw": "regression_arg(0) + regression_arg(1)"
                    " * parameter(txpower)",
                    "regression_args": [100, 2],
                },
            }
        },
    },
    "transitions": [
        {
            "name": "init",
            "origin": ["UNINITIALIZED", "IDLE"],
            "destination": "IDLE",
            "duration": {"static": 50000,},
            "set_param": {"txpower": 10},
        },
        {
            "name": "setTxPower",
            "origin": "IDLE",
            "destination": "IDLE",
            "duration": {"static": 120},
            "energy ": {"static": 10000},
            "arg_to_param_map": {0: "txpower"},
            "argument_values": [[10, 20, 30]],
        },
        {
            "name": "send",
            "origin": "IDLE",
            "destination": "TX",
            "duration": {
                "static": 10,
                "function": {
                    "raw": "regression_arg(0) + regression_arg(1)" " * function_arg(1)",
                    "regression_args": [48, 8],
                },
            },
            "energy": {
                "static": 3,
                "function": {
                    "raw": "regression_arg(0) + regression_arg(1)" " * function_arg(1)",
                    "regression_args": [3, 5],
                },
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
                "static": 2000,
                "function": {
                    "raw": "regression_arg(0) + regression_arg(1)"
                    " * parameter(txbytes)",
                    "regression_args": [500, 16],
                },
            },
        },
    ],
}


class TestCG(unittest.TestCase):
    def test_statetransition_immediate(self):
        pta = PTA.from_json(example_json_1)
        pta.set_random_energy_model()
        pta.state["IDLE"].power.value = 9
        cg = get_simulated_accountingmethod("static_statetransition_immediate")(
            pta, 1000000, "uint8_t", "uint8_t", "uint8_t", "uint8_t"
        )
        cg.current_state = pta.state["IDLE"]
        cg.sleep(7)
        self.assertEqual(cg.get_energy(), 9 * 7)
        pta.transitions[1].energy.value = 123
        cg.pass_transition(pta.transitions[1])
        self.assertEqual(cg.get_energy(), 9 * 7 + 123)
        cg.pass_transition(pta.transitions[1])
        self.assertEqual(cg.get_energy(), (9 * 7 + 123 + 123) % 256)

        cg = get_simulated_accountingmethod("static_statetransition_immediate")(
            pta, 100000, "uint8_t", "uint8_t", "uint8_t", "uint8_t"
        )
        cg.current_state = pta.state["IDLE"]
        cg.sleep(7)
        self.assertEqual(cg.get_energy(), 0)
        cg.sleep(15)
        self.assertEqual(cg.get_energy(), 90)
        cg.sleep(90)
        self.assertEqual(cg.get_energy(), 900 % 256)

        cg = get_simulated_accountingmethod("static_statetransition_immediate")(
            pta, 100000, "uint8_t", "uint8_t", "uint8_t", "uint16_t"
        )
        cg.current_state = pta.state["IDLE"]
        cg.sleep(7)
        self.assertEqual(cg.get_energy(), 0)
        cg.sleep(15)
        self.assertEqual(cg.get_energy(), 90)
        cg.sleep(90)
        self.assertEqual(cg.get_energy(), 900)

        pta.state["IDLE"].power.value = 9  # -> 90 uW
        pta.transitions[1].energy.value = 1  # -> 100 pJ
        cg = get_simulated_accountingmethod("static_statetransition_immediate")(
            pta, 1000000, "uint8_t", "uint8_t", "uint8_t", "uint8_t", 1e-5, 1e-5, 1e-10
        )
        cg.current_state = pta.state["IDLE"]
        cg.sleep(10)  # 10 us
        self.assertEqual(cg.get_energy(), 90 * 10)
        cg.pass_transition(pta.transitions[1])
        self.assertAlmostEqual(cg.get_energy(), 90 * 10 + 100, places=0)
        cg.pass_transition(pta.transitions[1])
        self.assertAlmostEqual(cg.get_energy(), 90 * 10 + 100 + 100, places=0)

    def test_statetransition(self):
        pta = PTA.from_json(example_json_1)
        pta.set_random_energy_model()
        pta.state["IDLE"].power.value = 9
        cg = get_simulated_accountingmethod("static_statetransition")(
            pta, 1000000, "uint8_t", "uint8_t", "uint8_t", "uint8_t"
        )
        cg.current_state = pta.state["IDLE"]
        cg.sleep(7)
        self.assertEqual(cg.get_energy(), 9 * 7)
        pta.transitions[1].energy.value = 123
        cg.pass_transition(pta.transitions[1])
        self.assertEqual(cg.get_energy(), 9 * 7 + 123)
        cg.pass_transition(pta.transitions[1])
        self.assertEqual(cg.get_energy(), (9 * 7 + 123 + 123) % 256)

    def test_state_immediate(self):
        pta = PTA.from_json(example_json_1)
        pta.set_random_energy_model()
        pta.state["IDLE"].power.value = 9
        cg = get_simulated_accountingmethod("static_state_immediate")(
            pta, 1000000, "uint8_t", "uint8_t", "uint8_t", "uint8_t"
        )
        cg.current_state = pta.state["IDLE"]
        cg.sleep(7)
        self.assertEqual(cg.get_energy(), 9 * 7)
        pta.transitions[1].energy.value = 123
        cg.pass_transition(pta.transitions[1])
        self.assertEqual(cg.get_energy(), 9 * 7)
        cg.pass_transition(pta.transitions[1])
        self.assertEqual(cg.get_energy(), 9 * 7)

    def test_state(self):
        pta = PTA.from_json(example_json_1)
        pta.set_random_energy_model()
        pta.state["IDLE"].power.value = 9
        cg = get_simulated_accountingmethod("static_state")(
            pta, 1000000, "uint8_t", "uint8_t", "uint8_t", "uint8_t"
        )
        cg.current_state = pta.state["IDLE"]
        cg.sleep(7)
        self.assertEqual(cg.get_energy(), 9 * 7)
        pta.transitions[1].energy.value = 123
        cg.pass_transition(pta.transitions[1])
        self.assertEqual(cg.get_energy(), 9 * 7)
        cg.pass_transition(pta.transitions[1])
        self.assertEqual(cg.get_energy(), 9 * 7)

        cg = get_simulated_accountingmethod("static_state")(
            pta, 1000000, "uint8_t", "uint16_t", "uint16_t", "uint16_t"
        )


if __name__ == "__main__":
    unittest.main()
