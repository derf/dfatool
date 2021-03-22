#!/usr/bin/env python3

from dfatool.functions import StaticFunction
from dfatool.loader import RawData, pta_trace_to_aggregate
from dfatool.model import PTAModel
from dfatool.utils import by_name_to_by_param
from dfatool.validation import CrossValidator
import os
import unittest
import pytest

import numpy as np


class TestSynthetic(unittest.TestCase):
    def test_model_validation(self):
        # rng = np.random.default_rng(seed=1312) # requiresy NumPy >= 1.17
        np.random.seed(1312)
        X = np.arange(500) % 50
        parameter_names = ["p_mod5", "p_linear"]

        s1_duration_base = 70
        s1_duration_scale = 2
        s1_power_base = 50
        s1_power_scale = 7
        s2_duration_base = 700
        s2_duration_scale = 1
        s2_power_base = 1500
        s2_power_scale = 10

        by_name = {
            "raw_state_1": {
                "isa": "state",
                "param": [(x % 5, x) for x in X],
                "duration": s1_duration_base
                + np.random.normal(size=X.size, scale=s1_duration_scale),
                "power": s1_power_base
                + X
                + np.random.normal(size=X.size, scale=s1_power_scale),
                "attributes": ["duration", "power"],
            },
            "raw_state_2": {
                "isa": "state",
                "param": [(x % 5, x) for x in X],
                "duration": s2_duration_base
                - 2 * X
                + np.random.normal(size=X.size, scale=s2_duration_scale),
                "power": s2_power_base
                + X
                + np.random.normal(size=X.size, scale=s2_power_scale),
                "attributes": ["duration", "power"],
            },
        }
        by_param = by_name_to_by_param(by_name)
        model = PTAModel(by_name, parameter_names, dict())
        static_model = model.get_static()

        # x ∈ [0, 50] -> mean(X) is 25
        self.assertAlmostEqual(
            static_model("raw_state_1", "duration"), s1_duration_base, places=0
        )
        self.assertAlmostEqual(
            static_model("raw_state_1", "power"), s1_power_base + 25, delta=7
        )
        self.assertAlmostEqual(
            static_model("raw_state_2", "duration"), s2_duration_base - 2 * 25, delta=2
        )
        self.assertAlmostEqual(
            static_model("raw_state_2", "power"), s2_power_base + 25, delta=7
        )

        param_model, param_info = model.get_fitted()

        self.assertAlmostEqual(
            param_model("raw_state_1", "duration", param=[0, 10]),
            s1_duration_base,
            places=0,
        )
        self.assertAlmostEqual(
            param_model("raw_state_1", "duration", param=[0, 50]),
            s1_duration_base,
            places=0,
        )
        self.assertAlmostEqual(
            param_model("raw_state_1", "duration", param=[0, 70]),
            s1_duration_base,
            places=0,
        )

        self.assertAlmostEqual(
            param_model("raw_state_1", "power", param=[0, 10]),
            s1_power_base + 10,
            places=0,
        )
        self.assertAlmostEqual(
            param_model("raw_state_1", "power", param=[0, 50]),
            s1_power_base + 50,
            places=0,
        )
        self.assertAlmostEqual(
            param_model("raw_state_1", "power", param=[0, 70]),
            s1_power_base + 70,
            places=0,
        )

        self.assertAlmostEqual(
            param_model("raw_state_2", "duration", param=[0, 10]),
            s2_duration_base - 2 * 10,
            places=0,
        )
        self.assertAlmostEqual(
            param_model("raw_state_2", "duration", param=[0, 50]),
            s2_duration_base - 2 * 50,
            places=0,
        )
        self.assertAlmostEqual(
            param_model("raw_state_2", "duration", param=[0, 70]),
            s2_duration_base - 2 * 70,
            places=0,
        )

        self.assertAlmostEqual(
            param_model("raw_state_2", "power", param=[0, 10]),
            s2_power_base + 10,
            delta=50,
        )
        self.assertAlmostEqual(
            param_model("raw_state_2", "power", param=[0, 50]),
            s2_power_base + 50,
            delta=50,
        )
        self.assertAlmostEqual(
            param_model("raw_state_2", "power", param=[0, 70]),
            s2_power_base + 70,
            delta=50,
        )

        static_quality = model.assess(static_model)
        param_quality = model.assess(param_model)

        # static quality reflects normal distribution scale for non-parameterized data

        # the Root Mean Square Deviation must not be greater the scale (i.e., standard deviation) of the normal distribution
        # Low Mean Absolute Error (< 2)
        self.assertTrue(static_quality["raw_state_1"]["duration"]["mae"] < 2)
        # Low Root Mean Square Deviation (< scale == 2)
        self.assertTrue(static_quality["raw_state_1"]["duration"]["rmsd"] < 2)
        # Relatively low error percentage (~~ MAE * 100% / s1_duration_base)
        self.assertAlmostEqual(
            static_quality["raw_state_1"]["duration"]["mape"],
            static_quality["raw_state_1"]["duration"]["mae"] * 100 / s1_duration_base,
            places=1,
        )
        self.assertAlmostEqual(
            static_quality["raw_state_1"]["duration"]["smape"],
            static_quality["raw_state_1"]["duration"]["mae"] * 100 / s1_duration_base,
            places=1,
        )

        # static error is high for parameterized data

        # MAE == mean(abs(actual value - model value))
        # parameter range is [0, 50) -> mean 25, deviation range is [0, 25) -> mean deviation is 12.5 ± gauss scale
        self.assertAlmostEqual(
            static_quality["raw_state_1"]["power"]["mae"], 12.5, delta=1
        )
        self.assertAlmostEqual(
            static_quality["raw_state_1"]["power"]["rmsd"], 16, delta=2
        )
        # high percentage error due to low s1_power_base
        self.assertAlmostEqual(
            static_quality["raw_state_1"]["power"]["mape"], 19, delta=2
        )
        self.assertAlmostEqual(
            static_quality["raw_state_1"]["power"]["smape"], 19, delta=2
        )

        # parameter range is [0, 100) -> mean deviation is 25 ± gauss scale
        self.assertAlmostEqual(
            static_quality["raw_state_2"]["duration"]["mae"], 25, delta=2
        )
        self.assertAlmostEqual(
            static_quality["raw_state_2"]["duration"]["rmsd"], 30, delta=2
        )

        # low percentage error due to high s2_duration_base (~~ 3.5 %)
        self.assertAlmostEqual(
            static_quality["raw_state_2"]["duration"]["mape"],
            25 * 100 / s2_duration_base,
            delta=1,
        )
        self.assertAlmostEqual(
            static_quality["raw_state_2"]["duration"]["smape"],
            25 * 100 / s2_duration_base,
            delta=1,
        )

        self.assertAlmostEqual(
            static_quality["raw_state_2"]["power"]["mae"], 12.5, delta=2
        )
        self.assertAlmostEqual(
            static_quality["raw_state_2"]["power"]["rmsd"], 17, delta=2
        )

        # low percentage error due to high s2_power_base (~~ 1.7 %)
        self.assertAlmostEqual(
            static_quality["raw_state_2"]["power"]["mape"],
            25 * 100 / s2_power_base,
            delta=1,
        )
        self.assertAlmostEqual(
            static_quality["raw_state_2"]["power"]["smape"],
            25 * 100 / s2_power_base,
            delta=1,
        )

        # raw_state_1/duration does not depend on parameters and delegates to the static model
        self.assertAlmostEqual(
            param_quality["raw_state_1"]["duration"]["mae"],
            static_quality["raw_state_1"]["duration"]["mae"],
        )
        self.assertAlmostEqual(
            param_quality["raw_state_1"]["duration"]["rmsd"],
            static_quality["raw_state_1"]["duration"]["rmsd"],
        )
        self.assertAlmostEqual(
            param_quality["raw_state_1"]["duration"]["mape"],
            static_quality["raw_state_1"]["duration"]["mape"],
        )
        self.assertAlmostEqual(
            param_quality["raw_state_1"]["duration"]["smape"],
            static_quality["raw_state_1"]["duration"]["smape"],
        )

        # fitted param-model quality reflects normal distribution scale for all data
        self.assertAlmostEqual(
            param_quality["raw_state_2"]["power"]["mape"], 0.9, places=1
        )
        self.assertAlmostEqual(
            param_quality["raw_state_2"]["power"]["smape"], 0.9, places=1
        )

        self.assertTrue(param_quality["raw_state_1"]["power"]["mae"] < s1_power_scale)
        self.assertTrue(param_quality["raw_state_1"]["power"]["rmsd"] < s1_power_scale)
        self.assertAlmostEqual(
            param_quality["raw_state_1"]["power"]["mape"], 7.5, delta=1
        )
        self.assertAlmostEqual(
            param_quality["raw_state_1"]["power"]["smape"], 7.5, delta=1
        )

        self.assertAlmostEqual(
            param_quality["raw_state_2"]["duration"]["mae"],
            s2_duration_scale,
            delta=0.2,
        )
        self.assertAlmostEqual(
            param_quality["raw_state_2"]["duration"]["rmsd"],
            s2_duration_scale,
            delta=0.2,
        )
        self.assertAlmostEqual(
            param_quality["raw_state_2"]["duration"]["mape"], 0.12, delta=0.01
        )
        self.assertAlmostEqual(
            param_quality["raw_state_2"]["duration"]["smape"], 0.12, delta=0.01
        )

        # ... unless the signal-to-noise ratio (parameter range = [0 .. 50] vs. scale = 10) is bad, leading to
        # increased regression errors
        self.assertTrue(param_quality["raw_state_2"]["power"]["mae"] < 15)
        self.assertTrue(param_quality["raw_state_2"]["power"]["rmsd"] < 18)

        # still: low percentage error due to high s2_power_base
        self.assertAlmostEqual(
            param_quality["raw_state_2"]["power"]["mape"], 0.9, places=1
        )
        self.assertAlmostEqual(
            param_quality["raw_state_2"]["power"]["smape"], 0.9, places=1
        )

    def test_model_crossvalidation_10fold(self):
        # rng = np.random.default_rng(seed=1312) # requiresy NumPy >= 1.17
        np.random.seed(1312)
        X = np.arange(500) % 50
        parameter_names = ["p_mod5", "p_linear"]

        s1_duration_base = 70
        s1_duration_scale = 2
        s1_power_base = 50
        s1_power_scale = 7
        s2_duration_base = 700
        s2_duration_scale = 1
        s2_power_base = 1500
        s2_power_scale = 10

        by_name = {
            "raw_state_1": {
                "isa": "state",
                "param": [(x % 5, x) for x in X],
                "duration": s1_duration_base
                + np.random.normal(size=X.size, scale=s1_duration_scale),
                "power": s1_power_base
                + X
                + np.random.normal(size=X.size, scale=s1_power_scale),
                "attributes": ["duration", "power"],
            },
            "raw_state_2": {
                "isa": "state",
                "param": [(x % 5, x) for x in X],
                "duration": s2_duration_base
                - 2 * X
                + np.random.normal(size=X.size, scale=s2_duration_scale),
                "power": s2_power_base
                + X
                + np.random.normal(size=X.size, scale=s2_power_scale),
                "attributes": ["duration", "power"],
            },
        }
        by_param = by_name_to_by_param(by_name)
        arg_count = dict()
        model = PTAModel(by_name, parameter_names, arg_count)
        validator = CrossValidator(PTAModel, by_name, parameter_names, arg_count)

        static_quality = validator.kfold(lambda m: m.get_static(), 10)
        param_quality = validator.kfold(lambda m: m.get_fitted()[0], 10)

        # static quality reflects normal distribution scale for non-parameterized data

        # the Root Mean Square Deviation must not be greater the scale (i.e., standard deviation) of the normal distribution
        # Low Mean Absolute Error (< 2)
        self.assertTrue(static_quality["raw_state_1"]["duration"]["mae"] < 2)
        # Low Root Mean Square Deviation (< scale == 2)
        self.assertTrue(static_quality["raw_state_1"]["duration"]["rmsd"] < 2)
        # Relatively low error percentage (~~ MAE * 100% / s1_duration_base)
        self.assertAlmostEqual(
            static_quality["raw_state_1"]["duration"]["smape"],
            static_quality["raw_state_1"]["duration"]["mae"] * 100 / s1_duration_base,
            places=1,
        )

        # static error is high for parameterized data

        # MAE == mean(abs(actual value - model value))
        # parameter range is [0, 50) -> mean 25, deviation range is [0, 25) -> mean deviation is 12.5 ± gauss scale
        self.assertAlmostEqual(
            static_quality["raw_state_1"]["power"]["mae"], 12.5, delta=1
        )
        self.assertAlmostEqual(
            static_quality["raw_state_1"]["power"]["rmsd"], 16, delta=2
        )
        # high percentage error due to low s1_power_base
        self.assertAlmostEqual(
            static_quality["raw_state_1"]["power"]["smape"], 19, delta=2
        )

        # parameter range is [0, 100) -> mean deviation is 25 ± gauss scale
        self.assertAlmostEqual(
            static_quality["raw_state_2"]["duration"]["mae"], 25, delta=2
        )
        self.assertAlmostEqual(
            static_quality["raw_state_2"]["duration"]["rmsd"], 30, delta=2
        )

        # low percentage error due to high s2_duration_base (~~ 3.5 %)
        self.assertAlmostEqual(
            static_quality["raw_state_2"]["duration"]["smape"],
            25 * 100 / s2_duration_base,
            delta=1,
        )

        self.assertAlmostEqual(
            static_quality["raw_state_2"]["power"]["mae"], 12.5, delta=2
        )
        self.assertAlmostEqual(
            static_quality["raw_state_2"]["power"]["rmsd"], 17, delta=2
        )

        # low percentage error due to high s2_power_base (~~ 1.7 %)
        self.assertAlmostEqual(
            static_quality["raw_state_2"]["power"]["smape"],
            25 * 100 / s2_power_base,
            delta=1,
        )

        # raw_state_1/duration does not depend on parameters and delegates to the static model
        self.assertAlmostEqual(
            param_quality["raw_state_1"]["duration"]["mae"],
            static_quality["raw_state_1"]["duration"]["mae"],
        )
        self.assertAlmostEqual(
            param_quality["raw_state_1"]["duration"]["rmsd"],
            static_quality["raw_state_1"]["duration"]["rmsd"],
        )
        self.assertAlmostEqual(
            param_quality["raw_state_1"]["duration"]["smape"],
            static_quality["raw_state_1"]["duration"]["smape"],
        )

        # fitted param-model quality reflects normal distribution scale for all data
        self.assertAlmostEqual(
            param_quality["raw_state_2"]["power"]["smape"], 0.9, places=1
        )

        self.assertTrue(param_quality["raw_state_1"]["power"]["mae"] < s1_power_scale)
        self.assertTrue(param_quality["raw_state_1"]["power"]["rmsd"] < s1_power_scale)
        self.assertAlmostEqual(
            param_quality["raw_state_1"]["power"]["smape"], 7.5, delta=1
        )

        self.assertAlmostEqual(
            param_quality["raw_state_2"]["duration"]["mae"],
            s2_duration_scale,
            delta=0.2,
        )
        self.assertAlmostEqual(
            param_quality["raw_state_2"]["duration"]["rmsd"],
            s2_duration_scale,
            delta=0.2,
        )
        self.assertAlmostEqual(
            param_quality["raw_state_2"]["duration"]["smape"], 0.12, delta=0.01
        )

        # ... unless the signal-to-noise ratio (parameter range = [0 .. 50] vs. scale = 10) is bad, leading to
        # increased regression errors
        self.assertTrue(param_quality["raw_state_2"]["power"]["mae"] < 15)
        self.assertTrue(param_quality["raw_state_2"]["power"]["rmsd"] < 18)

        # still: low percentage error due to high s2_power_base
        self.assertAlmostEqual(
            param_quality["raw_state_2"]["power"]["smape"], 0.9, places=1
        )


class TestFromFile(unittest.TestCase):
    def test_singlefile_rf24(self):
        raw_data = RawData(["test-data/20170220_164723_RF24_int_A.tar"])
        preprocessed_data = raw_data.get_preprocessed_data()
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = PTAModel(by_name, parameters, arg_count)
        self.assertEqual(model.states, "POWERDOWN RX STANDBY1 TX".split(" "))
        self.assertEqual(
            model.transitions,
            "begin epilogue powerDown powerUp setDataRate_num setPALevel_num startListening stopListening write_nb".split(
                " "
            ),
        )
        static_model = model.get_static()
        self.assertAlmostEqual(static_model("POWERDOWN", "power"), 0, places=0)
        self.assertAlmostEqual(static_model("RX", "power"), 52254, places=0)
        self.assertAlmostEqual(static_model("STANDBY1", "power"), 7, places=0)
        self.assertAlmostEqual(static_model("TX", "power"), 18414, places=0)
        self.assertAlmostEqual(static_model("begin", "power"), 84, places=0)
        self.assertAlmostEqual(static_model("epilogue", "power"), 381, places=0)
        self.assertAlmostEqual(static_model("powerDown", "power"), 51, places=0)
        self.assertAlmostEqual(static_model("powerUp", "power"), 164, places=0)
        self.assertAlmostEqual(static_model("setDataRate_num", "power"), 58, places=0)
        self.assertAlmostEqual(static_model("setPALevel_num", "power"), 52, places=0)
        self.assertAlmostEqual(static_model("startListening", "power"), 16354, places=0)
        self.assertAlmostEqual(static_model("stopListening", "power"), 691, places=0)
        self.assertAlmostEqual(static_model("write_nb", "power"), 428, places=0)
        self.assertAlmostEqual(static_model("begin", "rel_power_prev"), 84, places=0)
        self.assertAlmostEqual(
            static_model("epilogue", "rel_power_prev"), -18025, places=0
        )
        self.assertAlmostEqual(
            static_model("powerDown", "rel_power_prev"), 43, places=0
        )
        self.assertAlmostEqual(static_model("powerUp", "rel_power_prev"), 163, places=0)
        self.assertAlmostEqual(
            static_model("setDataRate_num", "rel_power_prev"), 51, places=0
        )
        self.assertAlmostEqual(
            static_model("setPALevel_num", "rel_power_prev"), 41, places=0
        )
        self.assertAlmostEqual(
            static_model("startListening", "rel_power_prev"), 16348, places=0
        )
        self.assertAlmostEqual(
            static_model("stopListening", "rel_power_prev"), -51567, places=0
        )
        self.assertAlmostEqual(
            static_model("write_nb", "rel_power_prev"), 421, places=0
        )
        """
        self.assertAlmostEqual(static_model("begin", "energy"), 1652249, places=0)
        self.assertAlmostEqual(static_model("epilogue", "energy"), 15449, places=0)
        self.assertAlmostEqual(static_model("powerDown", "energy"), 4547, places=0)
        self.assertAlmostEqual(static_model("powerUp", "energy"), 1641765, places=0)
        self.assertAlmostEqual(
            static_model("setDataRate_num", "energy"), 7749, places=0
        )
        self.assertAlmostEqual(static_model("setPALevel_num", "energy"), 4700, places=0)
        self.assertAlmostEqual(
            static_model("startListening", "energy"), 4309602, places=0
        )
        self.assertAlmostEqual(
            static_model("stopListening", "energy"), 193775, places=0
        )
        self.assertAlmostEqual(static_model("write_nb", "energy"), 218339, places=0)
        self.assertAlmostEqual(
            static_model("begin", "rel_energy_prev"), 1649571, places=0
        )
        self.assertAlmostEqual(
            static_model("epilogue", "rel_energy_prev"), -744114, places=0
        )
        self.assertAlmostEqual(
            static_model("powerDown", "rel_energy_prev"), 3854, places=0
        )
        self.assertAlmostEqual(
            static_model("powerUp", "rel_energy_prev"), 1641381, places=0
        )
        self.assertAlmostEqual(
            static_model("setDataRate_num", "rel_energy_prev"), 6777, places=0
        )
        self.assertAlmostEqual(
            static_model("setPALevel_num", "rel_energy_prev"), 3728, places=0
        )
        self.assertAlmostEqual(
            static_model("startListening", "rel_energy_prev"), 4307769, places=0
        )
        self.assertAlmostEqual(
            static_model("stopListening", "rel_energy_prev"), -13533693, places=0
        )
        self.assertAlmostEqual(
            static_model("write_nb", "rel_energy_prev"), 214618, places=0
        )
        """
        self.assertAlmostEqual(static_model("begin", "duration"), 19830, places=0)
        self.assertAlmostEqual(static_model("epilogue", "duration"), 40, places=0)
        self.assertAlmostEqual(static_model("powerDown", "duration"), 90, places=0)
        self.assertAlmostEqual(static_model("powerUp", "duration"), 10030, places=0)
        self.assertAlmostEqual(
            static_model("setDataRate_num", "duration"), 140, places=0
        )
        self.assertAlmostEqual(static_model("setPALevel_num", "duration"), 90, places=0)
        self.assertAlmostEqual(
            static_model("startListening", "duration"), 260, places=0
        )
        self.assertAlmostEqual(static_model("stopListening", "duration"), 260, places=0)
        self.assertAlmostEqual(static_model("write_nb", "duration"), 510, places=0)

        self.assertAlmostEqual(
            model.attr_by_name["POWERDOWN"]["power"].stats.param_dependence_ratio(
                "datarate"
            ),
            0,
            places=2,
        )
        self.assertAlmostEqual(
            model.attr_by_name["POWERDOWN"]["power"].stats.param_dependence_ratio(
                "txbytes"
            ),
            0,
            places=2,
        )
        self.assertAlmostEqual(
            model.attr_by_name["POWERDOWN"]["power"].stats.param_dependence_ratio(
                "txpower"
            ),
            0,
            places=2,
        )
        self.assertAlmostEqual(
            model.attr_by_name["RX"]["power"].stats.param_dependence_ratio("datarate"),
            0.99,
            places=2,
        )
        self.assertAlmostEqual(
            model.attr_by_name["RX"]["power"].stats.param_dependence_ratio("txbytes"),
            0,
            places=2,
        )
        self.assertAlmostEqual(
            model.attr_by_name["RX"]["power"].stats.param_dependence_ratio("txpower"),
            0.01,
            places=2,
        )
        self.assertAlmostEqual(
            model.attr_by_name["STANDBY1"]["power"].stats.param_dependence_ratio(
                "datarate"
            ),
            0.04,
            places=2,
        )
        self.assertAlmostEqual(
            model.attr_by_name["STANDBY1"]["power"].stats.param_dependence_ratio(
                "txbytes"
            ),
            0.35,
            places=2,
        )
        self.assertAlmostEqual(
            model.attr_by_name["STANDBY1"]["power"].stats.param_dependence_ratio(
                "txpower"
            ),
            0.32,
            places=2,
        )
        self.assertAlmostEqual(
            model.attr_by_name["TX"]["power"].stats.param_dependence_ratio("datarate"),
            1,
            places=2,
        )
        self.assertAlmostEqual(
            model.attr_by_name["TX"]["power"].stats.param_dependence_ratio("txbytes"),
            0.09,
            places=2,
        )
        self.assertAlmostEqual(
            model.attr_by_name["TX"]["power"].stats.param_dependence_ratio("txpower"),
            1,
            places=2,
        )

        param_model, param_info = model.get_fitted()
        self.assertIsInstance(param_info("POWERDOWN", "power"), StaticFunction)
        self.assertEqual(
            param_info("RX", "power").model_function,
            "0 + regression_arg(0) + regression_arg(1) * np.sqrt(parameter(datarate))",
        )
        self.assertAlmostEqual(
            param_info("RX", "power").model_args[0], 48530.7, places=0
        )
        self.assertAlmostEqual(param_info("RX", "power").model_args[1], 117, places=0)
        self.assertIsInstance(param_info("STANDBY1", "power"), StaticFunction)
        self.assertEqual(
            param_info("TX", "power").model_function,
            "0 + regression_arg(0) + regression_arg(1) * 1/(parameter(datarate)) + regression_arg(2) * parameter(txpower) + regression_arg(3) * 1/(parameter(datarate)) * parameter(txpower)",
        )
        self.assertEqual(
            param_info("epilogue", "timeout").model_function,
            "0 + regression_arg(0) + regression_arg(1) * 1/(parameter(datarate))",
        )
        self.assertEqual(
            param_info("stopListening", "duration").model_function,
            "0 + regression_arg(0) + regression_arg(1) * 1/(parameter(datarate))",
        )

        self.assertAlmostEqual(
            param_model("RX", "power", param=[1, None, None]), 48647, places=-1
        )

    def test_decisiontrees_rf24(self):
        raw_data = RawData(["test-data/20191024-152648-nrf24l01-var-ack.tar"])
        preprocessed_data = raw_data.get_preprocessed_data()
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = PTAModel(by_name, parameters, arg_count)
        self.assertEqual(model.states, "RX STANDBY1".split(" "))
        self.assertEqual(
            model.transitions,
            "setAutoAck setDataRate setPALevel setup startListening stopListening write".split(
                " "
            ),
        )
        static_model = model.get_static()
        self.assertAlmostEqual(static_model("RX", "power"), 47964, places=0)
        self.assertAlmostEqual(static_model("STANDBY1", "power"), 128, places=0)
        self.assertAlmostEqual(static_model("setAutoAck", "power"), 151, places=0)
        self.assertAlmostEqual(static_model("setDataRate", "power"), 146, places=0)
        self.assertAlmostEqual(static_model("setPALevel", "power"), 147, places=0)
        self.assertAlmostEqual(static_model("setup", "power"), 153, places=0)
        self.assertAlmostEqual(static_model("startListening", "power"), 18954, places=0)
        self.assertAlmostEqual(static_model("stopListening", "power"), 2426, places=0)
        self.assertAlmostEqual(static_model("write", "power"), 17629, places=0)

        self.assertAlmostEqual(static_model("setAutoAck", "duration"), 90, places=0)
        self.assertAlmostEqual(static_model("setDataRate", "duration"), 240, places=0)
        self.assertAlmostEqual(static_model("setPALevel", "duration"), 160, places=0)
        self.assertAlmostEqual(static_model("setup", "duration"), 6550, places=0)
        self.assertAlmostEqual(
            static_model("startListening", "duration"), 470, places=0
        )
        self.assertAlmostEqual(static_model("stopListening", "duration"), 510, places=0)
        self.assertAlmostEqual(static_model("write", "duration"), 11230, places=0)

        self.assertAlmostEqual(
            model.attr_by_name["write"]["duration"].stats.param_dependence_ratio(
                "auto_ack!"
            ),
            1,
            places=2,
        )
        self.assertAlmostEqual(
            model.attr_by_name["write"]["power"].stats.param_dependence_ratio(
                "auto_ack!"
            ),
            0.99,
            places=2,
        )

        param_model, param_info = model.get_fitted()

        self.assertAlmostEqual(
            param_model(
                "write",
                "duration",
                param=[0, 76, 1000, 0, 10, None, None, 1500, 0, None, 9, None, None],
            ),
            1133,
            places=0,
        )

        # only bitrate and packet length are relevant
        self.assertAlmostEqual(
            param_model(
                "write",
                "duration",
                param=[
                    0,
                    None,
                    1000,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            1133,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "duration",
                param=[
                    0,
                    None,
                    250,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            2100,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "duration",
                param=[
                    0,
                    None,
                    2000,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            972,
            places=0,
        )

        # auto_ack == 1 has a different write duration, still only bitrate and packet length are relevant
        self.assertAlmostEqual(
            param_model(
                "write",
                "duration",
                param=[1, 76, 1000, 0, 10, None, None, 1500, 0, None, 9, None, None],
            ),
            22327,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "duration",
                param=[
                    1,
                    None,
                    1000,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            22327,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "duration",
                param=[
                    1,
                    None,
                    250,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            33273,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "duration",
                param=[
                    1,
                    None,
                    2000,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            20503,
            places=0,
        )

    def test_decisiontrees_more_rf24(self):
        raw_data = RawData(["test-data/20191024-150723-nrf24l01-var-ack-retry.tar"])
        preprocessed_data = raw_data.get_preprocessed_data()
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = PTAModel(by_name, parameters, arg_count)
        self.assertEqual(model.states, "STANDBY1".split(" "))
        self.assertEqual(
            model.transitions, "setAutoAck setPALevel setRetries setup write".split(" ")
        )
        static_model = model.get_static()
        self.assertAlmostEqual(static_model("STANDBY1", "power"), 130, places=0)
        self.assertAlmostEqual(static_model("setAutoAck", "power"), 150, places=0)
        self.assertAlmostEqual(static_model("setPALevel", "power"), 150, places=0)
        self.assertAlmostEqual(static_model("setRetries", "power"), 150, places=0)
        self.assertAlmostEqual(static_model("setup", "power"), 156, places=0)
        self.assertAlmostEqual(static_model("write", "power"), 14247, places=0)

        self.assertAlmostEqual(static_model("setAutoAck", "duration"), 90, places=0)
        self.assertAlmostEqual(static_model("setPALevel", "duration"), 160, places=0)
        self.assertAlmostEqual(static_model("setRetries", "duration"), 90, places=0)
        self.assertAlmostEqual(static_model("setup", "duration"), 6550, places=0)
        self.assertAlmostEqual(static_model("write", "duration"), 1245, places=0)

        self.assertAlmostEqual(
            model.attr_by_name["write"]["duration"].stats.param_dependence_ratio(
                "auto_ack!"
            ),
            1,
            places=2,
        )
        self.assertAlmostEqual(
            model.attr_by_name["write"]["duration"].stats.param_dependence_ratio(
                "max_retry_count"
            ),
            1,
            places=2,
        )
        self.assertAlmostEqual(
            model.attr_by_name["write"]["duration"].stats.param_dependence_ratio(
                "retry_delay"
            ),
            1,
            places=2,
        )
        self.assertAlmostEqual(
            model.attr_by_name["write"]["duration"].stats.param_dependence_ratio(
                "tx_power"
            ),
            0.36,
            places=2,
        )
        self.assertAlmostEqual(
            model.attr_by_name["write"]["power"].stats.param_dependence_ratio(
                "auto_ack!"
            ),
            1,
            places=2,
        )
        self.assertAlmostEqual(
            model.attr_by_name["write"]["power"].stats.param_dependence_ratio(
                "max_retry_count"
            ),
            0.98,
            places=2,
        )
        self.assertAlmostEqual(
            model.attr_by_name["write"]["power"].stats.param_dependence_ratio(
                "retry_delay"
            ),
            1,
            places=2,
        )
        self.assertAlmostEqual(
            model.attr_by_name["write"]["power"].stats.param_dependence_ratio(
                "tx_power"
            ),
            1,
            places=2,
        )

        param_model, param_info = model.get_fitted()

        # auto_ack! == 0 -> max_retry_count and retry_delay have no effect on duraton

        self.assertAlmostEqual(
            param_model(
                "write",
                "duration",
                param=[0, None, 1000, None, 0, None, None, 250, 0, None, 9, None, None],
            ),
            1148,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "duration",
                param=[0, None, 1000, None, 5, None, None, 250, 0, None, 9, None, None],
            ),
            1148,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "duration",
                param=[
                    0,
                    None,
                    1000,
                    None,
                    5,
                    None,
                    None,
                    2750,
                    0,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            1148,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "duration",
                param=[
                    0,
                    None,
                    1000,
                    None,
                    5,
                    None,
                    None,
                    2750,
                    15,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            1148,
            places=0,
        )

        # auto_ack! == 1 -> max_retry_count and retry_delay affect duration, tx_power does not
        self.assertAlmostEqual(
            param_model(
                "write",
                "duration",
                param=[1, None, 1000, None, 0, None, None, 250, 0, None, 9, None, None],
            ),
            1473,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "duration",
                param=[
                    1,
                    None,
                    1000,
                    None,
                    0,
                    None,
                    None,
                    250,
                    12,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            1473,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "duration",
                param=[1, None, 1000, None, 5, None, None, 250, 0, None, 9, None, None],
            ),
            5030,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "duration",
                param=[
                    1,
                    None,
                    1000,
                    None,
                    5,
                    None,
                    None,
                    250,
                    16,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            5030,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "duration",
                param=[
                    1,
                    None,
                    1000,
                    None,
                    5,
                    None,
                    None,
                    2750,
                    0,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            20029,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "duration",
                param=[
                    1,
                    None,
                    1000,
                    None,
                    5,
                    None,
                    None,
                    2750,
                    15,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            20029,
            places=0,
        )

        # auto_ack! == 0 -> max_retry_count and retry_delay have no effect on power, txpower does

        self.assertAlmostEqual(
            param_model(
                "write",
                "power",
                param=[0, None, 1000, None, 0, None, None, 250, 0, None, 9, None, None],
            ),
            12420,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "power",
                param=[
                    0,
                    None,
                    1000,
                    None,
                    0,
                    None,
                    None,
                    250,
                    12,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            19172,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "power",
                param=[0, None, 1000, None, 5, None, None, 250, 0, None, 9, None, None],
            ),
            12420,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "power",
                param=[
                    0,
                    None,
                    1000,
                    None,
                    5,
                    None,
                    None,
                    250,
                    12,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            19172,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "power",
                param=[
                    0,
                    None,
                    1000,
                    None,
                    5,
                    None,
                    None,
                    2750,
                    0,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            12420,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "power",
                param=[
                    0,
                    None,
                    1000,
                    None,
                    5,
                    None,
                    None,
                    2750,
                    12,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            19172,
            places=0,
        )

        # auto_ack! == 1 -> max_retry_count and retry_delay also affect power
        self.assertAlmostEqual(
            param_model(
                "write",
                "power",
                param=[1, None, 1000, None, 0, None, None, 250, 0, None, 9, None, None],
            ),
            16692,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "power",
                param=[
                    1,
                    None,
                    1000,
                    None,
                    0,
                    None,
                    None,
                    250,
                    12,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            22317,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "power",
                param=[1, None, 1000, None, 5, None, None, 250, 0, None, 9, None, None],
            ),
            26361,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "power",
                param=[
                    1,
                    None,
                    1000,
                    None,
                    5,
                    None,
                    None,
                    250,
                    12,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            35292,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "power",
                param=[
                    1,
                    None,
                    1000,
                    None,
                    5,
                    None,
                    None,
                    2750,
                    0,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            7931,
            places=0,
        )
        self.assertAlmostEqual(
            param_model(
                "write",
                "power",
                param=[
                    1,
                    None,
                    1000,
                    None,
                    5,
                    None,
                    None,
                    2750,
                    12,
                    None,
                    9,
                    None,
                    None,
                ],
            ),
            10356,
            places=0,
        )

    def test_singlefile_mmparam(self):
        raw_data = RawData(["test-data/20161221_123347_mmparam.tar"])
        preprocessed_data = raw_data.get_preprocessed_data()
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = PTAModel(by_name, parameters, arg_count)
        self.assertEqual(model.states, "OFF ON".split(" "))
        self.assertEqual(model.transitions, "off setBrightness".split(" "))
        static_model = model.get_static()
        self.assertAlmostEqual(static_model("OFF", "power"), 7124, places=0)
        self.assertAlmostEqual(static_model("ON", "power"), 17866, places=0)
        self.assertAlmostEqual(static_model("off", "power"), 29362, places=0)
        self.assertAlmostEqual(static_model("setBrightness", "power"), 18500, places=0)
        self.assertAlmostEqual(static_model("off", "rel_power_prev"), 11496, places=0)
        self.assertAlmostEqual(
            static_model("setBrightness", "rel_power_prev"), 11368, places=0
        )
        """
        self.assertAlmostEqual(static_model("off", "energy"), 268079197, places=0)
        self.assertAlmostEqual(
            static_model("setBrightness", "energy"), 168912773, places=0
        )
        self.assertAlmostEqual(
            static_model("off", "rel_energy_prev"), 105040198, places=0
        )
        self.assertAlmostEqual(
            static_model("setBrightness", "rel_energy_prev"), 103745586, places=0
        )
        """
        self.assertAlmostEqual(static_model("off", "duration"), 9130, places=0)
        self.assertAlmostEqual(
            static_model("setBrightness", "duration"), 9130, places=0
        )

        param_lut_model = model.get_param_lut()
        self.assertAlmostEqual(
            param_lut_model("OFF", "power", param=[None, None]), 7124, places=0
        )
        with self.assertRaises(KeyError):
            param_lut_model("ON", "power", param=[None, None])
            param_lut_model("ON", "power", param=["a"])
            param_lut_model("ON", "power", param=[0])
        self.assertTrue(param_lut_model("ON", "power", param=[0, 0]))
        param_lut_model = model.get_param_lut(fallback=True)
        self.assertAlmostEqual(
            param_lut_model("ON", "power", param=[None, None]), 17866, places=0
        )

    def test_et_bar(self):
        raw_data = RawData(["test-data/20200722-113624-timedResistiveLoad.tar"])
        preprocessed_data = raw_data.get_preprocessed_data()
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = PTAModel(by_name, parameters, arg_count)
        self.assertEqual(model.states, "P14MW P235UW P3_4MW SLEEP".split(" "))
        self.assertEqual(
            model.transitions,
            "nop10K nop1K0 nop3K3 setup switchTo3K3 switchTo47K switchTo750 switchToNone".split(
                " "
            ),
        )
        static_model = model.get_static()
        self.assertAlmostEqual(static_model("P14MW", "power"), 14540, places=0)
        self.assertAlmostEqual(static_model("P235UW", "power"), 899, places=0)
        self.assertAlmostEqual(static_model("P3_4MW", "power"), 3974, places=0)
        self.assertAlmostEqual(static_model("SLEEP", "power"), 672, places=0)
        self.assertAlmostEqual(static_model("nop10K", "duration"), 514, places=0)
        self.assertAlmostEqual(static_model("nop10K", "power"), 939, places=0)
        self.assertAlmostEqual(static_model("nop1K0", "duration"), 514, places=0)
        self.assertAlmostEqual(static_model("nop1K0", "power"), 914, places=0)
        self.assertAlmostEqual(static_model("nop3K3", "duration"), 514, places=0)
        self.assertAlmostEqual(static_model("nop3K3", "power"), 1669, places=0)
        self.assertAlmostEqual(static_model("setup", "duration"), 27, places=0)
        self.assertAlmostEqual(static_model("setup", "power"), 726, places=0)
        self.assertAlmostEqual(static_model("switchTo3K3", "duration"), 19, places=0)
        self.assertAlmostEqual(static_model("switchTo3K3", "power"), 743, places=0)
        self.assertAlmostEqual(static_model("switchTo47K", "duration"), 19, places=0)
        self.assertAlmostEqual(static_model("switchTo47K", "power"), 733, places=0)
        self.assertAlmostEqual(static_model("switchTo750", "duration"), 19, places=0)
        self.assertAlmostEqual(static_model("switchTo750", "power"), 732, places=0)
        self.assertAlmostEqual(static_model("switchToNone", "duration"), 19, places=0)
        self.assertAlmostEqual(static_model("switchToNone", "power"), 741, places=0)

    def test_et_la_dco(self):
        raw_data = RawData(["test-data/20201203-112341-et_la_dco.tar"])
        preprocessed_data = raw_data.get_preprocessed_data()
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = PTAModel(by_name, parameters, arg_count)
        self.assertEqual(model.states, "IDLE".split(" "))
        self.assertEqual(
            model.transitions,
            "setup trans100u trans10m trans1m trans2m trans5m".split(" "),
        )
        static_model = model.get_static()
        self.assertAlmostEqual(static_model("IDLE", "power"), 766, places=0)
        self.assertAlmostEqual(static_model("setup", "duration"), 15, places=0)
        self.assertAlmostEqual(static_model("setup", "power"), 939, places=0)
        self.assertAlmostEqual(static_model("trans100u", "duration"), 146, places=0)
        self.assertAlmostEqual(static_model("trans100u", "power"), 939, places=0)
        self.assertAlmostEqual(static_model("trans10m", "duration"), 10084, places=0)
        self.assertAlmostEqual(static_model("trans10m", "power"), 6236, places=0)
        self.assertAlmostEqual(static_model("trans1m", "duration"), 1025, places=0)
        self.assertAlmostEqual(static_model("trans1m", "power"), 3878, places=0)
        self.assertAlmostEqual(static_model("trans2m", "duration"), 2031, places=0)
        self.assertAlmostEqual(static_model("trans2m", "power"), 5194, places=0)
        self.assertAlmostEqual(static_model("trans5m", "duration"), 5049, places=0)
        self.assertAlmostEqual(static_model("trans5m", "power"), 6013, places=0)

    def test_et_timer_dco(self):
        raw_data = RawData(["test-data/20201203-110526-et_timer_dco.tar"])
        preprocessed_data = raw_data.get_preprocessed_data()
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = PTAModel(by_name, parameters, arg_count)
        self.assertEqual(model.states, "IDLE".split(" "))
        self.assertEqual(
            model.transitions,
            "setup trans100u trans10m trans1m trans2m trans5m".split(" "),
        )
        static_model = model.get_static()
        self.assertAlmostEqual(static_model("IDLE", "power"), 764, places=0)
        self.assertAlmostEqual(static_model("setup", "duration"), 28, places=0)
        self.assertAlmostEqual(static_model("setup", "power"), 935, places=0)
        self.assertAlmostEqual(static_model("trans100u", "duration"), 158, places=0)
        self.assertAlmostEqual(static_model("trans100u", "power"), 935, places=0)
        self.assertAlmostEqual(static_model("trans10m", "duration"), 10097, places=0)
        self.assertAlmostEqual(static_model("trans10m", "power"), 6084, places=0)
        self.assertAlmostEqual(static_model("trans1m", "duration"), 1038, places=0)
        self.assertAlmostEqual(static_model("trans1m", "power"), 2913, places=0)
        self.assertAlmostEqual(static_model("trans2m", "duration"), 2044, places=0)
        self.assertAlmostEqual(static_model("trans2m", "power"), 4800, places=0)
        self.assertAlmostEqual(static_model("trans5m", "duration"), 5062, places=0)
        self.assertAlmostEqual(static_model("trans5m", "power"), 5900, places=0)

    def test_et_la_hfxt0(self):
        raw_data = RawData(["test-data/20201203-113313-et_la_hfxt0.tar"])
        preprocessed_data = raw_data.get_preprocessed_data()
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = PTAModel(by_name, parameters, arg_count)
        self.assertEqual(model.states, "IDLE".split(" "))
        self.assertEqual(
            model.transitions,
            "setup trans100u trans10m trans1m trans2m trans5m".split(" "),
        )
        static_model = model.get_static()
        self.assertAlmostEqual(static_model("IDLE", "power"), 1032, places=0)
        self.assertAlmostEqual(static_model("setup", "duration"), 15, places=0)
        self.assertAlmostEqual(static_model("setup", "power"), 1103, places=0)
        self.assertAlmostEqual(static_model("trans100u", "duration"), 147, places=0)
        self.assertAlmostEqual(static_model("trans100u", "power"), 1214, places=0)
        self.assertAlmostEqual(static_model("trans10m", "duration"), 10151, places=0)
        self.assertAlmostEqual(static_model("trans10m", "power"), 6519, places=0)
        self.assertAlmostEqual(static_model("trans1m", "duration"), 1032, places=0)
        self.assertAlmostEqual(static_model("trans1m", "power"), 5296, places=0)
        self.assertAlmostEqual(static_model("trans2m", "duration"), 2045, places=0)
        self.assertAlmostEqual(static_model("trans2m", "power"), 6074, places=0)
        self.assertAlmostEqual(static_model("trans5m", "duration"), 5083, places=0)
        self.assertAlmostEqual(static_model("trans5m", "power"), 6459, places=0)

    def test_et_timer_hfxt0(self):
        raw_data = RawData(["test-data/20201203-114004-et_timer_hfxt0.tar"])
        preprocessed_data = raw_data.get_preprocessed_data()
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = PTAModel(by_name, parameters, arg_count)
        self.assertEqual(model.states, "IDLE".split(" "))
        self.assertEqual(
            model.transitions,
            "setup trans100u trans10m trans1m trans2m trans5m".split(" "),
        )
        static_model = model.get_static()
        self.assertAlmostEqual(static_model("IDLE", "power"), 1044, places=0)
        self.assertAlmostEqual(static_model("setup", "duration"), 28, places=0)
        self.assertAlmostEqual(static_model("setup", "power"), 1982, places=0)
        self.assertAlmostEqual(static_model("trans100u", "duration"), 159, places=0)
        self.assertAlmostEqual(static_model("trans100u", "power"), 1408, places=0)
        self.assertAlmostEqual(static_model("trans10m", "duration"), 10164, places=0)
        self.assertAlmostEqual(static_model("trans10m", "power"), 6440, places=0)
        self.assertAlmostEqual(static_model("trans1m", "duration"), 1045, places=0)
        self.assertAlmostEqual(static_model("trans1m", "power"), 4253, places=0)
        self.assertAlmostEqual(static_model("trans2m", "duration"), 2058, places=0)
        self.assertAlmostEqual(static_model("trans2m", "power"), 5635, places=0)
        self.assertAlmostEqual(static_model("trans5m", "duration"), 5096, places=0)
        self.assertAlmostEqual(static_model("trans5m", "power"), 6405, places=0)

    def test_multifile_lm75x(self):
        testfiles = [
            "test-data/20170116_124500_LM75x.tar",
            "test-data/20170116_131306_LM75x.tar",
        ]
        raw_data = RawData(testfiles)
        preprocessed_data = raw_data.get_preprocessed_data()
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = PTAModel(by_name, parameters, arg_count)
        self.assertEqual(model.states, "ACTIVE POWEROFF".split(" "))
        self.assertEqual(
            model.transitions, "getTemp setHyst setOS shutdown start".split(" ")
        )
        static_model = model.get_static()
        self.assertAlmostEqual(static_model("ACTIVE", "power"), 332, places=0)
        self.assertAlmostEqual(static_model("POWEROFF", "power"), 7, places=0)
        self.assertAlmostEqual(static_model("getTemp", "power"), 2041, places=0)
        self.assertAlmostEqual(static_model("setHyst", "power"), 2415, places=0)
        self.assertAlmostEqual(static_model("setOS", "power"), 2382, places=0)
        self.assertAlmostEqual(static_model("shutdown", "power"), 1693, places=0)
        self.assertAlmostEqual(static_model("start", "power"), 1784, places=0)
        self.assertAlmostEqual(
            static_model("getTemp", "rel_power_prev"), 1704, places=0
        )
        self.assertAlmostEqual(
            static_model("setHyst", "rel_power_prev"), 2078, places=0
        )
        self.assertAlmostEqual(static_model("setOS", "rel_power_prev"), 2045, places=0)
        self.assertAlmostEqual(
            static_model("shutdown", "rel_power_prev"), 1683, places=0
        )
        self.assertAlmostEqual(static_model("start", "rel_power_prev"), 1776, places=0)
        """
        self.assertAlmostEqual(static_model("getTemp", "energy"), 26016748, places=0)
        self.assertAlmostEqual(static_model("setHyst", "energy"), 22082226, places=0)
        self.assertAlmostEqual(static_model("setOS", "energy"), 21774238, places=0)
        self.assertAlmostEqual(static_model("shutdown", "energy"), 11808160, places=0)
        self.assertAlmostEqual(static_model("start", "energy"), 12445302, places=0)
        self.assertAlmostEqual(
            static_model("getTemp", "rel_energy_prev"), 21722720, places=0
        )
        self.assertAlmostEqual(
            static_model("setHyst", "rel_energy_prev"), 19001499, places=0
        )
        self.assertAlmostEqual(
            static_model("setOS", "rel_energy_prev"), 18693283, places=0
        )
        self.assertAlmostEqual(
            static_model("shutdown", "rel_energy_prev"), 11746224, places=0
        )
        self.assertAlmostEqual(
            static_model("start", "rel_energy_prev"), 12391462, places=0
        )
        """
        self.assertAlmostEqual(static_model("getTemp", "duration"), 12740, places=0)
        self.assertAlmostEqual(static_model("setHyst", "duration"), 9140, places=0)
        self.assertAlmostEqual(static_model("setOS", "duration"), 9140, places=0)
        self.assertAlmostEqual(static_model("shutdown", "duration"), 6980, places=0)
        self.assertAlmostEqual(static_model("start", "duration"), 6980, places=0)

    def test_multifile_sharp(self):
        testfiles = [
            "test-data/20170116_145420_sharpLS013B4DN.tar",
            "test-data/20170116_151348_sharpLS013B4DN.tar",
        ]
        raw_data = RawData(testfiles)
        preprocessed_data = raw_data.get_preprocessed_data()
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = PTAModel(by_name, parameters, arg_count)
        self.assertEqual(model.states, "DISABLED ENABLED".split(" "))
        self.assertEqual(
            model.transitions,
            "clear disable enable ioInit sendLine toggleVCOM".split(" "),
        )
        static_model = model.get_static()
        self.assertAlmostEqual(static_model("DISABLED", "power"), 22, places=0)
        self.assertAlmostEqual(static_model("ENABLED", "power"), 24, places=0)
        self.assertAlmostEqual(static_model("clear", "power"), 464, places=0)
        self.assertAlmostEqual(static_model("disable", "power"), 1240, places=0)
        self.assertAlmostEqual(static_model("enable", "power"), 1170, places=0)
        self.assertAlmostEqual(static_model("ioInit", "power"), 22, places=0)
        self.assertAlmostEqual(static_model("sendLine", "power"), 211, places=0)
        self.assertAlmostEqual(static_model("toggleVCOM", "power"), 1025, places=0)
        self.assertAlmostEqual(static_model("clear", "rel_power_prev"), 439, places=0)
        self.assertAlmostEqual(
            static_model("disable", "rel_power_prev"), 1216, places=0
        )
        self.assertAlmostEqual(static_model("enable", "rel_power_prev"), 1148, places=0)
        self.assertAlmostEqual(static_model("ioInit", "rel_power_prev"), -2, places=0)
        self.assertAlmostEqual(
            static_model("sendLine", "rel_power_prev"), 186, places=0
        )
        self.assertAlmostEqual(
            static_model("toggleVCOM", "rel_power_prev"), 1000, places=0
        )
        """
        self.assertAlmostEqual(static_model("clear", "energy"), 14059, places=0)
        self.assertAlmostEqual(static_model("disable", "energy"), 0, places=0)
        self.assertAlmostEqual(static_model("enable", "energy"), 0, places=0)
        self.assertAlmostEqual(static_model("ioInit", "energy"), 0, places=0)
        self.assertAlmostEqual(static_model("sendLine", "energy"), 37874, places=0)
        self.assertAlmostEqual(static_model("toggleVCOM", "energy"), 30991, places=0)
        self.assertAlmostEqual(
            static_model("clear", "rel_energy_prev"), 13329, places=0
        )
        self.assertAlmostEqual(static_model("disable", "rel_energy_prev"), 0, places=0)
        self.assertAlmostEqual(static_model("enable", "rel_energy_prev"), 0, places=0)
        self.assertAlmostEqual(static_model("ioInit", "rel_energy_prev"), 0, places=0)
        self.assertAlmostEqual(
            static_model("sendLine", "rel_energy_prev"), 33447, places=0
        )
        self.assertAlmostEqual(
            static_model("toggleVCOM", "rel_energy_prev"), 30242, places=0
        )
        """
        self.assertAlmostEqual(static_model("clear", "duration"), 30, places=0)
        self.assertAlmostEqual(static_model("disable", "duration"), 0, places=0)
        self.assertAlmostEqual(static_model("enable", "duration"), 0, places=0)
        self.assertAlmostEqual(static_model("ioInit", "duration"), 0, places=0)
        self.assertAlmostEqual(static_model("sendLine", "duration"), 180, places=0)
        self.assertAlmostEqual(static_model("toggleVCOM", "duration"), 30, places=0)

    def test_multifile_mmstatic(self):
        testfiles = [
            "test-data/20170116_143516_mmstatic.tar",
            "test-data/20170116_142654_mmstatic.tar",
        ]
        raw_data = RawData(testfiles)
        preprocessed_data = raw_data.get_preprocessed_data()
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = PTAModel(by_name, parameters, arg_count)
        self.assertEqual(model.states, "B G OFF R".split(" "))
        self.assertEqual(model.transitions, "blue green off red".split(" "))
        static_model = model.get_static()
        self.assertAlmostEqual(static_model("B", "power"), 29443, places=0)
        self.assertAlmostEqual(static_model("G", "power"), 29432, places=0)
        self.assertAlmostEqual(static_model("OFF", "power"), 7057, places=0)
        self.assertAlmostEqual(static_model("R", "power"), 49068, places=0)
        self.assertAlmostEqual(static_model("blue", "power"), 40981, places=0)
        self.assertAlmostEqual(static_model("green", "power"), 40715, places=0)
        self.assertAlmostEqual(static_model("off", "power"), 40831, places=0)
        self.assertAlmostEqual(static_model("red", "power"), 41477, places=0)
        self.assertAlmostEqual(static_model("blue", "rel_power_prev"), 11551, places=0)
        self.assertAlmostEqual(static_model("green", "rel_power_prev"), 11278, places=0)
        self.assertAlmostEqual(static_model("off", "rel_power_prev"), 11342, places=0)
        self.assertAlmostEqual(static_model("red", "rel_power_prev"), 12090, places=0)
        """
        self.assertAlmostEqual(static_model("blue", "energy"), 374440955, places=0)
        self.assertAlmostEqual(static_model("green", "energy"), 372026027, places=0)
        self.assertAlmostEqual(static_model("off", "energy"), 372999554, places=0)
        self.assertAlmostEqual(static_model("red", "energy"), 378936634, places=0)
        self.assertAlmostEqual(
            static_model("blue", "rel_energy_prev"), 105535587, places=0
        )
        self.assertAlmostEqual(
            static_model("green", "rel_energy_prev"), 102999371, places=0
        )
        self.assertAlmostEqual(
            static_model("off", "rel_energy_prev"), 103613698, places=0
        )
        self.assertAlmostEqual(
            static_model("red", "rel_energy_prev"), 110474331, places=0
        )
        """
        self.assertAlmostEqual(static_model("blue", "duration"), 9140, places=0)
        self.assertAlmostEqual(static_model("green", "duration"), 9140, places=0)
        self.assertAlmostEqual(static_model("off", "duration"), 9140, places=0)
        self.assertAlmostEqual(static_model("red", "duration"), 9140, places=0)

    @pytest.mark.skipif(
        "TEST_SLOW" not in os.environ, reason="slow test, set TEST_SLOW=1 to run"
    )
    def test_multifile_cc1200(self):
        testfiles = [
            "test-data/20170125_125433_cc1200.tar",
            "test-data/20170125_142420_cc1200.tar",
            "test-data/20170125_144957_cc1200.tar",
            "test-data/20170125_151149_cc1200.tar",
            "test-data/20170125_151824_cc1200.tar",
            "test-data/20170125_154019_cc1200.tar",
        ]
        raw_data = RawData(testfiles)
        preprocessed_data = raw_data.get_preprocessed_data()
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = PTAModel(by_name, parameters, arg_count)
        self.assertEqual(
            model.states, "IDLE RX SLEEP SLEEP_EWOR SYNTH_ON TX XOFF".split(" ")
        )
        self.assertEqual(
            model.transitions,
            "crystal_off eWOR idle init prepare_xmit receive send setSymbolRate setTxPower sleep txDone".split(
                " "
            ),
        )
        static_model = model.get_static()
        self.assertAlmostEqual(static_model("IDLE", "power"), 9500, places=0)
        self.assertAlmostEqual(static_model("RX", "power"), 85177, places=0)
        self.assertAlmostEqual(static_model("SLEEP", "power"), 143, places=0)
        self.assertAlmostEqual(static_model("SLEEP_EWOR", "power"), 81801, places=0)
        self.assertAlmostEqual(static_model("SYNTH_ON", "power"), 60036, places=0)
        self.assertAlmostEqual(static_model("TX", "power"), 92461, places=0)
        self.assertAlmostEqual(static_model("XOFF", "power"), 780, places=0)
        self.assertAlmostEqual(static_model("crystal_off", "power"), 6140, places=0)
        self.assertAlmostEqual(static_model("eWOR", "power"), 7511, places=0)
        self.assertAlmostEqual(static_model("idle", "power"), 48579, places=0)
        self.assertAlmostEqual(static_model("init", "power"), 4627, places=0)
        self.assertAlmostEqual(static_model("prepare_xmit", "power"), 19079, places=0)
        self.assertAlmostEqual(static_model("receive", "power"), 19386, places=0)
        self.assertAlmostEqual(static_model("send", "power"), 10478, places=0)
        self.assertAlmostEqual(static_model("setSymbolRate", "power"), 9656, places=0)
        self.assertAlmostEqual(static_model("setTxPower", "power"), 9625, places=0)
        self.assertAlmostEqual(static_model("sleep", "power"), 5734, places=0)
        self.assertAlmostEqual(static_model("txDone", "power"), 9707, places=0)
        self.assertAlmostEqual(
            static_model("crystal_off", "rel_power_prev"), -3325, places=0
        )
        self.assertAlmostEqual(static_model("eWOR", "rel_power_prev"), -1968, places=0)
        self.assertAlmostEqual(static_model("idle", "rel_power_prev"), -33472, places=0)
        self.assertAlmostEqual(static_model("init", "rel_power_prev"), 4472, places=0)
        self.assertAlmostEqual(
            static_model("prepare_xmit", "rel_power_prev"), 9618, places=0
        )
        self.assertAlmostEqual(
            static_model("receive", "rel_power_prev"), 9887, places=0
        )
        self.assertAlmostEqual(static_model("send", "rel_power_prev"), 986, places=0)
        self.assertAlmostEqual(
            static_model("setSymbolRate", "rel_power_prev"), 160, places=0
        )
        self.assertAlmostEqual(
            static_model("setTxPower", "rel_power_prev"), 132, places=0
        )
        self.assertAlmostEqual(static_model("sleep", "rel_power_prev"), -3758, places=0)
        self.assertAlmostEqual(
            static_model("txDone", "rel_power_prev"), -82886, places=0
        )
        """
        self.assertAlmostEqual(static_model("crystal_off", "energy"), 114658, places=0)
        self.assertAlmostEqual(static_model("eWOR", "energy"), 317556, places=0)
        self.assertAlmostEqual(static_model("idle", "energy"), 717713, places=0)
        self.assertAlmostEqual(static_model("init", "energy"), 23028941, places=0)
        self.assertAlmostEqual(static_model("prepare_xmit", "energy"), 378552, places=0)
        self.assertAlmostEqual(static_model("receive", "energy"), 380335, places=0)
        self.assertAlmostEqual(static_model("send", "energy"), 4282597, places=0)
        self.assertAlmostEqual(
            static_model("setSymbolRate", "energy"), 962060, places=0
        )
        self.assertAlmostEqual(static_model("setTxPower", "energy"), 288701, places=0)
        self.assertAlmostEqual(static_model("sleep", "energy"), 104445, places=0)
        self.assertEqual(static_model("txDone", "energy"), 0)
        """

        param_model, param_info = model.get_fitted()
        self.assertIsInstance(param_info("IDLE", "power"), StaticFunction)
        self.assertEqual(
            param_info("RX", "power").model_function,
            "0 + regression_arg(0) + regression_arg(1) * np.log(parameter(symbolrate) + 1)",
        )
        self.assertIsInstance(param_info("SLEEP", "power"), StaticFunction)
        self.assertIsInstance(param_info("SLEEP_EWOR", "power"), StaticFunction)
        self.assertIsInstance(param_info("SYNTH_ON", "power"), StaticFunction)
        self.assertIsInstance(param_info("XOFF", "power"), StaticFunction)

        self.assertAlmostEqual(param_info("RX", "power").model_args[0], 84415, places=0)
        self.assertAlmostEqual(param_info("RX", "power").model_args[1], 206, places=0)


if __name__ == "__main__":
    unittest.main()
