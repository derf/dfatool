#!/usr/bin/env python3

from dfatool.functions import StaticFunction
from dfatool.loader import TimingData, pta_trace_to_aggregate
from dfatool.model import AnalyticModel
import os
import unittest


class TestModels(unittest.TestCase):
    def test_model_singlefile_rf24(self):
        raw_data = TimingData(["test-data/20190815_111745_nRF24_no-rx.json"])
        preprocessed_data = raw_data.get_preprocessed_data()
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = AnalyticModel(by_name, parameters, arg_count)
        self.assertEqual(model.names, "setPALevel setRetries setup write".split(" "))
        static_model = model.get_static()
        self.assertAlmostEqual(static_model("setPALevel", "duration"), 146, places=0)
        self.assertAlmostEqual(static_model("setRetries", "duration"), 73, places=0)
        self.assertAlmostEqual(static_model("setup", "duration"), 6533, places=0)
        self.assertAlmostEqual(static_model("write", "duration"), 12634, places=0)

        for transition in "setPALevel setRetries setup write".split(" "):
            self.assertAlmostEqual(
                model.attr_by_name[transition]["duration"].stats.param_dependence_ratio(
                    "channel"
                ),
                0,
                places=2,
            )

        param_model, param_info = model.get_fitted()
        self.assertIsInstance(param_info("setPALevel", "duration"), StaticFunction)
        self.assertIsInstance(param_info("setRetries", "duration"), StaticFunction)
        self.assertIsInstance(param_info("setup", "duration"), StaticFunction)
        self.assertEqual(
            param_info("write", "duration").model_function,
            "0 + regression_arg(0) + regression_arg(1) * parameter(max_retry_count) + regression_arg(2) * parameter(retry_delay) + regression_arg(3) * function_arg(1) + regression_arg(4) * parameter(max_retry_count) * parameter(retry_delay) + regression_arg(5) * parameter(max_retry_count) * function_arg(1) + regression_arg(6) * parameter(retry_delay) * function_arg(1) + regression_arg(7) * parameter(max_retry_count) * parameter(retry_delay) * function_arg(1)",
        )

        self.assertAlmostEqual(
            param_info("write", "duration").model_args[0], 1281, places=0
        )
        self.assertAlmostEqual(
            param_info("write", "duration").model_args[1], 464, places=0
        )
        self.assertAlmostEqual(
            param_info("write", "duration").model_args[2], 1, places=0
        )
        self.assertAlmostEqual(
            param_info("write", "duration").model_args[3], -9, places=0
        )
        self.assertAlmostEqual(
            param_info("write", "duration").model_args[4], 1, places=0
        )
        self.assertAlmostEqual(
            param_info("write", "duration").model_args[5], 0, places=0
        )
        self.assertAlmostEqual(
            param_info("write", "duration").model_args[6], 0, places=0
        )
        self.assertAlmostEqual(
            param_info("write", "duration").model_args[7], 0, places=0
        )

    def test_dependent_parameter_pruning(self):
        raw_data = TimingData(["test-data/20190815_103347_nRF24_no-rx.json"])
        preprocessed_data = raw_data.get_preprocessed_data()
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = AnalyticModel(by_name, parameters, arg_count)
        self.assertEqual(
            model.names, "getObserveTx setPALevel setRetries setup write".split(" ")
        )
        static_model = model.get_static()
        self.assertAlmostEqual(static_model("getObserveTx", "duration"), 75, places=0)
        self.assertAlmostEqual(static_model("setPALevel", "duration"), 146, places=0)
        self.assertAlmostEqual(static_model("setRetries", "duration"), 73, places=0)
        self.assertAlmostEqual(static_model("setup", "duration"), 6533, places=0)
        self.assertAlmostEqual(static_model("write", "duration"), 12634, places=0)

        for transition in "getObserveTx setPALevel setRetries setup write".split(" "):
            self.assertAlmostEqual(
                model.attr_by_name[transition]["duration"].stats.param_dependence_ratio(
                    "channel"
                ),
                0,
                places=2,
            )

        param_model, param_info = model.get_fitted()
        self.assertIsInstance(param_info("getObserveTx", "duration"), StaticFunction)
        self.assertIsInstance(param_info("setPALevel", "duration"), StaticFunction)
        self.assertIsInstance(param_info("setRetries", "duration"), StaticFunction)
        self.assertIsInstance(param_info("setup", "duration"), StaticFunction)
        self.assertEqual(
            param_info("write", "duration").model_function,
            "0 + regression_arg(0) + regression_arg(1) * parameter(max_retry_count) + regression_arg(2) * parameter(retry_delay) + regression_arg(3) * function_arg(1) + regression_arg(4) * parameter(max_retry_count) * parameter(retry_delay) + regression_arg(5) * parameter(max_retry_count) * function_arg(1) + regression_arg(6) * parameter(retry_delay) * function_arg(1) + regression_arg(7) * parameter(max_retry_count) * parameter(retry_delay) * function_arg(1)",
        )

        self.assertAlmostEqual(
            param_info("write", "duration").model_args[0], 1282, places=0
        )
        self.assertAlmostEqual(
            param_info("write", "duration").model_args[1], 463, places=0
        )
        self.assertAlmostEqual(
            param_info("write", "duration").model_args[2], 1, places=0
        )
        self.assertAlmostEqual(
            param_info("write", "duration").model_args[3], -9, places=0
        )
        self.assertAlmostEqual(
            param_info("write", "duration").model_args[4], 1, places=0
        )
        self.assertAlmostEqual(
            param_info("write", "duration").model_args[5], 0, places=0
        )
        self.assertAlmostEqual(
            param_info("write", "duration").model_args[6], 0, places=0
        )
        self.assertAlmostEqual(
            param_info("write", "duration").model_args[7], 0, places=0
        )

    def test_function_override(self):
        os.environ["DFATOOL_NO_DECISIONTREES"] = "1"
        raw_data = TimingData(["test-data/20190815_122531_nRF24_no-rx.json"])
        preprocessed_data = raw_data.get_preprocessed_data()
        by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
        model = AnalyticModel(
            by_name,
            parameters,
            arg_count,
            function_override={
                (
                    "write",
                    "duration",
                ): "(parameter(auto_ack!) * (regression_arg(0) + regression_arg(1) * parameter(max_retry_count) + regression_arg(2) * parameter(retry_delay) + regression_arg(3) * parameter(max_retry_count) * parameter(retry_delay))) + ((1 - parameter(auto_ack!)) * regression_arg(4))"
            },
        )
        self.assertEqual(
            model.names, "setAutoAck setPALevel setRetries setup write".split(" ")
        )
        static_model = model.get_static()
        self.assertAlmostEqual(static_model("setAutoAck", "duration"), 72, places=0)
        self.assertAlmostEqual(static_model("setPALevel", "duration"), 146, places=0)
        self.assertAlmostEqual(static_model("setRetries", "duration"), 73, places=0)
        self.assertAlmostEqual(static_model("setup", "duration"), 6533, places=0)
        self.assertAlmostEqual(static_model("write", "duration"), 1181, places=0)

        for transition in "setAutoAck setPALevel setRetries setup write".split(" "):
            self.assertAlmostEqual(
                model.attr_by_name[transition]["duration"].stats.param_dependence_ratio(
                    "channel"
                ),
                0,
                places=2,
            )

        param_model, param_info = model.get_fitted()
        self.assertIsInstance(param_info("setAutoAck", "duration"), StaticFunction)
        self.assertIsInstance(param_info("setPALevel", "duration"), StaticFunction)
        self.assertIsInstance(param_info("setRetries", "duration"), StaticFunction)
        self.assertIsInstance(param_info("setup", "duration"), StaticFunction)
        self.assertEqual(
            param_info("write", "duration").model_function,
            "(parameter(auto_ack!) * (regression_arg(0) + regression_arg(1) * parameter(max_retry_count) + regression_arg(2) * parameter(retry_delay) + regression_arg(3) * parameter(max_retry_count) * parameter(retry_delay))) + ((1 - parameter(auto_ack!)) * regression_arg(4))",
        )

        self.assertAlmostEqual(
            param_info("write", "duration").model_args[0], 1162, places=0
        )
        self.assertAlmostEqual(
            param_info("write", "duration").model_args[1], 464, places=0
        )
        self.assertAlmostEqual(
            param_info("write", "duration").model_args[2], 1, places=0
        )
        self.assertAlmostEqual(
            param_info("write", "duration").model_args[3], 1, places=0
        )
        self.assertAlmostEqual(
            param_info("write", "duration").model_args[4], 1086, places=0
        )
        os.environ.pop("DFATOOL_NO_DECISIONTREES")


if __name__ == "__main__":
    unittest.main()
