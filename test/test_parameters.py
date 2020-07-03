#!/usr/bin/env python3

from dfatool import dfatool as dt
from dfatool import parameters
from dfatool.utils import by_name_to_by_param
from dfatool.functions import analytic
import unittest

import numpy as np


class TestModels(unittest.TestCase):
    def test_distinct_param_values(self):
        X = np.arange(35)
        by_name = {
            "TX": {
                "param": [(x % 5, x % 7) for x in X],
                "power": X,
                "attributes": ["power"],
            }
        }
        self.assertEqual(
            parameters.distinct_param_values(by_name, "TX"),
            [list(range(5)), list(range(7))],
        )

    def test_parameter_detection_linear(self):
        rng = np.random.default_rng(seed=1312)
        X = np.arange(200) % 50
        Y = X + rng.normal(size=X.size)
        parameter_names = ["p_mod5", "p_linear"]

        # Test input data:
        # * param[0] ("p_mod5") == X % 5 (bogus data to test detection of non-influence)
        # * param[1] ("p_linear") == X
        # * TX power == X ± gaussian noise
        # -> TX power depends linearly on "p_linear"
        by_name = {
            "TX": {
                "param": [(x % 5, x) for x in X],
                "power": Y,
                "attributes": ["power"],
            }
        }
        by_param = by_name_to_by_param(by_name)
        stats = parameters.ParamStats(by_name, by_param, parameter_names, dict())

        self.assertEqual(stats.depends_on_param("TX", "power", "p_mod5"), False)
        self.assertEqual(stats.depends_on_param("TX", "power", "p_linear"), True)

        # Fit individual functions for each parameter (only "p_linear" in this case)

        paramfit = dt.ParallelParamFit(by_param)
        paramfit.enqueue("TX", "power", 1, "p_linear")
        paramfit.fit()

        fit_result = paramfit.get_result("TX", "power")
        self.assertEqual(fit_result["p_linear"]["best"], "linear")
        self.assertEqual("p_mod5" not in fit_result, True)

        # Fit a single function for all parameters (still only "p_linear" in this case)

        combined_fit = analytic.function_powerset(fit_result, parameter_names, 0)

        self.assertEqual(
            combined_fit._model_str,
            "0 + regression_arg(0) + regression_arg(1) * parameter(p_linear)",
        )
        self.assertEqual(
            combined_fit._function_str,
            "0 + reg_param[0] + reg_param[1] * model_param[1]",
        )

        combined_fit.fit(by_param, "TX", "power")

        self.assertEqual(combined_fit.fit_success, True)

        self.assertEqual(combined_fit.is_predictable([None, None]), False)
        self.assertEqual(combined_fit.is_predictable([None, 0]), True)
        self.assertEqual(combined_fit.is_predictable([None, 50]), True)
        self.assertEqual(combined_fit.is_predictable([0, None]), False)
        self.assertEqual(combined_fit.is_predictable([50, None]), False)
        self.assertEqual(combined_fit.is_predictable([0, 0]), True)
        self.assertEqual(combined_fit.is_predictable([0, 50]), True)
        self.assertEqual(combined_fit.is_predictable([50, 0]), True)
        self.assertEqual(combined_fit.is_predictable([50, 50]), True)

        # The function should be linear without offset or skew
        for i in range(100):
            self.assertAlmostEqual(combined_fit.eval([None, i]), i, places=0)


if __name__ == "__main__":
    unittest.main()
