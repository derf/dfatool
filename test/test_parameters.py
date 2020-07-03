#!/usr/bin/env python3

from dfatool import dfatool as dt
from dfatool import parameters
from dfatool.utils import by_name_to_by_param
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
        rng = np.random.default_rng()
        X = np.arange(200) % 50
        Y = X + rng.normal(size=X.size)
        by_name = {
            "TX": {
                "param": [(x % 5, x) for x in X],
                "power": Y,
                "attributes": ["power"],
            }
        }
        by_param = by_name_to_by_param(by_name)
        stats = parameters.ParamStats(by_name, by_param, ["p_mod5", "p_linear"], dict())

        self.assertEqual(stats.depends_on_param("TX", "power", "p_mod5"), False)
        self.assertEqual(stats.depends_on_param("TX", "power", "p_linear"), True)

        paramfit = dt.ParallelParamFit(by_param)
        paramfit.enqueue("TX", "power", 1, "p_linear")
        paramfit.fit()

        fit_result = paramfit.get_result("TX", "power")
        self.assertEqual(fit_result["p_linear"]["best"], "linear")


if __name__ == "__main__":
    unittest.main()
