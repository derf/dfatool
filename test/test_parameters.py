#!/usr/bin/env python3

from dfatool import parameters
from dfatool.utils import by_name_to_by_param
from dfatool.functions import analytic
from dfatool.model import ParallelParamFit
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
            parameters.distinct_param_values(by_name["TX"]["param"]),
            [list(range(5)), list(range(7))],
        )

    def test_parameter_detection_linear(self):
        # rng = np.random.default_rng(seed=1312) # requiresy NumPy >= 1.17
        np.random.seed(1312)
        X = np.arange(200) % 50
        # Y = X + rng.normal(size=X.size) # requiry NumPy >= 1.17
        Y = X + np.random.normal(size=X.size)
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

        stats = parameters.ParamStats(
            parameters._compute_param_statistics(
                by_name["TX"]["power"], parameter_names, by_name["TX"]["param"]
            )
        )

        self.assertEqual(stats.depends_on_param("p_mod5"), False)
        self.assertEqual(stats.depends_on_param("p_linear"), True)

        # Fit individual functions for each parameter (only "p_linear" in this case)

        paramfit = ParallelParamFit()
        paramfit.enqueue(("TX", "power"), "p_linear", (stats.by_param, 1, False))
        paramfit.fit()

        fit_result = paramfit.get_result(("TX", "power"))
        self.assertEqual(fit_result["p_linear"]["best"], "linear")
        self.assertEqual("p_mod5" not in fit_result, True)

        # Fit a single function for all parameters (still only "p_linear" in this case)

        combined_fit = analytic.function_powerset(fit_result, parameter_names, 0)

        self.assertEqual(
            combined_fit.model_function,
            "0 + regression_arg(0) + regression_arg(1) * parameter(p_linear)",
        )
        self.assertEqual(
            combined_fit._function_str,
            "0 + reg_param[0] + reg_param[1] * model_param[1]",
        )

        combined_fit.fit(stats.by_param)

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

    def test_parameter_detection_multi_dimensional(self):
        # rng = np.random.default_rng(seed=1312) # requires NumPy >= 1.17
        np.random.seed(1312)
        # vary each parameter from 1 to 10
        Xi = (np.arange(50) % 10) + 1
        # Three parameters -> Build input array [[1, 1, 1], [1, 1, 2], ..., [10, 10, 10]]
        X = np.array(np.meshgrid(Xi, Xi, Xi)).T.reshape(-1, 3)

        f_lls = np.vectorize(
            lambda x: 42 + 7 * x[0] + 10 * np.log(x[1]) - 0.5 * x[2] * x[2],
            signature="(n)->()",
        )
        f_ll = np.vectorize(
            lambda x: 23 + 5 * x[0] - 3 * x[0] / x[1], signature="(n)->()"
        )

        # Y_lls = f_lls(X) + rng.normal(size=X.shape[0]) # requires NumPy >= 1.17
        # Y_ll = f_ll(X) + rng.normal(size=X.shape[0]) # requires NumPy >= 1.17
        Y_lls = f_lls(X) + np.random.normal(size=X.shape[0])
        Y_ll = f_ll(X) + np.random.normal(size=X.shape[0])

        parameter_names = ["lin_lin", "log_inv", "square_none"]

        by_name = {
            "someKey": {
                "param": X,
                "lls": Y_lls,
                "ll": Y_ll,
                "attributes": ["lls", "ll"],
            }
        }
        by_param = by_name_to_by_param(by_name)
        lls_stats = parameters.ParamStats(
            parameters._compute_param_statistics(
                by_name["someKey"]["lls"], parameter_names, by_name["someKey"]["param"]
            )
        )
        ll_stats = parameters.ParamStats(
            parameters._compute_param_statistics(
                by_name["someKey"]["ll"], parameter_names, by_name["someKey"]["param"]
            )
        )

        self.assertEqual(lls_stats.depends_on_param("lin_lin"), True)
        self.assertEqual(lls_stats.depends_on_param("log_inv"), True)
        self.assertEqual(lls_stats.depends_on_param("square_none"), True)

        self.assertEqual(ll_stats.depends_on_param("lin_lin"), True)
        self.assertEqual(ll_stats.depends_on_param("log_inv"), True)
        self.assertEqual(ll_stats.depends_on_param("square_none"), False)

        paramfit = ParallelParamFit()
        paramfit.enqueue(("someKey", "lls"), "lin_lin", (lls_stats.by_param, 0, False))
        paramfit.enqueue(("someKey", "lls"), "log_inv", (lls_stats.by_param, 1, False))
        paramfit.enqueue(
            ("someKey", "lls"), "square_none", (lls_stats.by_param, 2, False)
        )
        paramfit.enqueue(("someKey", "ll"), "lin_lin", (ll_stats.by_param, 0, False))
        paramfit.enqueue(("someKey", "ll"), "log_inv", (ll_stats.by_param, 1, False))
        paramfit.fit()

        fit_lls = paramfit.get_result(("someKey", "lls"))
        self.assertEqual(fit_lls["lin_lin"]["best"], "linear")
        self.assertEqual(fit_lls["log_inv"]["best"], "logarithmic")
        self.assertEqual(fit_lls["square_none"]["best"], "square")

        combined_fit_lls = analytic.function_powerset(fit_lls, parameter_names, 0)

        self.assertEqual(
            combined_fit_lls.model_function,
            "0 + regression_arg(0) + regression_arg(1) * parameter(lin_lin)"
            " + regression_arg(2) * np.log(parameter(log_inv))"
            " + regression_arg(3) * (parameter(square_none))**2"
            " + regression_arg(4) * parameter(lin_lin) * np.log(parameter(log_inv))"
            " + regression_arg(5) * parameter(lin_lin) * (parameter(square_none))**2"
            " + regression_arg(6) * np.log(parameter(log_inv)) * (parameter(square_none))**2"
            " + regression_arg(7) * parameter(lin_lin) * np.log(parameter(log_inv)) * (parameter(square_none))**2",
        )

        combined_fit_lls.fit(lls_stats.by_param)

        self.assertEqual(combined_fit_lls.fit_success, True)

        # Verify that f_lls parameters have been found
        self.assertAlmostEqual(combined_fit_lls.model_args[0], 42, places=0)
        self.assertAlmostEqual(combined_fit_lls.model_args[1], 7, places=0)
        self.assertAlmostEqual(combined_fit_lls.model_args[2], 10, places=0)
        self.assertAlmostEqual(combined_fit_lls.model_args[3], -0.5, places=1)
        self.assertAlmostEqual(combined_fit_lls.model_args[4], 0, places=2)
        self.assertAlmostEqual(combined_fit_lls.model_args[5], 0, places=2)
        self.assertAlmostEqual(combined_fit_lls.model_args[6], 0, places=2)
        self.assertAlmostEqual(combined_fit_lls.model_args[7], 0, places=2)

        self.assertEqual(combined_fit_lls.is_predictable([None, None, None]), False)
        self.assertEqual(combined_fit_lls.is_predictable([None, None, 11]), False)
        self.assertEqual(combined_fit_lls.is_predictable([None, 11, None]), False)
        self.assertEqual(combined_fit_lls.is_predictable([None, 11, 11]), False)
        self.assertEqual(combined_fit_lls.is_predictable([11, None, None]), False)
        self.assertEqual(combined_fit_lls.is_predictable([11, None, 11]), False)
        self.assertEqual(combined_fit_lls.is_predictable([11, 11, None]), False)
        self.assertEqual(combined_fit_lls.is_predictable([11, 11, 11]), True)

        # Verify that fitted function behaves like input function
        for i, x in enumerate(X):
            self.assertAlmostEqual(combined_fit_lls.eval(x), f_lls(x), places=0)

        fit_ll = paramfit.get_result(("someKey", "ll"))
        self.assertEqual(fit_ll["lin_lin"]["best"], "linear")
        self.assertEqual(fit_ll["log_inv"]["best"], "inverse")
        self.assertEqual("quare_none" not in fit_ll, True)

        combined_fit_ll = analytic.function_powerset(fit_ll, parameter_names, 0)

        self.assertEqual(
            combined_fit_ll.model_function,
            "0 + regression_arg(0) + regression_arg(1) * parameter(lin_lin)"
            " + regression_arg(2) * 1/(parameter(log_inv))"
            " + regression_arg(3) * parameter(lin_lin) * 1/(parameter(log_inv))",
        )

        combined_fit_ll.fit(ll_stats.by_param)

        self.assertEqual(combined_fit_ll.fit_success, True)

        # Verify that f_ll parameters have been found
        self.assertAlmostEqual(combined_fit_ll.model_args[0], 23, places=0)
        self.assertAlmostEqual(combined_fit_ll.model_args[1], 5, places=0)
        self.assertAlmostEqual(combined_fit_ll.model_args[2], 0, places=1)
        self.assertAlmostEqual(combined_fit_ll.model_args[3], -3, places=0)

        self.assertEqual(combined_fit_ll.is_predictable([None, None, None]), False)
        self.assertEqual(combined_fit_ll.is_predictable([None, None, 11]), False)
        self.assertEqual(combined_fit_ll.is_predictable([None, 11, None]), False)
        self.assertEqual(combined_fit_ll.is_predictable([None, 11, 11]), False)
        self.assertEqual(combined_fit_ll.is_predictable([11, None, None]), False)
        self.assertEqual(combined_fit_ll.is_predictable([11, None, 11]), False)
        self.assertEqual(combined_fit_ll.is_predictable([11, 11, None]), True)
        self.assertEqual(combined_fit_ll.is_predictable([11, 11, 11]), True)

        # Verify that fitted function behaves like input function
        for i, x in enumerate(X):
            self.assertAlmostEqual(combined_fit_ll.eval(x), f_ll(x), places=0)


if __name__ == "__main__":
    unittest.main()
