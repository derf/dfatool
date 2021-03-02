#!/usr/bin/env python3

import logging
import numpy as np
from multiprocessing import Pool
from scipy import optimize
from .functions import analytic
from .utils import (
    is_numeric,
    param_slice_eq,
    remove_index_from_tuple,
    match_parameter_values,
    aggregate_measures,
    regression_measures,
)

logger = logging.getLogger(__name__)


class ParallelParamFit:
    """
    Fit a set of functions on parameterized measurements.

    One parameter is variale, all others are fixed. Reports the best-fitting
    function type for each parameter.
    """

    def __init__(self):
        """Create a new ParallelParamFit object."""
        self.fit_queue = list()

    def enqueue(self, key, param, args):
        """
        Add state_or_tran/attribute/param_name to fit queue.

        This causes fit() to compute the best-fitting function for this model part.

        :param key: arbitrary key used to retrieve param result in `get_result`. Typically (state/transition name, model attribute).
            Different parameter names may have the same key. Identical parameter names must have different keys.
        :param param: parameter name
        :param args: [by_param, param_index, safe_functions_enabled, param_filter]
            by_param[(param 1, param2, ...)] holds measurements.
        """
        self.fit_queue.append({"key": (key, param), "args": args})

    def fit(self):
        """
        Fit functions on previously enqueue data.

        Fitting is one in parallel with one process per core.

        Results can be accessed using the public ParallelParamFit.results object.
        """
        with Pool() as pool:
            self.results = pool.map(_try_fits_parallel, self.fit_queue)

    def get_result(self, key):
        """
        Parse and sanitize fit results.

        Filters out results where the best function is worse (or not much better than) static mean/median estimates.

        :param key: arbitrary key used in `enqueue`. Typically (state/transition name, model attribute).
        :param param: parameter name
        :returns: dict with fit result (see `_try_fits`) for each successfully fitted parameter. E.g. {'param 1': {'best' : 'function name', ...} }
        """
        fit_result = dict()
        for result in self.results:
            if (
                result["key"][0] == key and result["result"]["best"] is not None
            ):  # dürfte an ['best'] != None liegen-> Fit für gefilterten Kram schlägt fehl?
                this_result = result["result"]
                if this_result["best_rmsd"] >= min(
                    this_result["mean_rmsd"], this_result["median_rmsd"]
                ):
                    logger.debug(
                        "Not modeling {} as function of {}: best ({:.0f}) is worse than ref ({:.0f}, {:.0f})".format(
                            result["key"][0],
                            result["key"][1],
                            this_result["best_rmsd"],
                            this_result["mean_rmsd"],
                            this_result["median_rmsd"],
                        )
                    )
                # See notes on depends_on_param
                elif this_result["best_rmsd"] >= 0.8 * min(
                    this_result["mean_rmsd"], this_result["median_rmsd"]
                ):
                    logger.debug(
                        "Not modeling {} as function of {}: best ({:.0f}) is not much better than ref ({:.0f}, {:.0f})".format(
                            result["key"][0],
                            result["key"][1],
                            this_result["best_rmsd"],
                            this_result["mean_rmsd"],
                            this_result["median_rmsd"],
                        )
                    )
                else:
                    fit_result[result["key"][1]] = this_result
        return fit_result


def _try_fits_parallel(arg):
    """
    Call _try_fits(*arg['args']) and return arg['key'] and the _try_fits result.

    Must be a global function as it is called from a multiprocessing Pool.
    """
    return {"key": arg["key"], "result": _try_fits(*arg["args"])}


def _try_fits(
    n_by_param, param_index, safe_functions_enabled=False, param_filter: dict = None
):
    """
    Determine goodness-of-fit for prediction of `n_by_param[(param1_value, param2_value, ...)]` dependence on `param_index` using various functions.

    This is done by varying `param_index` while keeping all other parameters constant and doing one least squares optimization for each function and for each combination of the remaining parameters.
    The value of the parameter corresponding to `param_index` (e.g. txpower or packet length) is the sole input to the model function.
    Only numeric parameter values (as determined by `utils.is_numeric`) are used for fitting, non-numeric values such as None or enum strings are ignored.
    Fitting is only performed if at least three distinct parameter values exist in `by_param[*]`.

    :returns:  a dictionary with the following elements:
        best -- name of the best-fitting function (see `analytic.functions`). `None` in case of insufficient data.
        best_rmsd -- mean Root Mean Square Deviation of best-fitting function over all combinations of the remaining parameters
        mean_rmsd -- mean Root Mean Square Deviation of a reference model using the mean of its respective input data as model value
        median_rmsd -- mean Root Mean Square Deviation of a reference model using the median of its respective input data as model value
        results -- mean goodness-of-fit measures for the individual functions. See `analytic.functions` for keys and `aggregate_measures` for values

    :param n_by_param: measurements of a specific model attribute partitioned by parameter values.
        Example: `{(0, 2): [2], (0, 4): [4], (0, 6): [6]}`

    :param param_index: index of the parameter used as model input
    :param safe_functions_enabled: Include "safe" variants of functions with limited argument range.
    :param param_filter: Only use measurements whose parameters match param_filter for fitting.
    """

    functions = analytic.functions(safe_functions_enabled=safe_functions_enabled)

    for param_key in n_by_param.keys():
        # We might remove elements from 'functions' while iterating over
        # its keys. A generator will not allow this, so we need to
        # convert to a list.
        function_names = list(functions.keys())
        for function_name in function_names:
            function_object = functions[function_name]
            if is_numeric(param_key[param_index]) and not function_object.is_valid(
                param_key[param_index]
            ):
                functions.pop(function_name, None)

    raw_results = dict()
    raw_results_by_param = dict()
    ref_results = {"mean": list(), "median": list()}
    results = dict()
    results_by_param = dict()

    seen_parameter_combinations = set()

    # for each parameter combination:
    for param_key in filter(
        lambda x: remove_index_from_tuple(x, param_index)
        not in seen_parameter_combinations
        and len(n_by_param[x])
        and match_parameter_values(n_by_param[x][0], param_filter),
        n_by_param.keys(),
    ):
        X = []
        Y = []
        num_valid = 0
        num_total = 0

        # Ensure that each parameter combination is only optimized once. Otherwise, with parameters (1, 2, 5), (1, 3, 5), (1, 4, 5) and param_index == 1,
        # the parameter combination (1, *, 5) would be optimized three times, both wasting time and biasing results towards more frequently occuring combinations of non-param_index parameters
        seen_parameter_combinations.add(remove_index_from_tuple(param_key, param_index))

        # for each value of the parameter denoted by param_index (all other parameters remain the same):
        for k, v in filter(
            lambda kv: param_slice_eq(kv[0], param_key, param_index), n_by_param.items()
        ):
            num_total += 1
            if is_numeric(k[param_index]):
                num_valid += 1
                X.extend([float(k[param_index])] * len(v))
                Y.extend(v)

        if num_valid > 2:
            X = np.array(X)
            Y = np.array(Y)
            other_parameters = remove_index_from_tuple(k, param_index)
            raw_results_by_param[other_parameters] = dict()
            results_by_param[other_parameters] = dict()
            for function_name, param_function in functions.items():
                if function_name not in raw_results:
                    raw_results[function_name] = dict()
                error_function = param_function.error_function
                res = optimize.least_squares(
                    error_function, [0, 1], args=(X, Y), xtol=2e-15
                )
                measures = regression_measures(param_function.eval(res.x, X), Y)
                raw_results_by_param[other_parameters][function_name] = measures
                for measure, error_rate in measures.items():
                    if measure not in raw_results[function_name]:
                        raw_results[function_name][measure] = list()
                    raw_results[function_name][measure].append(error_rate)
                # print(function_name, res, measures)
            mean_measures = aggregate_measures(np.mean(Y), Y)
            ref_results["mean"].append(mean_measures["rmsd"])
            raw_results_by_param[other_parameters]["mean"] = mean_measures
            median_measures = aggregate_measures(np.median(Y), Y)
            ref_results["median"].append(median_measures["rmsd"])
            raw_results_by_param[other_parameters]["median"] = median_measures

    if not len(ref_results["mean"]):
        # Insufficient data for fitting
        # print('[W] Insufficient data for fitting {}'.format(param_index))
        return {"best": None, "best_rmsd": np.inf, "results": results}

    for (
        other_parameter_combination,
        other_parameter_results,
    ) in raw_results_by_param.items():
        best_fit_val = np.inf
        best_fit_name = None
        results = dict()
        for function_name, result in other_parameter_results.items():
            if len(result) > 0:
                results[function_name] = result
                rmsd = result["rmsd"]
                if rmsd < best_fit_val:
                    best_fit_val = rmsd
                    best_fit_name = function_name
        results_by_param[other_parameter_combination] = {
            "best": best_fit_name,
            "best_rmsd": best_fit_val,
            "mean_rmsd": results["mean"]["rmsd"],
            "median_rmsd": results["median"]["rmsd"],
            "results": results,
        }

    best_fit_val = np.inf
    best_fit_name = None
    results = dict()
    for function_name, result in raw_results.items():
        if len(result) > 0:
            results[function_name] = {}
            for measure in result.keys():
                results[function_name][measure] = np.mean(result[measure])
            rmsd = results[function_name]["rmsd"]
            if rmsd < best_fit_val:
                best_fit_val = rmsd
                best_fit_name = function_name

    return {
        "best": best_fit_name,
        "best_rmsd": best_fit_val,
        "mean_rmsd": np.mean(ref_results["mean"]),
        "median_rmsd": np.mean(ref_results["median"]),
        "results": results,
        "results_by_other_param": results_by_param,
    }
