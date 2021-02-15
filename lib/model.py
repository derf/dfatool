#!/usr/bin/env python3

import logging
import numpy as np
from scipy import optimize
from sklearn.metrics import r2_score
from multiprocessing import Pool
from .automata import PTA
from .functions import analytic
from .functions import AnalyticFunction
from .parameters import ParamStats
from .utils import is_numeric, soft_cast_int, param_slice_eq, remove_index_from_tuple
from .utils import by_name_to_by_param, by_param_to_by_name, match_parameter_values

logger = logging.getLogger(__name__)
arg_support_enabled = True


def aggregate_measures(aggregate: float, actual: list) -> dict:
    """
    Calculate error measures for model value on data list.

    arguments:
    aggregate -- model value (float or int)
    actual -- real-world / reference values (list of float or int)

    return value:
    See regression_measures
    """
    aggregate_array = np.array([aggregate] * len(actual))
    return regression_measures(aggregate_array, np.array(actual))


def regression_measures(predicted: np.ndarray, actual: np.ndarray):
    """
    Calculate error measures by comparing model values to reference values.

    arguments:
    predicted -- model values (np.ndarray)
    actual -- real-world / reference values (np.ndarray)

    Returns a dict containing the following measures:
    mae -- Mean Absolute Error
    mape -- Mean Absolute Percentage Error,
            if all items in actual are non-zero (NaN otherwise)
    smape -- Symmetric Mean Absolute Percentage Error,
             if no 0,0-pairs are present in actual and predicted (NaN otherwise)
    msd -- Mean Square Deviation
    rmsd -- Root Mean Square Deviation
    ssr -- Sum of Squared Residuals
    rsq -- R^2 measure, see sklearn.metrics.r2_score
    count -- Number of values
    """
    if type(predicted) != np.ndarray:
        raise ValueError("first arg must be ndarray, is {}".format(type(predicted)))
    if type(actual) != np.ndarray:
        raise ValueError("second arg must be ndarray, is {}".format(type(actual)))
    deviations = predicted - actual
    # mean = np.mean(actual)
    if len(deviations) == 0:
        return {}
    measures = {
        "mae": np.mean(np.abs(deviations), dtype=np.float64),
        "msd": np.mean(deviations ** 2, dtype=np.float64),
        "rmsd": np.sqrt(np.mean(deviations ** 2), dtype=np.float64),
        "ssr": np.sum(deviations ** 2, dtype=np.float64),
        "rsq": r2_score(actual, predicted),
        "count": len(actual),
    }

    # rsq_quotient = np.sum((actual - mean)**2, dtype=np.float64) * np.sum((predicted - mean)**2, dtype=np.float64)

    if np.all(actual != 0):
        measures["mape"] = np.mean(np.abs(deviations / actual)) * 100  # bad measure
    else:
        measures["mape"] = np.nan
    if np.all(np.abs(predicted) + np.abs(actual) != 0):
        measures["smape"] = (
            np.mean(np.abs(deviations) / ((np.abs(predicted) + np.abs(actual)) / 2))
            * 100
        )
    else:
        measures["smape"] = np.nan
    # if np.all(rsq_quotient != 0):
    #    measures['rsq'] = (np.sum((actual - mean) * (predicted - mean), dtype=np.float64)**2) / rsq_quotient

    return measures


class ParallelParamFit:
    """
    Fit a set of functions on parameterized measurements.

    One parameter is variale, all others are fixed. Reports the best-fitting
    function type for each parameter.
    """

    def __init__(self, by_param):
        """Create a new ParallelParamFit object."""
        self.fit_queue = []
        self.by_param = by_param

    def enqueue(
        self,
        state_or_tran,
        attribute,
        param_index,
        param_name,
        safe_functions_enabled=False,
        param_filter=None,
    ):
        """
        Add state_or_tran/attribute/param_name to fit queue.

        This causes fit() to compute the best-fitting function for this model part.
        """
        # Transform by_param[(state_or_tran, param_value)][attribute] = ...
        # into n_by_param[param_value] = ...
        # (param_value is dynamic, the rest is fixed)
        n_by_param = dict()
        for k, v in self.by_param.items():
            if k[0] == state_or_tran:
                n_by_param[k[1]] = v[attribute]
        self.fit_queue.append(
            {
                "key": [state_or_tran, attribute, param_name, param_filter],
                "args": [n_by_param, param_index, safe_functions_enabled, param_filter],
            }
        )

    def fit(self):
        """
        Fit functions on previously enqueue data.

        Fitting is one in parallel with one process per core.

        Results can be accessed using the public ParallelParamFit.results object.
        """
        with Pool() as pool:
            self.results = pool.map(_try_fits_parallel, self.fit_queue)

    def get_result(self, name, attribute, param_filter: dict = None):
        """
        Parse and sanitize fit results for state/transition/... 'name' and model attribute 'attribute'.

        Filters out results where the best function is worse (or not much better than) static mean/median estimates.

        :param name: state/transition/... name, e.g. 'TX'
        :param attribute: model attribute, e.g. 'duration'
        :param param_filter:
        :returns: dict with fit result (see `_try_fits`) for each successfully fitted parameter. E.g. {'param 1': {'best' : 'function name', ...} }
        """
        fit_result = dict()
        for result in self.results:
            if (
                result["key"][0] == name
                and result["key"][1] == attribute
                and result["key"][3] == param_filter
                and result["result"]["best"] is not None
            ):  # dürfte an ['best'] != None liegen-> Fit für gefilterten Kram schlägt fehl?
                this_result = result["result"]
                if this_result["best_rmsd"] >= min(
                    this_result["mean_rmsd"], this_result["median_rmsd"]
                ):
                    logger.debug(
                        "Not modeling {} {} as function of {}: best ({:.0f}) is worse than ref ({:.0f}, {:.0f})".format(
                            name,
                            attribute,
                            result["key"][2],
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
                        "Not modeling {} {} as function of {}: best ({:.0f}) is not much better than ref ({:.0f}, {:.0f})".format(
                            name,
                            attribute,
                            result["key"][2],
                            this_result["best_rmsd"],
                            this_result["mean_rmsd"],
                            this_result["median_rmsd"],
                        )
                    )
                else:
                    fit_result[result["key"][2]] = this_result
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


def _num_args_from_by_name(by_name):
    num_args = dict()
    for key, value in by_name.items():
        if "args" in value:
            num_args[key] = len(value["args"][0])
    return num_args


class AnalyticModel:
    """
    Parameter-aware analytic energy/data size/... model.

    Supports both static and parameter-based model attributes, and automatic detection of parameter-dependence.

    These provide measurements aggregated by (function/state/...) name
    and (for by_param) parameter values. Layout:
    dictionary with one key per name ('send', 'TX', ...) or
    one key per name and parameter combination
    (('send', (1, 2)), ('send', (2, 3)), ('TX', (1, 2)), ('TX', (2, 3)), ...).

    Parameter values must be ordered corresponding to the lexically sorted parameter names.

    Each element is in turn a dict with the following elements:
    - param: list of parameter values in each measurement (-> list of lists)
    - attributes: list of keys that should be analyzed,
        e.g. ['power', 'duration']
    - for each attribute mentioned in 'attributes': A list with measurements.
      All list except for 'attributes' must have the same length.

    For example:
    parameters = ['foo_count', 'irrelevant']
    by_name = {
        'foo' : [1, 1, 2],
        'bar' : [5, 6, 7],
        'attributes' : ['foo', 'bar'],
        'param' : [[1, 0], [1, 0], [2, 0]]
    }

    methods:
    get_static -- return static (parameter-unaware) model.
    get_param_lut -- return parameter-aware look-up-table model. Cannot model parameter combinations not present in by_param.
    get_fitted -- return parameter-aware model using fitted functions for behaviour prediction.

    variables:
    names -- function/state/... names (i.e., the keys of by_name)
    parameters -- parameter names
    stats -- ParamStats object providing parameter-dependency statistics for each name and attribute
    assess -- calculate model quality
    """

    def __init__(
        self,
        by_name,
        parameters,
        arg_count=None,
        function_override=dict(),
        use_corrcoef=False,
    ):
        """
        Create a new AnalyticModel and compute parameter statistics.

        :param by_name: measurements aggregated by (function/state/...) name.
            Layout: dictionary with one key per name ('send', 'TX', ...) or
            one key per name and parameter combination
            (('send', (1, 2)), ('send', (2, 3)), ('TX', (1, 2)), ('TX', (2, 3)), ...).

            Parameter values must be ordered corresponding to the lexically sorted parameter names.

            Each element is in turn a dict with the following elements:
            - param: list of parameter values in each measurement (-> list of lists)
            - attributes: list of keys that should be analyzed,
                e.g. ['power', 'duration']
            - for each attribute mentioned in 'attributes': A list with measurements.
            All list except for 'attributes' must have the same length.

            For example:
            parameters = ['foo_count', 'irrelevant']
            by_name = {
                'foo' : [1, 1, 2],
                'duration' : [5, 6, 7],
                'attributes' : ['foo', 'duration'],
                'param' : [[1, 0], [1, 0], [2, 0]]
                # foo_count-^  ^-irrelevant
            }
        :param parameters: List of parameter names
        :param function_override: dict of overrides for automatic parameter function generation.
            If (state or transition name, model attribute) is present in function_override,
            the corresponding text string is the function used for analytic (parameter-aware/fitted)
            modeling of this attribute. It is passed to AnalyticFunction, see
            there for the required format. Note that this happens regardless of
            parameter dependency detection: The provided analytic function will be assigned
            even if it seems like the model attribute is static / parameter-independent.
        :param use_corrcoef: use correlation coefficient instead of stddev comparison to detect whether a model attribute depends on a parameter
        """
        self.cache = dict()
        self.by_name = by_name
        self.by_param = by_name_to_by_param(by_name)
        self.names = sorted(by_name.keys())
        self.parameters = sorted(parameters)
        self.function_override = function_override.copy()
        self._use_corrcoef = use_corrcoef
        self._num_args = arg_count
        if self._num_args is None:
            self._num_args = _num_args_from_by_name(by_name)

        self.stats = ParamStats(
            self.by_name,
            self.by_param,
            self.parameters,
            self._num_args,
            use_corrcoef=use_corrcoef,
        )

    def _get_model_from_dict(self, model_dict, model_function):
        model = {}
        for name, elem in model_dict.items():
            model[name] = {}
            for key in elem["attributes"]:
                try:
                    model[name][key] = model_function(elem[key])
                except RuntimeWarning:
                    logger.warning("Got no data for {} {}".format(name, key))
                except FloatingPointError as fpe:
                    logger.warning("Got no data for {} {}: {}".format(name, key, fpe))
        return model

    def param_index(self, param_name):
        if param_name in self.parameters:
            return self.parameters.index(param_name)
        return len(self.parameters) + int(param_name)

    def param_name(self, param_index):
        if param_index < len(self.parameters):
            return self.parameters[param_index]
        return str(param_index)

    def get_static(self, use_mean=False):
        """
        Get static model function: name, attribute -> model value.

        Uses the median of by_name for modeling.
        """
        getter_function = np.median

        if use_mean:
            getter_function = np.mean

        static_model = self._get_model_from_dict(self.by_name, getter_function)

        def static_model_getter(name, key, **kwargs):
            return static_model[name][key]

        return static_model_getter

    def get_param_lut(self, fallback=False):
        """
        Get parameter-look-up-table model function: name, attribute, parameter values -> model value.

        The function can only give model values for parameter combinations
        present in by_param. By default, it raises KeyError for other values.

        arguments:
        fallback -- Fall back to the (non-parameter-aware) static model when encountering unknown parameter values
        """
        static_model = self._get_model_from_dict(self.by_name, np.median)
        lut_model = self._get_model_from_dict(self.by_param, np.median)

        def lut_median_getter(name, key, param, arg=[], **kwargs):
            param.extend(map(soft_cast_int, arg))
            try:
                return lut_model[(name, tuple(param))][key]
            except KeyError:
                if fallback:
                    return static_model[name][key]
                raise

        return lut_median_getter

    def get_fitted(self, safe_functions_enabled=False):
        """
        Get paramete-aware model function and model information function.

        Returns two functions:
        model_function(name, attribute, param=parameter values) -> model value.
        model_info(name, attribute) -> {'fit_result' : ..., 'function' : ... } or None
        """
        if "fitted_model_getter" in self.cache and "fitted_info_getter" in self.cache:
            return self.cache["fitted_model_getter"], self.cache["fitted_info_getter"]

        static_model = self._get_model_from_dict(self.by_name, np.median)
        param_model = dict([[name, {}] for name in self.by_name.keys()])
        paramfit = ParallelParamFit(self.by_param)

        for name in self.by_name.keys():
            for attribute in self.by_name[name]["attributes"]:
                for param_index, param in enumerate(self.parameters):
                    if self.stats.depends_on_param(name, attribute, param):
                        paramfit.enqueue(name, attribute, param_index, param, False)
                if arg_support_enabled and name in self._num_args:
                    for arg_index in range(self._num_args[name]):
                        if self.stats.depends_on_arg(name, attribute, arg_index):
                            paramfit.enqueue(
                                name,
                                attribute,
                                len(self.parameters) + arg_index,
                                arg_index,
                                False,
                            )

        paramfit.fit()

        for name in self.by_name.keys():
            num_args = 0
            if name in self._num_args:
                num_args = self._num_args[name]
            for attribute in self.by_name[name]["attributes"]:
                fit_result = paramfit.get_result(name, attribute)

                if (name, attribute) in self.function_override:
                    function_str = self.function_override[(name, attribute)]
                    x = AnalyticFunction(function_str, self.parameters, num_args)
                    x.fit(self.by_param, name, attribute)
                    if x.fit_success:
                        param_model[name][attribute] = {
                            "fit_result": fit_result,
                            "function": x,
                        }
                elif len(fit_result.keys()):
                    x = analytic.function_powerset(
                        fit_result, self.parameters, num_args
                    )
                    x.fit(self.by_param, name, attribute)

                    if x.fit_success:
                        param_model[name][attribute] = {
                            "fit_result": fit_result,
                            "function": x,
                        }

        def model_getter(name, key, **kwargs):
            if "arg" in kwargs and "param" in kwargs:
                kwargs["param"].extend(map(soft_cast_int, kwargs["arg"]))
            if key in param_model[name]:
                param_list = kwargs["param"]
                param_function = param_model[name][key]["function"]
                if param_function.is_predictable(param_list):
                    return param_function.eval(param_list)
            return static_model[name][key]

        def info_getter(name, key):
            if key in param_model[name]:
                return param_model[name][key]
            return None

        self.cache["fitted_model_getter"] = model_getter
        self.cache["fitted_info_getter"] = info_getter

        return model_getter, info_getter

    def assess(self, model_function):
        """
        Calculate MAE, SMAPE, etc. of model_function for each by_name entry.

        state/transition/... name and parameter values are fed into model_function.
        The by_name entries of this AnalyticModel are used as ground truth and
        compared with the values predicted by model_function.

        For proper model assessments, the data used to generate model_function
        and the data fed into this AnalyticModel instance must be mutually
        exclusive (e.g. by performing cross validation). Otherwise,
        overfitting cannot be detected.
        """
        detailed_results = {}
        for name, elem in sorted(self.by_name.items()):
            detailed_results[name] = {}
            for attribute in elem["attributes"]:
                predicted_data = np.array(
                    list(
                        map(
                            lambda i: model_function(
                                name, attribute, param=elem["param"][i]
                            ),
                            range(len(elem[attribute])),
                        )
                    )
                )
                measures = regression_measures(predicted_data, elem[attribute])
                detailed_results[name][attribute] = measures

        return {"by_name": detailed_results}

    def to_json(self):
        # TODO
        pass


class PTAModel:
    """
    Parameter-aware PTA-based energy model.

    Supports both static and parameter-based model attributes, and automatic detection of parameter-dependence.

    The model heavily relies on two internal data structures:
    PTAModel.by_name and PTAModel.by_param.

    These provide measurements aggregated by state/transition name
    and (for by_param) parameter values. Layout:
    dictionary with one key per state/transition ('send', 'TX', ...) or
    one key per state/transition and parameter combination
    (('send', (1, 2)), ('send', (2, 3)), ('TX', (1, 2)), ('TX', (2, 3)), ...).
    For by_param, parameter values are ordered corresponding to the lexically sorted parameter names.

    Each element is in turn a dict with the following elements:
    - isa: 'state' or 'transition'
    - power: list of mean power measurements in µW
    - duration: list of durations in µs
    - power_std: list of stddev of power per state/transition
    - energy: consumed energy (power*duration) in pJ
    - paramkeys: list of parameter names in each measurement (-> list of lists)
    - param: list of parameter values in each measurement (-> list of lists)
    - attributes: list of keys that should be analyzed,
        e.g. ['power', 'duration']
    additionally, only if isa == 'transition':
    - timeout: list of duration of previous state in µs
    - rel_energy_prev: transition energy relative to previous state mean power in pJ
    - rel_energy_next: transition energy relative to next state mean power in pJ
    - rel_power_prev: transition power relative to previous state mean power in µW
    - rel_power_next: transition power relative to next state mean power in µW
    """

    def __init__(
        self,
        by_name,
        parameters,
        arg_count,
        traces=[],
        ignore_trace_indexes=[],
        function_override={},
        use_corrcoef=False,
        pta=None,
        pelt=None,
    ):
        """
        Prepare a new PTA energy model.

        Actual model generation is done on-demand by calling the respective functions.

        arguments:
        by_name -- state/transition measurements aggregated by name, as returned by pta_trace_to_aggregate.
        parameters -- list of parameter names, as returned by pta_trace_to_aggregate
        arg_count -- function arguments, as returned by pta_trace_to_aggregate
        traces -- list of preprocessed DFA traces, as returned by RawData.get_preprocessed_data()
        ignore_trace_indexes -- list of trace indexes. The corresponding traces will be ignored.
        function_override -- dict of overrides for automatic parameter function generation.
            If (state or transition name, model attribute) is present in function_override,
            the corresponding text string is the function used for analytic (parameter-aware/fitted)
            modeling of this attribute. It is passed to AnalyticFunction, see
            there for the required format. Note that this happens regardless of
            parameter dependency detection: The provided analytic function will be assigned
            even if it seems like the model attribute is static / parameter-independent.
        use_corrcoef -- use correlation coefficient instead of stddev comparison
            to detect whether a model attribute depends on a parameter
        pta -- hardware model as `PTA` object
        pelt -- perform sub-state detection via PELT and model sub-states as well. Requires traces to be set.
        """
        self.by_name = by_name
        self.by_param = by_name_to_by_param(by_name)
        self._parameter_names = sorted(parameters)
        self._num_args = arg_count
        self._use_corrcoef = use_corrcoef
        self.traces = traces
        if traces is not None and pelt is not None:
            from .pelt import PELT

            self.pelt = PELT(**pelt)
            self.find_substates()
        else:
            self.pelt = None
        self.stats = ParamStats(
            self.by_name,
            self.by_param,
            self._parameter_names,
            self._num_args,
            self._use_corrcoef,
        )
        self.cache = {}
        np.seterr("raise")
        self.function_override = function_override.copy()
        self.pta = pta
        self.ignore_trace_indexes = ignore_trace_indexes
        self._aggregate_to_ndarray(self.by_name)

    def _aggregate_to_ndarray(self, aggregate):
        for elem in aggregate.values():
            for key in elem["attributes"]:
                elem[key] = np.array(elem[key])

    # This heuristic is very similar to the "function is not much better than
    # median" checks in get_fitted. So far, doing it here as well is mostly
    # a performance and not an algorithm quality decision.
    # --df, 2018-04-18
    def depends_on_param(self, state_or_trans, key, param):
        return self.stats.depends_on_param(state_or_trans, key, param)

    # See notes on depends_on_param
    def depends_on_arg(self, state_or_trans, key, param):
        return self.stats.depends_on_arg(state_or_trans, key, param)

    def _get_model_from_dict(self, model_dict, model_function):
        model = {}
        for name, elem in model_dict.items():
            model[name] = {}
            for key in elem["attributes"]:
                try:
                    model[name][key] = model_function(elem[key])
                except RuntimeWarning:
                    logger.warning("Got no data for {} {}".format(name, key))
                except FloatingPointError as fpe:
                    logger.warning("Got no data for {} {}: {}".format(name, key, fpe))
        return model

    def get_static(self, use_mean=False):
        """
        Get static model function: name, attribute -> model value.

        Uses the median of by_name for modeling, unless `use_mean` is set.
        """
        getter_function = np.median

        if use_mean:
            getter_function = np.mean

        static_model = self._get_model_from_dict(self.by_name, getter_function)

        def static_model_getter(name, key, **kwargs):
            return static_model[name][key]

        return static_model_getter

    def get_param_lut(self, fallback=False):
        """
        Get parameter-look-up-table model function: name, attribute, parameter values -> model value.

        The function can only give model values for parameter combinations
        present in by_param. By default, it raises KeyError for other values.

        arguments:
        fallback -- Fall back to the (non-parameter-aware) static model when encountering unknown parameter values
        """
        static_model = self._get_model_from_dict(self.by_name, np.median)
        lut_model = self._get_model_from_dict(self.by_param, np.median)

        def lut_median_getter(name, key, param, arg=[], **kwargs):
            param.extend(map(soft_cast_int, arg))
            try:
                return lut_model[(name, tuple(param))][key]
            except KeyError:
                if fallback:
                    return static_model[name][key]
                raise

        return lut_median_getter

    def param_index(self, param_name):
        if param_name in self._parameter_names:
            return self._parameter_names.index(param_name)
        return len(self._parameter_names) + int(param_name)

    def param_name(self, param_index):
        if param_index < len(self._parameter_names):
            return self._parameter_names[param_index]
        return str(param_index)

    def get_fitted(self, safe_functions_enabled=False):
        """
        Get parameter-aware model function and model information function.

        Returns two functions:
        model_function(name, attribute, param=parameter values) -> model value.
        model_info(name, attribute) -> {'fit_result' : ..., 'function' : ... } or None
        """
        if "fitted_model_getter" in self.cache and "fitted_info_getter" in self.cache:
            return self.cache["fitted_model_getter"], self.cache["fitted_info_getter"]

        static_model = self._get_model_from_dict(self.by_name, np.median)
        param_model = dict(
            [[state_or_tran, {}] for state_or_tran in self.by_name.keys()]
        )
        paramfit = ParallelParamFit(self.by_param)
        for state_or_tran in self.by_name.keys():
            for model_attribute in self.by_name[state_or_tran]["attributes"]:
                fit_results = {}
                for parameter_index, parameter_name in enumerate(self._parameter_names):
                    if self.depends_on_param(
                        state_or_tran, model_attribute, parameter_name
                    ):
                        paramfit.enqueue(
                            state_or_tran,
                            model_attribute,
                            parameter_index,
                            parameter_name,
                            safe_functions_enabled,
                        )
                if (
                    arg_support_enabled
                    and self.by_name[state_or_tran]["isa"] == "transition"
                ):
                    for arg_index in range(self._num_args[state_or_tran]):
                        if self.depends_on_arg(
                            state_or_tran, model_attribute, arg_index
                        ):
                            paramfit.enqueue(
                                state_or_tran,
                                model_attribute,
                                len(self._parameter_names) + arg_index,
                                arg_index,
                                safe_functions_enabled,
                            )
        paramfit.fit()

        for state_or_tran in self.by_name.keys():
            num_args = 0
            if (
                arg_support_enabled
                and self.by_name[state_or_tran]["isa"] == "transition"
            ):
                num_args = self._num_args[state_or_tran]
            for model_attribute in self.by_name[state_or_tran]["attributes"]:
                fit_results = paramfit.get_result(state_or_tran, model_attribute)

                if (state_or_tran, model_attribute) in self.function_override:
                    function_str = self.function_override[
                        (state_or_tran, model_attribute)
                    ]
                    x = AnalyticFunction(function_str, self._parameter_names, num_args)
                    x.fit(self.by_param, state_or_tran, model_attribute)
                    if x.fit_success:
                        param_model[state_or_tran][model_attribute] = {
                            "fit_result": fit_results,
                            "function": x,
                        }
                elif len(fit_results.keys()):
                    x = analytic.function_powerset(
                        fit_results, self._parameter_names, num_args
                    )
                    x.fit(self.by_param, state_or_tran, model_attribute)
                    if x.fit_success:
                        param_model[state_or_tran][model_attribute] = {
                            "fit_result": fit_results,
                            "function": x,
                        }

        def model_getter(name, key, **kwargs):
            if "arg" in kwargs and "param" in kwargs:
                kwargs["param"].extend(map(soft_cast_int, kwargs["arg"]))
            if key in param_model[name]:
                param_list = kwargs["param"]
                param_function = param_model[name][key]["function"]
                if param_function.is_predictable(param_list):
                    return param_function.eval(param_list)
            return static_model[name][key]

        def info_getter(name, key):
            if key in param_model[name]:
                return param_model[name][key]
            return None

        self.cache["fitted_model_getter"] = model_getter
        self.cache["fitted_info_getter"] = info_getter

        return model_getter, info_getter

    def pelt_refine(self, by_param_key):
        logger.debug(f"PELT: {by_param_key} needs refinement")

        penalty_by_trace = list()
        changepoints_by_penalty_by_trace = list()
        num_changepoints_by_trace = list()
        changepoints_by_trace = list()

        for power_values in self.by_param[by_param_key]["power_traces"]:
            penalty, changepoints_by_penalty = self.pelt.get_penalty_and_changepoints(
                power_values
            )
            penalty_by_trace.append(penalty)
            changepoints_by_penalty_by_trace.append(changepoints_by_penalty)
            num_changepoints_by_trace.append(len(changepoints_by_penalty[penalty]))
            changepoints_by_trace.append(changepoints_by_penalty[penalty])

        if np.median(num_changepoints_by_trace) < 1:
            logger.debug(
                f"    we found no changepoints {num_changepoints_by_trace} with penalties {penalty_by_trace}"
            )
            substate_counts = [1 for i in self.by_param[by_param_key]["param"]]
            substate_data = {
                "duration": self.by_param[by_param_key]["duration"],
                "power": self.by_param[by_param_key]["power"],
                "power_std": self.by_param[by_param_key]["power_std"],
            }
            return (substate_counts, substate_data)

        num_changepoints = np.argmax(np.bincount(num_changepoints_by_trace))

        logger.debug(
            f"    we found {num_changepoints} changepoints {num_changepoints_by_trace} with penalties {penalty_by_trace}"
        )
        return self.pelt.calc_raw_states(
            self.by_param[by_param_key]["timestamps"],
            self.by_param[by_param_key]["power_traces"],
            changepoints_by_trace,
            num_changepoints,
        )

    def find_substates(self):
        """
        Finds substates via PELT and adds substate_count to by_name and by_param.
        """
        states = self.states()
        substates_by_param = dict()
        for k in self.by_param.keys():
            if (
                self.pelt.name_filter is None or k[0] == self.pelt.name_filter
            ) and self.pelt.needs_refinement(self.by_param[k]["power_traces"]):
                substates_by_param[k] = self.pelt_refine(k)
            else:
                substate_counts = [1 for i in self.by_param[k]["param"]]
                substate_data = {
                    "duration": self.by_param[k]["duration"],
                    "power": self.by_param[k]["power"],
                    "power_std": self.by_param[k]["power_std"],
                }
                substates_by_param[k] = (substate_counts, substate_data)

        # suitable for AEMR modeling
        sc_by_param = dict()
        for param_key, (substate_counts, _) in substates_by_param.items():
            # do not append "substate_count" to "attributes" here.
            # by_param[(foo, *)]["attributes"] is the same object as by_name[foo]["attributes"]
            self.by_param[param_key]["substate_count"] = substate_counts

        for state_name in states:
            param_offset = dict()
            state = self.by_name[state_name]
            state["attributes"].append("substate_count")
            state["substate_count"] = list()
            for i, param in enumerate(state["param"]):
                param = tuple(param)
                if param not in param_offset:
                    param_offset[param] = 0
                state["substate_count"].append(
                    self.by_param[(state_name, param)]["substate_count"][
                        param_offset[param]
                    ]
                )
                param_offset[param] += 1

    def get_substates(self):
        states = self.states()

        substates_by_param = dict()
        for k in self.by_param.keys():
            if k[0] in states:
                state_name = k[0]
                if self.pelt.needs_refinement(self.by_param[k]["power_traces"]):
                    substates_by_param[k] = self.pelt_refine(k)
                else:
                    substate_counts = [1 for i in self.by_param[k]["param"]]
                    substate_data = {
                        "duration": self.by_param[k]["duration"],
                        "power": self.by_param[k]["power"],
                        "power_std": self.by_param[k]["power_std"],
                    }
                    substates_by_param[k] = (substate_counts, substate_data)

        # suitable for AEMR modeling
        sc_by_param = dict()
        for param_key, (substate_counts, _) in substates_by_param.items():
            sc_by_param[param_key] = {
                "attributes": ["substate_count"],
                "isa": "state",
                "substate_count": substate_counts,
                "param": self.by_param[param_key]["param"],
            }

        sc_by_name = by_param_to_by_name(sc_by_param)
        self.sc_by_name = sc_by_name
        self.sc_by_param = sc_by_param
        static_model = self._get_model_from_dict(self.sc_by_name, np.median)

        def static_model_getter(name, key, **kwargs):
            return static_model[name][key]

        return static_model_getter

        """
        for k in self.by_param.keys():
            if k[0] in states:
                state_name = k[0]
                if state_name not in pelt_by_name:
                    pelt_by_name[state_name] = dict()
                if self.pelt.needs_refinement(self.by_param[k]["power_traces"]):
                    res = self.pelt_refine(k)
                    for substate_index, substate in enumerate(res):
                        if substate_index not in pelt_by_name[state_name]:
                            pelt_by_name[state_name][substate_index] = {
                                "attribute": ["power", "duration"],
                                "isa": "state",
                                "param": list(),
                                "power": list(),
                                "duration": list()
                            }
                        pelt_by_name[state_name][substate_index]["param"].extend(self.by_param[k]["param"][:len(substate["power"])])
                        pelt_by_name[state_name][substate_index]["power"].extend(substate["power"])
                        pelt_by_name[state_name][substate_index]["duration"].extend(substate["duration"])
        print(pelt_by_name)
        """

        return None, None

    def to_json(self):
        static_model = self.get_static()
        static_quality = self.assess(static_model)
        param_model, param_info = self.get_fitted()
        analytic_quality = self.assess(param_model)
        pta = self.pta
        if pta is None:
            pta = PTA(self.states(), parameters=self._parameter_names)
        pta.update(
            static_model,
            param_info,
            static_error=static_quality["by_name"],
            analytic_error=analytic_quality["by_name"],
        )
        return pta.to_json()

    def states(self):
        """Return sorted list of state names."""
        return sorted(
            list(
                filter(lambda k: self.by_name[k]["isa"] == "state", self.by_name.keys())
            )
        )

    def transitions(self):
        """Return sorted list of transition names."""
        return sorted(
            list(
                filter(
                    lambda k: self.by_name[k]["isa"] == "transition",
                    self.by_name.keys(),
                )
            )
        )

    def states_and_transitions(self):
        """Return list of states and transition names."""
        ret = self.states()
        ret.extend(self.transitions())
        return ret

    def parameters(self):
        return self._parameter_names

    def attributes(self, state_or_trans):
        return self.by_name[state_or_trans]["attributes"]

    def assess(self, model_function, ref=None):
        """
        Calculate MAE, SMAPE, etc. of model_function for each by_name entry.

        state/transition/... name and parameter values are fed into model_function.
        The by_name entries of this PTAModel are used as ground truth and
        compared with the values predicted by model_function.

        For proper model assessments, the data used to generate model_function
        and the data fed into this AnalyticModel instance must be mutually
        exclusive (e.g. by performing cross validation). Otherwise,
        overfitting cannot be detected.
        """
        detailed_results = {}
        if ref is None:
            ref = self.by_name
        for name, elem in sorted(ref.items()):
            detailed_results[name] = {}
            for key in elem["attributes"]:
                predicted_data = np.array(
                    list(
                        map(
                            lambda i: model_function(name, key, param=elem["param"][i]),
                            range(len(elem[key])),
                        )
                    )
                )
                measures = regression_measures(predicted_data, elem[key])
                detailed_results[name][key] = measures
            if elem["isa"] == "transition":
                predicted_data = np.array(
                    list(
                        map(
                            lambda i: model_function(
                                name, "power", param=elem["param"][i]
                            )
                            * model_function(name, "duration", param=elem["param"][i]),
                            range(len(elem["power"])),
                        )
                    )
                )
                measures = regression_measures(
                    predicted_data, elem["power"] * elem["duration"]
                )
                detailed_results[name]["energy_Pt"] = measures

        return {"by_name": detailed_results}

    def assess_states(
        self, model_function, model_attribute="power", distribution: dict = None
    ):
        """
        Calculate overall model error assuming equal distribution of states
        """
        # TODO calculate mean power draw for distribution and use it to
        # calculate relative error from MAE combination
        model_quality = self.assess(model_function)
        num_states = len(self.states())
        if distribution is None:
            distribution = dict(map(lambda x: [x, 1 / num_states], self.states()))

        if not np.isclose(sum(distribution.values()), 1):
            raise ValueError(
                "distribution must be a probability distribution with sum 1"
            )

        # total_value = None
        # try:
        #     total_value = sum(map(lambda x: model_function(x, model_attribute) * distribution[x], self.states()))
        # except KeyError:
        #     pass

        total_error = np.sqrt(
            sum(
                map(
                    lambda x: np.square(
                        model_quality["by_name"][x][model_attribute]["mae"]
                        * distribution[x]
                    ),
                    self.states(),
                )
            )
        )
        return total_error

    def assess_on_traces(self, model_function):
        """
        Calculate MAE, SMAPE, etc. of model_function for each trace known to this PTAModel instance.

        :returns: dict of `duration_by_trace`, `energy_by_trace`, `timeout_by_trace`, `rel_energy_by_trace` and `state_energy_by_trace`.
            Each entry holds regression measures for the corresponding measure. Note that the determined model quality heavily depends on the
            traces: small-ish absolute errors in states which frequently occur may have more effect than large absolute errors in rarely occuring states
        """
        model_energy_list = []
        real_energy_list = []
        model_rel_energy_list = []
        model_state_energy_list = []
        model_duration_list = []
        real_duration_list = []
        model_timeout_list = []
        real_timeout_list = []

        for trace in self.traces:
            if trace["id"] not in self.ignore_trace_indexes:
                for rep_id in range(len(trace["trace"][0]["offline"])):
                    model_energy = 0.0
                    real_energy = 0.0
                    model_rel_energy = 0.0
                    model_state_energy = 0.0
                    model_duration = 0.0
                    real_duration = 0.0
                    model_timeout = 0.0
                    real_timeout = 0.0
                    for i, trace_part in enumerate(trace["trace"]):
                        name = trace_part["name"]
                        prev_name = trace["trace"][i - 1]["name"]
                        isa = trace_part["isa"]
                        if name != "UNINITIALIZED":
                            try:
                                param = trace_part["offline_aggregates"]["param"][
                                    rep_id
                                ]
                                prev_param = trace["trace"][i - 1][
                                    "offline_aggregates"
                                ]["param"][rep_id]
                                power = trace_part["offline"][rep_id]["uW_mean"]
                                duration = trace_part["offline"][rep_id]["us"]
                                prev_duration = trace["trace"][i - 1]["offline"][
                                    rep_id
                                ]["us"]
                                real_energy += power * duration
                                if isa == "state":
                                    model_energy += (
                                        model_function(name, "power", param=param)
                                        * duration
                                    )
                                else:
                                    model_energy += model_function(
                                        name, "energy", param=param
                                    )
                                    # If i == 1, the previous state was UNINITIALIZED, for which we do not have model data
                                    if i == 1:
                                        model_rel_energy += model_function(
                                            name, "energy", param=param
                                        )
                                    else:
                                        model_rel_energy += model_function(
                                            prev_name, "power", param=prev_param
                                        ) * (prev_duration + duration)
                                        model_state_energy += model_function(
                                            prev_name, "power", param=prev_param
                                        ) * (prev_duration + duration)
                                    model_rel_energy += model_function(
                                        name, "rel_energy_prev", param=param
                                    )
                                    real_duration += duration
                                    model_duration += model_function(
                                        name, "duration", param=param
                                    )
                                    if (
                                        "plan" in trace_part
                                        and trace_part["plan"]["level"] == "epilogue"
                                    ):
                                        real_timeout += trace_part["offline"][rep_id][
                                            "timeout"
                                        ]
                                        model_timeout += model_function(
                                            name, "timeout", param=param
                                        )
                            except KeyError:
                                # if states/transitions have been removed via --filter-param, this is harmless
                                pass
                    real_energy_list.append(real_energy)
                    model_energy_list.append(model_energy)
                    model_rel_energy_list.append(model_rel_energy)
                    model_state_energy_list.append(model_state_energy)
                    real_duration_list.append(real_duration)
                    model_duration_list.append(model_duration)
                    real_timeout_list.append(real_timeout)
                    model_timeout_list.append(model_timeout)

        return {
            "duration_by_trace": regression_measures(
                np.array(model_duration_list), np.array(real_duration_list)
            ),
            "energy_by_trace": regression_measures(
                np.array(model_energy_list), np.array(real_energy_list)
            ),
            "timeout_by_trace": regression_measures(
                np.array(model_timeout_list), np.array(real_timeout_list)
            ),
            "rel_energy_by_trace": regression_measures(
                np.array(model_rel_energy_list), np.array(real_energy_list)
            ),
            "state_energy_by_trace": regression_measures(
                np.array(model_state_energy_list), np.array(real_energy_list)
            ),
        }
