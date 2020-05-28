"""
Utilities for analytic description of parameter-dependent model attributes.

This module provides classes and helper functions useful for least-squares
regression and general handling of model functions.
"""
from itertools import chain, combinations
import numpy as np
import re
from scipy import optimize
from .utils import is_numeric, vprint

arg_support_enabled = True


def powerset(iterable):
    """
    Return powerset of `iterable` elements.

    Example: `powerset([1, 2])` -> `[(), (1), (2), (1, 2)]`
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class ParamFunction:
    """
    A one-dimensional model function, ready for least squares optimization and similar.

    Supports validity checks (e.g. if it is undefined for x <= 0) and an
    error measure.
    """

    def __init__(self, param_function, validation_function, num_vars):
        """
        Create function object suitable for regression analysis.

        This documentation assumes that 1-dimensional functions
        (-> single float as model input) are used. However, n-dimensional
        functions (-> list of float as model input) are also supported.

        :param param_function: regression function (reg_param, model_param) -> float.
            reg_param is a list of regression variable values,
            model_param is the model input value (float).
            Example: `lambda rp, mp: rp[0] + rp[1] * mp`
        :param validation_function: function used to check whether param_function
            is defined for a given model_param. Signature:
            model_param -> bool
            Example: `lambda mp: mp > 0`
        :param num_vars: How many regression variables are used by this function,
            i.e., the length of param_function's reg_param argument.
        """
        self._param_function = param_function
        self._validation_function = validation_function
        self._num_variables = num_vars

    def is_valid(self, arg: float) -> bool:
        """
        Check whether the regression function is defined for the given argument.

        :param arg: argument (e.g. model parameter) to check for
        :returns: True iff the function is defined for `arg`
        """
        return self._validation_function(arg)

    def eval(self, param: list, arg: float) -> float:
        """
        Evaluate regression function.

        :param param: regression variable values (list of float)
        :param arg: model input (float)
        :returns: regression function output (float)
        """
        return self._param_function(param, arg)

    def error_function(self, P: list, X: float, y: float) -> float:
        """
        Calculate model error.

        :param P: regression variables as returned by optimization (list of float)
        :param X: model input (float)
        :param y: expected model output / ground truth for model input (float)
        :returns: Deviation between model output and ground truth (float)
        """
        return self._param_function(P, X) - y


class NormalizationFunction:
    """
    Wrapper for parameter normalization functions used in YAML PTA/DFA models.
    """

    def __init__(self, function_str: str):
        """
        Create a new normalization function from `function_str`.

        :param function_str: Function string. Must use the single argument
        `param` and return a float.
        """
        self._function_str = function_str
        self._function = eval("lambda param: " + function_str)

    def eval(self, param_value: float) -> float:
        """
        Evaluate the normalization function and return its output.

        :param param_value: Parameter value
        """
        return self._function(param_value)


class AnalyticFunction:
    """
    A multi-dimensional model function, generated from a string, which can be optimized using regression.

    The function describes a single model attribute (e.g. TX duration or send(...) energy)
    and how it is influenced by model parameters such as configured bit rate or
    packet length.
    """

    def __init__(
        self, function_str, parameters, num_args, verbose=True, regression_args=None
    ):
        """
        Create a new AnalyticFunction object from a function string.

        :param function_str: the function.
            Refer to regression variables using regression_arg(123),
            to parameters using parameter(name),
            and to function arguments (if any) using function_arg(123).
            Example: "regression_arg(0) + regression_arg(1) * parameter(txbytes)"
        :param parameters: list containing the names of all model parameters,
            including those not used in function_str, sorted lexically.
            Sorting is mandatory, as parameter indexes (and not names) are used internally.
        :param num_args: number of local function arguments, if any. Set to 0 if
            the model attribute does not belong to a function or if function
            arguments are not included in the model.
        :param verbose: complain about odd events
        :param regression_args: Initial regression variable values,
            both for function usage and least squares optimization.
            If unset, defaults to [1, 1, 1, ...]
        """
        self._parameter_names = parameters
        self._num_args = num_args
        self._model_str = function_str
        rawfunction = function_str
        self._dependson = [False] * (len(parameters) + num_args)
        self.fit_success = False
        self.verbose = verbose

        if type(function_str) == str:
            num_vars_re = re.compile(r"regression_arg\(([0-9]+)\)")
            num_vars = max(map(int, num_vars_re.findall(function_str))) + 1
            for i in range(len(parameters)):
                if rawfunction.find("parameter({})".format(parameters[i])) >= 0:
                    self._dependson[i] = True
                    rawfunction = rawfunction.replace(
                        "parameter({})".format(parameters[i]),
                        "model_param[{:d}]".format(i),
                    )
            for i in range(0, num_args):
                if rawfunction.find("function_arg({:d})".format(i)) >= 0:
                    self._dependson[len(parameters) + i] = True
                    rawfunction = rawfunction.replace(
                        "function_arg({:d})".format(i),
                        "model_param[{:d}]".format(len(parameters) + i),
                    )
            for i in range(num_vars):
                rawfunction = rawfunction.replace(
                    "regression_arg({:d})".format(i), "reg_param[{:d}]".format(i)
                )
            self._function_str = rawfunction
            self._function = eval("lambda reg_param, model_param: " + rawfunction)
        else:
            self._function_str = "raise ValueError"
            self._function = function_str

        if regression_args:
            self._regression_args = regression_args.copy()
            self._fit_success = True
        elif type(function_str) == str:
            self._regression_args = list(np.ones((num_vars)))
        else:
            self._regression_args = []

    def get_fit_data(self, by_param, state_or_tran, model_attribute):
        """
        Return training data suitable for scipy.optimize.least_squares.

        :param by_param: measurement data, partitioned by state/transition name and parameter/arg values.
            This function only uses by_param[(state_or_tran, *)][model_attribute],
            which must be a list or 1-D NumPy array containing the ground truth.
            The parameter values in (state_or_tran, *) must be numeric for
            all parameters this function depends on -- otherwise, the
            corresponding data will be left out. Parameter values must be
            ordered according to the order of parameter names used in
            the ParamFunction constructor. Argument values (if any) always come after
            parameters, in the order of their index in the function signature.
        :param state_or_tran: state or transition name, e.g. "TX" or "send"
        :param model_attribute: model attribute name, e.g. "power" or "duration"

        :return: (X, Y, num_valid, num_total):
            X -- 2-D NumPy array of parameter combinations (model input).
                First dimension is the parameter/argument index, the second
                dimension contains its values.
                Example: X[0] contains the first parameter's values.
            Y -- 1-D NumPy array of training data (desired model output).
            num_valid -- amount of distinct parameter values suitable for optimization
            num_total -- total amount of distinct parameter values
        """
        dimension = len(self._parameter_names) + self._num_args
        X = [[] for i in range(dimension)]
        Y = []

        num_valid = 0
        num_total = 0

        for key, val in by_param.items():
            if key[0] == state_or_tran and len(key[1]) == dimension:
                valid = True
                num_total += 1
                for i in range(dimension):
                    if self._dependson[i] and not is_numeric(key[1][i]):
                        valid = False
                if valid:
                    num_valid += 1
                    Y.extend(val[model_attribute])
                    for i in range(dimension):
                        if self._dependson[i]:
                            X[i].extend([float(key[1][i])] * len(val[model_attribute]))
                        else:
                            X[i].extend([np.nan] * len(val[model_attribute]))
            elif key[0] == state_or_tran and len(key[1]) != dimension:
                vprint(
                    self.verbose,
                    "[W] Invalid parameter key length while gathering fit data for {}/{}. is {}, want {}.".format(
                        state_or_tran, model_attribute, len(key[1]), dimension
                    ),
                )
        X = np.array(X)
        Y = np.array(Y)

        return X, Y, num_valid, num_total

    def fit(self, by_param, state_or_tran, model_attribute):
        """
        Fit the function on measurements via least squares regression.

        :param by_param: measurement data, partitioned by state/transition name and parameter/arg values
        :param state_or_tran: state or transition name, e.g. "TX" or "send"
        :param model_attribute: model attribute name, e.g. "power" or "duration"

        The ground truth is read from by_param[(state_or_tran, *)][model_attribute],
        which must be a list or 1-D NumPy array. Parameter values must be
        ordered according to the parameter names in the constructor. If
        argument values are present, they must come after parameter values
        in the order of their appearance in the function signature.
        """
        X, Y, num_valid, num_total = self.get_fit_data(
            by_param, state_or_tran, model_attribute
        )
        if num_valid > 2:
            error_function = lambda P, X, y: self._function(P, X) - y
            try:
                res = optimize.least_squares(
                    error_function, self._regression_args, args=(X, Y), xtol=2e-15
                )
            except ValueError as err:
                vprint(
                    self.verbose,
                    "[W] Fit failed for {}/{}: {} (function: {})".format(
                        state_or_tran, model_attribute, err, self._model_str
                    ),
                )
                return
            if res.status > 0:
                self._regression_args = res.x
                self.fit_success = True
            else:
                vprint(
                    self.verbose,
                    "[W] Fit failed for {}/{}: {} (function: {})".format(
                        state_or_tran, model_attribute, res.message, self._model_str
                    ),
                )
        else:
            vprint(
                self.verbose,
                "[W] Insufficient amount of valid parameter keys, cannot fit {}/{}".format(
                    state_or_tran, model_attribute
                ),
            )

    def is_predictable(self, param_list):
        """
        Return whether the model function can be evaluated on the given parameter values.

        The first value corresponds to the lexically first model parameter, etc.
        All parameters must be set, not just the ones this function depends on.

        Returns False iff a parameter the function depends on is not numeric
        (e.g. None).
        """
        for i, param in enumerate(param_list):
            if self._dependson[i] and not is_numeric(param):
                return False
        return True

    def eval(self, param_list, arg_list=[]):
        """
        Evaluate model function with specified param/arg values.

        :param param_list: parameter values (list of float). First item
            corresponds to lexically first parameter, etc.
        :param arg_list: argument values (list of float), if arguments are used.
        """
        if len(self._regression_args) == 0:
            return self._function(param_list, arg_list)
        return self._function(self._regression_args, param_list)


class analytic:
    """
    Utilities for analytic description of parameter-dependent model attributes and regression analysis.

    provided functions:
    functions -- retrieve pre-defined set of regression function candidates
    function_powerset -- combine several per-parameter functions into a single AnalyticFunction
    """

    _num0_8 = np.vectorize(lambda x: 8 - bin(int(x)).count("1"))
    _num0_16 = np.vectorize(lambda x: 16 - bin(int(x)).count("1"))
    _num1 = np.vectorize(lambda x: bin(int(x)).count("1"))
    _safe_log = np.vectorize(lambda x: np.log(np.abs(x)) if np.abs(x) > 0.001 else 1.0)
    _safe_inv = np.vectorize(lambda x: 1 / x if np.abs(x) > 0.001 else 1.0)
    _safe_sqrt = np.vectorize(lambda x: np.sqrt(np.abs(x)))

    _function_map = {
        "linear": lambda x: x,
        "logarithmic": np.log,
        "logarithmic1": lambda x: np.log(x + 1),
        "exponential": np.exp,
        "square": lambda x: x ** 2,
        "inverse": lambda x: 1 / x,
        "sqrt": lambda x: np.sqrt(np.abs(x)),
        "num0_8": _num0_8,
        "num0_16": _num0_16,
        "num1": _num1,
        "safe_log": lambda x: np.log(np.abs(x)) if np.abs(x) > 0.001 else 1.0,
        "safe_inv": lambda x: 1 / x if np.abs(x) > 0.001 else 1.0,
        "safe_sqrt": lambda x: np.sqrt(np.abs(x)),
    }

    @staticmethod
    def functions(safe_functions_enabled=False):
        """
        Retrieve pre-defined set of regression function candidates.

        :param safe_functions_enabled: Include "safe" variants of functions with
            limited argument range, e.g. a safe
            inverse which returns 1 when dividing by 0.
        
        Returns a dict of functions which are typical for energy/timing
        behaviour of embedded hardware, e.g. linear, exponential or inverse
        dependency on a configuration setting/runtime variable.

        Each function is a ParamFunction object. In most cases, two regression
        variables are expected.
        """
        functions = {
            "linear": ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * model_param,
                lambda model_param: True,
                2,
            ),
            "logarithmic": ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * np.log(model_param),
                lambda model_param: model_param > 0,
                2,
            ),
            "logarithmic1": ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * np.log(model_param + 1),
                lambda model_param: model_param > -1,
                2,
            ),
            "exponential": ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * np.exp(model_param),
                lambda model_param: model_param <= 64,
                2,
            ),
            #'polynomial' : lambda reg_param, model_param: reg_param[0] + reg_param[1] * model_param + reg_param[2] * model_param ** 2,
            "square": ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * model_param ** 2,
                lambda model_param: True,
                2,
            ),
            "inverse": ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] / model_param,
                lambda model_param: model_param != 0,
                2,
            ),
            "sqrt": ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * np.sqrt(model_param),
                lambda model_param: model_param >= 0,
                2,
            ),
            "num0_8": ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * analytic._num0_8(model_param),
                lambda model_param: True,
                2,
            ),
            "num0_16": ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * analytic._num0_16(model_param),
                lambda model_param: True,
                2,
            ),
            "num1": ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * analytic._num1(model_param),
                lambda model_param: True,
                2,
            ),
        }

        if safe_functions_enabled:
            functions["safe_log"] = ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * analytic._safe_log(model_param),
                lambda model_param: True,
                2,
            )
            functions["safe_inv"] = ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * analytic._safe_inv(model_param),
                lambda model_param: True,
                2,
            )
            functions["safe_sqrt"] = ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * analytic._safe_sqrt(model_param),
                lambda model_param: True,
                2,
            )

        return functions

    @staticmethod
    def _fmap(reference_type, reference_name, function_type):
        """Map arg/parameter name and best-fit function name to function text suitable for AnalyticFunction."""
        ref_str = "{}({})".format(reference_type, reference_name)
        if function_type == "linear":
            return ref_str
        if function_type == "logarithmic":
            return "np.log({})".format(ref_str)
        if function_type == "logarithmic1":
            return "np.log({} + 1)".format(ref_str)
        if function_type == "exponential":
            return "np.exp({})".format(ref_str)
        if function_type == "exponential":
            return "np.exp({})".format(ref_str)
        if function_type == "square":
            return "({})**2".format(ref_str)
        if function_type == "inverse":
            return "1/({})".format(ref_str)
        if function_type == "sqrt":
            return "np.sqrt({})".format(ref_str)
        return "analytic._{}({})".format(function_type, ref_str)

    @staticmethod
    def function_powerset(fit_results, parameter_names, num_args=0):
        """
        Combine per-parameter regression results into a single multi-dimensional function.

        :param fit_results: results dict. One element per parameter, each containing
            a dict of the form {'best' : name of function with best fit}.
            Must not include parameters which do not influence the model attribute.
            Example: {'txpower' : {'best': 'exponential'}}
        :param parameter_names: Parameter names, including those left
            out in fit_results because they do not influence the model attribute.
            Must be sorted lexically.
            Example: ['bitrate', 'txpower']
        :param num_args: number of local function arguments, if any. Set to 0 if
            the model attribute does not belong to a function or if function
            arguments are not included in the model.

        Returns an AnalyticFunction instantce corresponding to the combined
        function.
        """
        buf = "0"
        arg_idx = 0
        for combination in powerset(fit_results.items()):
            buf += " + regression_arg({:d})".format(arg_idx)
            arg_idx += 1
            for function_item in combination:
                if arg_support_enabled and is_numeric(function_item[0]):
                    buf += " * {}".format(
                        analytic._fmap(
                            "function_arg", function_item[0], function_item[1]["best"]
                        )
                    )
                else:
                    buf += " * {}".format(
                        analytic._fmap(
                            "parameter", function_item[0], function_item[1]["best"]
                        )
                    )
        return AnalyticFunction(buf, parameter_names, num_args)
