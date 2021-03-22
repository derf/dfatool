#!/usr/bin/env python3
"""
Utilities for analytic description of parameter-dependent model attributes.

This module provides classes and helper functions useful for least-squares
regression and general handling of model functions.
"""
from itertools import chain, combinations
import logging
import numpy as np
import re
from scipy import optimize
from .utils import is_numeric

logger = logging.getLogger(__name__)


def powerset(iterable):
    """
    Return powerset of `iterable` elements.

    Example: `powerset([1, 2])` -> `[(), (1), (2), (1, 2)]`
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def gplearn_to_function(function_str: str):
    """
    Convert gplearn-style function string to Python function.

    Takes a function string like "mul(add(X0, X1), X2)" and returns
    a Python function implementing the specified behaviour,
    e.g. "lambda x, y, z: (x + y) * z".

    Supported functions:
    add  --  x + y
    sub  --  x - y
    mul  --  x * y
    div  --  x / y if |y| > 0.001, otherwise 1
    sqrt --  sqrt(|x|)
    log  --  log(|x|) if |x| > 0.001, otherwise 0
    inv  --  1 / x if |x| > 0.001, otherwise 0
    """
    eval_globals = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: np.divide(x, y) if np.abs(y) > 0.001 else 1.0,
        "sqrt": lambda x: np.sqrt(np.abs(x)),
        "log": lambda x: np.log(np.abs(x)) if np.abs(x) > 0.001 else 0.0,
        "inv": lambda x: 1.0 / x if np.abs(x) > 0.001 else 0.0,
    }

    last_arg_index = 0
    for i in range(0, 100):
        if function_str.find("X{:d}".format(i)) >= 0:
            last_arg_index = i

    arg_list = []
    for i in range(0, last_arg_index + 1):
        arg_list.append("X{:d}".format(i))

    eval_str = "lambda {}, *whatever: {}".format(",".join(arg_list), function_str)
    logger.debug(eval_str)
    return eval(eval_str, eval_globals)


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


class ModelFunction:
    def __init__(self, value):
        # a model always has a static (median/mean) value. For StaticFunction, it's the only data point.
        # For more complex models, it's usede both as fallback in case the model cannot predict the current
        # parameter combination, and for use cases requiring static models
        self.value = value

        # A ModelFunction may track its own accuracy, both of the static value and of the eval() method.
        # However, it does not specify how the accuracy was calculated (e.g. which data was used and whether cross-validation was performed)
        self.value_error = None
        self.function_error = None

    def is_predictable(self, param_list):
        raise NotImplementedError

    def eval(self, param_list):
        raise NotImplementedError

    def eval_mae(self, param_list):
        if self.is_predictable(param_list):
            return self.function_error["mae"]
        return self.value_error["mae"]

    def to_json(self):
        ret = {
            "value": self.value,
            "valueError": self.value_error,
            "functionError": self.function_error,
        }
        return ret

    @classmethod
    def from_json(cls, data):
        if data["type"] == "static":
            mf = StaticFunction.from_json(data)
        elif data["type"] == "split":
            mf = SplitFunction.from_json(data)
        elif data["type"] == "analytic":
            mf = AnalyticFunction.from_json(data)
        else:
            raise ValueError("Unknown ModelFunction type: " + data["type"])

        if "valueError" in data:
            mf.value_error = data["valueError"]
        if "functionError" in data:
            mf.function_error = data["functionError"]

        return mf

    @classmethod
    def from_json_maybe(cls, json_wrapped: dict, attribute: str):
        # Legacy Code for PTA / tests. Do not use.
        if type(json_wrapped) is dict and attribute in json_wrapped:
            # benchmark data obtained before 2021-03-04 uses {"attr": {"static": 0}}
            # benchmark data obtained after  2021-03-04 uses {"attr": {"type": "static", "value": 0}} or {"attr": None}
            # from_json expects the latter.
            if json_wrapped[attribute] is None:
                return None
            if (
                "static" in json_wrapped[attribute]
                and "type" not in json_wrapped[attribute]
            ):
                json_wrapped[attribute]["type"] = "static"
                json_wrapped[attribute]["value"] = json_wrapped[attribute]["static"]
                json_wrapped[attribute].pop("static")
            return cls.from_json(json_wrapped[attribute])
        return StaticFunction(0)


class StaticFunction(ModelFunction):
    def is_predictable(self, param_list=None):
        """
        Return whether the model function can be evaluated on the given parameter values.

        For a StaticFunction, this is always the case (i.e., this function always returns true).
        """
        return True

    def eval(self, param_list=None):
        """
        Evaluate model function with specified param/arg values.

        Far a Staticfunction, this is just the static value

        """
        return self.value

    def to_json(self):
        ret = super().to_json()
        ret.update({"type": "static", "value": self.value})
        return ret

    @classmethod
    def from_json(cls, data):
        assert data["type"] == "static"
        return cls(data["value"])

    def __repr__(self):
        return f"StaticFunction({self.value})"


class SplitFunction(ModelFunction):
    def __init__(self, value, param_index, child):
        super().__init__(value)
        self.param_index = param_index
        self.child = child

    def is_predictable(self, param_list):
        """
        Return whether the model function can be evaluated on the given parameter values.

        The first value corresponds to the lexically first model parameter, etc.
        All parameters must be set, not just the ones this function depends on.

        Returns False iff a parameter the function depends on is not numeric
        (e.g. None).
        """
        param_value = param_list[self.param_index]
        if param_value not in self.child:
            return False
        return self.child[param_value].is_predictable(param_list)

    def eval(self, param_list):
        param_value = param_list[self.param_index]
        return self.child[param_value].eval(param_list)

    def to_json(self):
        ret = super().to_json()
        ret.update(
            {
                "type": "split",
                "paramIndex": self.param_index,
                # TODO zusätzlich paramName
                "child": dict([[k, v.to_json()] for k, v in self.child.items()]),
            }
        )
        return ret

    @classmethod
    def from_json(cls, data):
        assert data["type"] == "split"
        self = cls(data["value"], data["paramIndex"], dict())

        for k, v in data["child"].items():
            self.child[k] = ModelFunction.from_json(v)

    def __repr__(self):
        return f"SplitFunction<{self.value}, param_index={self.param_index}>"


class SubstateFunction(ModelFunction):
    def __init__(self, value, sequence_by_count, count_model, sub_model):
        super().__init__(value)
        self.sequence_by_count = sequence_by_count
        self.count_model = count_model
        self.sub_model = sub_model

        # only used by analyze-archive model quality evaluation. Not serialized.
        self.static_duration = None

    def is_predictable(self, param_list):
        substate_count = round(self.count_model.eval(param_list))
        return substate_count in self.sequence_by_count

    def eval(self, param_list, duration=None):
        substate_count = round(self.count_model.eval(param_list))
        cumulative_energy = 0
        total_duration = 0
        substate_model, _ = self.sub_model.get_fitted()
        substate_sequence = self.sequence_by_count[substate_count]
        for i, sub_name in enumerate(substate_sequence):
            sub_duration = substate_model(sub_name, "duration", param=param_list)
            sub_power = substate_model(sub_name, "power", param=param_list)

            if i == substate_count - 1:
                if duration is not None:
                    sub_duration = duration - total_duration
                elif self.static_duration is not None:
                    sub_duration = self.static_duration - total_duration

            cumulative_energy += sub_power * sub_duration
            total_duration += sub_duration

        return cumulative_energy / total_duration

    def to_json(self):
        ret = super().to_json()
        ret.update(
            {
                "type": "substate",
                "sequence": self.sequence_by_count,
                "countModel": self.count_model.to_json(),
                "subModel": self.sub_model.to_json(),
            }
        )
        return ret

    @classmethod
    def from_json(cls, data):
        assert data["type"] == "substate"
        raise NotImplementedError

    def __repr__(self):
        return "SubstateFunction"


class AnalyticFunction(ModelFunction):
    """
    A multi-dimensional model function, generated from a string, which can be optimized using regression.

    The function describes a single model attribute (e.g. TX duration or send(...) energy)
    and how it is influenced by model parameters such as configured bit rate or
    packet length.
    """

    def __init__(
        self,
        value,
        function_str,
        parameters,
        num_args=0,
        regression_args=None,
        fit_by_param=None,
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
        :param regression_args: Initial regression variable values,
            both for function usage and least squares optimization.
            If unset, defaults to [1, 1, 1, ...]
        """
        super().__init__(value)
        self._parameter_names = parameters
        self._num_args = num_args
        self.model_function = function_str
        rawfunction = function_str
        self._dependson = [False] * (len(parameters) + num_args)
        self.fit_success = False
        self.fit_by_param = fit_by_param

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
            self.model_args = regression_args.copy()
            self._fit_success = True
        elif type(function_str) == str:
            self.model_args = list(np.ones((num_vars)))
        else:
            self.model_args = []

    def get_fit_data(self, by_param):
        """
        Return training data suitable for scipy.optimize.least_squares.

        :param by_param: measurement data, partitioned by parameter/arg values.
            by_param[*] must be a list or 1-D NumPy array containing the ground truth.
            The parameter values (dict keys) must be numeric for
            all parameters this function depends on -- otherwise, the
            corresponding data will be left out. Parameter values must be
            ordered according to the order of parameter names used in
            the ParamFunction constructor. Argument values (if any) always come after
            parameters, in the order of their index in the function signature.

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
            if len(key) == dimension:
                valid = True
                num_total += 1
                for i in range(dimension):
                    if self._dependson[i] and not is_numeric(key[i]):
                        valid = False
                if valid:
                    num_valid += 1
                    Y.extend(val)
                    for i in range(dimension):
                        if self._dependson[i]:
                            X[i].extend([float(key[i])] * len(val))
                        else:
                            X[i].extend([np.nan] * len(val))
            else:
                logger.warning(
                    "Invalid parameter key length while gathering fit data. is {}, want {}.".format(
                        len(key), dimension
                    )
                )
        X = np.array(X)
        Y = np.array(Y)

        return X, Y, num_valid, num_total

    def fit(self, by_param):
        """
        Fit the function on measurements via least squares regression.

        :param by_param: measurement data, partitioned by parameter/arg values

        The ground truth is read from by_param[*],
        which must be a list or 1-D NumPy array. Parameter values must be
        ordered according to the parameter names in the constructor. If
        argument values are present, they must come after parameter values
        in the order of their appearance in the function signature.
        """
        X, Y, num_valid, num_total = self.get_fit_data(by_param)
        if num_valid > 2:
            error_function = lambda P, X, y: self._function(P, X) - y
            try:
                res = optimize.least_squares(
                    error_function, self.model_args, args=(X, Y), xtol=2e-15
                )
            except ValueError as err:
                logger.warning(f"Fit failed: {err} (function: {self.model_function})")
                return
            if res.status > 0:
                self.model_args = res.x
                self.fit_success = True
            else:
                logger.warning(
                    f"Fit failed: {res.message} (function: {self.model_function})"
                )
        else:
            logger.warning("Insufficient amount of valid parameter keys, cannot fit")

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

    def eval(self, param_list):
        """
        Evaluate model function with specified param/arg values.

        :param param_list: parameter values (list of float). First item
            corresponds to lexically first parameter, etc.
        :param arg_list: argument values (list of float), if arguments are used.
        """
        return self._function(self.model_args, param_list)

    def to_json(self):
        ret = super().to_json()
        ret.update(
            {
                "type": "analytic",
                "functionStr": self.model_function,
                "argCount": self._num_args,
                "parameterNames": self._parameter_names,
                "regressionModel": list(self.model_args),
            }
        )
        return ret

    @classmethod
    def from_json(cls, data):
        assert data["type"] == "analytic"

        return cls(
            data["value"],
            data["functionStr"],
            data["parameterNames"],
            data["argCount"],
            data["regressionModel"],
        )

    def __repr__(self):
        return f"AnalyticFunction<{self.value}, {self.model_function}>"


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
                if is_numeric(function_item[0]):
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
        return AnalyticFunction(
            None, buf, parameter_names, num_args, fit_by_param=fit_results
        )
