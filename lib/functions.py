#!/usr/bin/env python3
"""
Utilities for analytic description of parameter-dependent model attributes.

This module provides classes and helper functions useful for least-squares
regression and general handling of model functions.
"""
from itertools import chain, combinations
import logging
import numpy as np
import os
import re
from scipy import optimize
from .utils import is_numeric, param_to_ndarray

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

    def __init__(self, param_function, validation_function, num_vars, repr_str=None):
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
        self.repr_str = repr_str

    def __repr__(self) -> str:
        if self.repr_str:
            return f"ParamFunction<{self.repr_str}>"
        return f"ParamFunction<{self._param_function}, {self.validation_function}, {self._num_variables}>"

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
    always_predictable = False
    has_eval_arr = False
    """
    Encapsulates the behaviour of a single model attribute, e.g. TX power or write duration.

    The behaviour may be constant or depend on a number of factors. Modelfunction is a virtual base class,
    individuel decendents describe actual behaviour.

    Common attributes:
    :param value: median data value
    :type value: float
    :param value_error: static model value error
    :type value_error: dict, optional
    :param function_error: model error
    :type value_error: dict, optional
    """

    def __init__(self, value, n_samples=None):
        # a model always has a static (median/mean) value. For StaticFunction, it's the only data point.
        # For more complex models, it's usede both as fallback in case the model cannot predict the current
        # parameter combination, and for use cases requiring static models
        self.value = value
        self.n_samples = n_samples

        # A ModelFunction may track its own accuracy, both of the static value and of the eval() method.
        # However, it does not specify how the accuracy was calculated (e.g. which data was used and whether cross-validation was performed)
        self.value_error = None
        self.function_error = None

    def is_predictable(self, param_list):
        raise NotImplementedError

    def eval(self, param_list):
        raise NotImplementedError

    def eval_arr(self, params):
        raise NotImplementedError

    def get_complexity_score(self):
        raise NotImplementedError

    def eval_mae(self, param_list):
        """Return model Mean Absolute Error (MAE) for `param_list`."""
        if self.is_predictable(param_list):
            return self.function_error["mae"]
        return self.value_error["mae"]

    def webconf_function_map(self):
        return list()

    def to_json(self, **kwargs):
        """Convert model to JSON."""
        ret = {
            "value": self.value,
            "n_samples": self.n_samples,
            "valueError": self.value_error,
            "functionError": self.function_error,
        }
        return ret

    @classmethod
    def from_json(cls, data):
        """
        Create ModelFunction instance from JSON.

        Delegates to StaticFunction, SplitFunction, etc. as appropriate.
        """
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
    always_predictable = True
    has_eval_arr = True

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

    def eval_arr(self, params):
        return [self.value for p in params]

    def get_complexity_score(self):
        return 1

    def to_json(self, **kwargs):
        ret = super().to_json(**kwargs)
        ret.update({"type": "static", "value": self.value})
        return ret

    def to_dot(self, pydot, graph, feature_names, parent=None):
        graph.add_node(
            pydot.Node(str(id(self)), label=f"{self.value:.2f}", shape="rectangle")
        )

    @classmethod
    def from_json(cls, data):
        assert data["type"] == "static"
        return cls(data["value"])

    def __repr__(self):
        return f"StaticFunction({self.value})"


class SplitFunction(ModelFunction):
    def __init__(self, value, param_index, child, **kwargs):
        super().__init__(value, **kwargs)
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
        if param_value in self.child:
            return self.child[param_value].is_predictable(param_list)
        return all(
            map(lambda child: child.is_predictable(param_list), self.child.values())
        )

    def eval(self, param_list):
        param_value = param_list[self.param_index]
        if param_value in self.child:
            return self.child[param_value].eval(param_list)
        return np.mean(
            list(map(lambda child: child.eval(param_list), self.child.values()))
        )

    def webconf_function_map(self):
        ret = list()
        for child in self.child.values():
            ret.extend(child.webconf_function_map())
        return ret

    def to_json(self, **kwargs):
        ret = super().to_json(**kwargs)
        with_param_name = kwargs.get("with_param_name", False)
        param_names = kwargs.get("param_names", list())
        update = {
            "type": "split",
            "paramIndex": self.param_index,
            "child": dict([[k, v.to_json(**kwargs)] for k, v in self.child.items()]),
        }
        if with_param_name and param_names:
            update["paramName"] = param_names[self.param_index]
        ret.update(update)
        return ret

    def get_number_of_nodes(self):
        ret = 1
        for v in self.child.values():
            if type(v) is SplitFunction:
                ret += v.get_number_of_nodes()
            else:
                ret += 1
        return ret

    def get_max_depth(self):
        ret = [0]
        for v in self.child.values():
            if type(v) is SplitFunction:
                ret.append(v.get_max_depth())
        return 1 + max(ret)

    def get_number_of_leaves(self):
        ret = 0
        for v in self.child.values():
            if type(v) is SplitFunction:
                ret += v.get_number_of_leaves()
            else:
                ret += 1
        return ret

    def get_complexity_score(self):
        if not self.child:
            return 1
        ret = 1
        for v in self.child.values():
            ret += v.get_complexity_score()
        return ret

    def to_dot(self, pydot, graph, feature_names, parent=None):
        try:
            label = feature_names[self.param_index]
        except IndexError:
            label = f"param{self.param_index}"
        graph.add_node(pydot.Node(str(id(self)), label=label))
        for key, child in self.child.items():
            child.to_dot(pydot, graph, feature_names, str(id(self)))
            graph.add_edge(pydot.Edge(str(id(self)), str(id(child)), label=key))

    @classmethod
    def from_json(cls, data):
        assert data["type"] == "split"
        self = cls(data["value"], data["paramIndex"], dict())

        for k, v in data["child"].items():
            self.child[k] = ModelFunction.from_json(v)

        return self

    def __repr__(self):
        return f"SplitFunction<{self.value}, param_index={self.param_index}>"


class SubstateFunction(ModelFunction):
    def __init__(self, value, sequence_by_count, count_model, sub_model, **kwargs):
        super().__init__(value, **kwargs)
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

    def to_json(self, **kwargs):
        ret = super().to_json(**kwargs)
        ret.update(
            {
                "type": "substate",
                "sequence": self.sequence_by_count,
                "countModel": self.count_model.to_json(**kwargs),
                "subModel": self.sub_model.to_json(**kwargs),
            }
        )
        return ret

    @classmethod
    def from_json(cls, data):
        assert data["type"] == "substate"
        raise NotImplementedError

    def __repr__(self):
        return "SubstateFunction"


class SKLearnRegressionFunction(ModelFunction):
    always_predictable = True
    has_eval_arr = True

    def __init__(self, value, regressor, categorial_to_index, ignore_index, **kwargs):
        super().__init__(value, **kwargs)
        self.regressor = regressor
        self.categorial_to_index = categorial_to_index
        self.ignore_index = ignore_index

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
        if param_list is None:
            return self.value
        actual_param_list = list()
        for i, param in enumerate(param_list):
            if not self.ignore_index[i]:
                if i in self.categorial_to_index:
                    try:
                        actual_param_list.append(self.categorial_to_index[i][param])
                    except KeyError:
                        # param was not part of training data. substitute an unused scalar.
                        # Note that all param values which were not part of training data map to the same scalar this way.
                        # This should be harmless.
                        actual_param_list.append(
                            max(self.categorial_to_index[i].values()) + 1
                        )
                else:
                    actual_param_list.append(int(param))
        predictions = self.regressor.predict(np.array([actual_param_list]))
        if predictions.shape == (1,):
            return predictions[0]
        return predictions

    def eval_arr(self, params):
        actual_params = list()
        for param_tuple in params:
            actual_param_list = list()
            for i, param in enumerate(param_tuple):
                if not self.ignore_index[i]:
                    if i in self.categorial_to_index:
                        try:
                            actual_param_list.append(self.categorial_to_index[i][param])
                        except KeyError:
                            # param was not part of training data. substitute an unused scalar.
                            # Note that all param values which were not part of training data map to the same scalar this way.
                            # This should be harmless.
                            actual_param_list.append(
                                max(self.categorial_to_index[i].values()) + 1
                            )
                    else:
                        actual_param_list.append(int(param))
            actual_params.append(actual_param_list)
        predictions = self.regressor.predict(np.array(actual_params))
        return predictions


class CARTFunction(SKLearnRegressionFunction):
    def get_number_of_nodes(self):
        return self.regressor.tree_.node_count

    def get_number_of_leaves(self):
        return self.regressor.tree_.n_leaves

    def get_max_depth(self):
        return self.regressor.get_depth()

    def get_complexity_score(self):
        return self.get_number_of_nodes()

    def to_json(self, feature_names=None, **kwargs):
        import sklearn.tree

        self.leaf_id = sklearn.tree._tree.TREE_LEAF
        self.feature_names = feature_names

        ret = super().to_json(**kwargs)
        ret.update(self.recurse_(self.regressor.tree_, 0))
        return ret

    # recursive function for all nodes:
    def recurse_(self, tree, node_id, depth=0):
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        # basic leaf with standard values
        # conversion because of numpy
        sub_data = {
            "functionError": None,
            "type": "static",
            "value": float(tree.value[node_id]),
            "valueError": float(tree.impurity[node_id]),
            # "samples": int(tree.n_node_samples[node_id])
        }

        # if has childs / not a leaf:
        if left_child != self.leaf_id or right_child != self.leaf_id:
            # sub_data["paramName"] = "X[" + str(self.regressor.tree_.feature[left_child_id]) + "]"
            # sub_data["paramIndex"] = int(self.regressor.tree_.feature[left_child_id])
            sub_data["paramName"] = self.feature_names[
                self.regressor.tree_.feature[node_id]
            ]
            sub_data["paramDecisionValue"] = tree.threshold[node_id]
            sub_data["type"] = "scalarSplit"

        # child value
        if left_child != self.leaf_id:
            sub_data["left"] = self.recurse_(tree, left_child, depth=depth + 1)
        if right_child != self.leaf_id:
            sub_data["right"] = self.recurse_(tree, right_child, depth=depth + 1)

        return sub_data


class LMTFunction(SKLearnRegressionFunction):
    def get_number_of_nodes(self):
        return self.regressor.node_count

    def get_number_of_leaves(self):
        return len(self.regressor._leaves.keys())

    def get_complexity_score(self):
        ret = self.get_number_of_nodes() - self.get_number_of_leaves()
        for leaf in self.regressor._leaves.values():
            ret += len(
                list(
                    filter(lambda x: x > 0, leaf.model.coef_ + [leaf.model.intercept_])
                )
            )
        return ret

    def get_max_depth(self):
        return max(map(len, self.regressor._leaves.keys())) + 1

    def to_json(self, feature_names=None, **kwargs):
        self.feature_names = feature_names
        ret = super().to_json(**kwargs)
        ret.update(self.recurse_(self.regressor.summary(), 0))
        return ret

    def recurse_(self, node_hash, node_index):
        node = node_hash[node_index]
        sub_data = dict()
        if "th" in node:
            return {
                "type": "scalarSplit",
                "paramName": self.feature_names[node["col"]],
                "paramDecisionValue": node["th"],
                "left": self.recurse_(node_hash, node["children"][0]),
                "right": self.recurse_(node_hash, node["children"][1]),
            }
        model = node["models"]
        fs = "0 + regression_arg(0)"
        for i, coef in enumerate(model.coef_):
            if coef:
                fs += f" + regression_arg({i+1}) * parameter({self.feature_names[i]})"
        return {
            "type": "analytic",
            "functionStr": fs,
            "parameterNames": self.feature_names,
            "regressionModel": [model.intercept_] + list(model.coef_),
        }


class XGBoostFunction(SKLearnRegressionFunction):
    def to_json(self, feature_names=None, **kwargs):
        import json

        tempfile = f"/tmp/xgb{os.getpid()}.json"

        self.regressor.get_booster().dump_model(
            tempfile, dump_format="json", with_stats=True
        )
        with open(tempfile, "r") as f:
            data = json.load(f)
        os.remove(tempfile)

        if feature_names:
            return list(
                map(
                    lambda tree: self.tree_to_webconf_json(
                        tree, feature_names, **kwargs
                    ),
                    data,
                )
            )
        return data

    def tree_to_webconf_json(self, tree, feature_names, **kwargs):
        ret = dict()
        if "children" in tree:
            return {
                "functionError": None,
                "type": "scalarSplit",
                "paramName": feature_names[int(tree["split"][1:])],
                "paramDecisionValue": tree["split_condition"],
                "value": None,
                "valueError": None,
                "left": self.tree_to_webconf_json(
                    tree["children"][0], feature_names, **kwargs
                ),
                "right": self.tree_to_webconf_json(
                    tree["children"][1], feature_names, **kwargs
                ),
            }
        else:
            return {
                "functionError": None,
                "type": "static",
                "value": tree["leaf"],
                "valueError": None,
            }

    def get_number_of_nodes(self):
        return sum(map(self._get_number_of_nodes, self.to_json()))

    def _get_number_of_nodes(self, data):
        ret = 1
        for child in data.get("children", list()):
            ret += self._get_number_of_nodes(child)
        return ret

    def get_number_of_leaves(self):
        return sum(map(self._get_number_of_leaves, self.to_json()))

    def _get_number_of_leaves(self, data):
        if "leaf" in data:
            return 1
        ret = 0
        for child in data.get("children", list()):
            ret += self._get_number_of_leaves(child)
        return ret

    def get_max_depth(self):
        return max(map(self._get_max_depth, self.to_json()))

    def _get_max_depth(self, data):
        ret = [0]
        for child in data.get("children", list()):
            ret.append(self._get_max_depth(child))
        return 1 + max(ret)

    def get_complexity_score(self):
        return self.get_number_of_nodes()

    def to_dref(self):
        return {
            "hyper/n estimators": self.regressor.n_estimators,
            "hyper/max depth": self.regressor.max_depth,
            "hyper/subsample": self.regressor.subsample,
            "hyper/eta": self.regressor.learning_rate,
            "hyper/gamma": self.regressor.gamma,
            "hyper/alpha": self.regressor.reg_alpha,
            "hyper/lambda": self.regressor.reg_lambda,
        }


# first-order linear function (no feature interaction)
class FOLFunction(ModelFunction):
    always_predictable = True

    def __init__(self, value, parameters, num_args=0, **kwargs):
        super().__init__(value, **kwargs)
        self.parameter_names = parameters
        self._num_args = num_args
        self.fit_success = False

    def fit(self, param_values, data, ignore_param_indexes=None):
        categorial_to_scalar = bool(
            int(os.getenv("DFATOOL_PARAM_CATEGORIAL_TO_SCALAR", "0"))
        )
        second_order = int(os.getenv("DFATOOL_FOL_SECOND_ORDER", "0"))
        fit_parameters, categorial_to_index, ignore_index = param_to_ndarray(
            param_values,
            with_nan=False,
            categorial_to_scalar=categorial_to_scalar,
            ignore_indexes=ignore_param_indexes,
        )
        self.categorial_to_index = categorial_to_index
        self.ignore_index = ignore_index
        fit_parameters = fit_parameters.swapaxes(0, 1)

        if second_order:
            num_param = fit_parameters.shape[0]
            rawbuf = "reg_param[0]"
            num_vars = 1
            for i in range(num_param):
                if second_order == 2:
                    rawbuf += f" + reg_param[{num_vars}] * model_param[{i}]"
                    num_vars += 1
                for j in range(i + 1, num_param):
                    rawbuf += f" + reg_param[{num_vars}] * model_param[{i}] * model_param[{j}]"
                    num_vars += 1
            funbuf = "regression_arg(0)"
            num_vars = 1
            for j, param_name in enumerate(self.parameter_names):
                if ignore_index[j]:
                    continue
                else:
                    if second_order == 2:
                        funbuf += (
                            f" + regression_arg({num_vars}) * parameter({param_name})"
                        )
                        num_vars += 1
                    for k in range(j + 1, len(self.parameter_names)):
                        if ignore_index[j]:
                            continue
                        funbuf += f" + regression_arg({num_vars}) * parameter({param_name}) * parameter({self.parameter_names[k]})"
                        num_vars += 1
        else:
            num_vars = fit_parameters.shape[0] + 1
            rawbuf = "reg_param[0]"
            for i in range(1, num_vars):
                rawbuf += f" + reg_param[{i}] * model_param[{i-1}]"
            funbuf = "regression_arg(0)"
            i = 1
            for j, param_name in enumerate(self.parameter_names):
                if ignore_index[j]:
                    continue
                else:
                    funbuf += f" + regression_arg({i}) * parameter({param_name})"
                    i += 1

        self.model_function = funbuf
        self._function_str = "lambda reg_param, model_param:" + rawbuf
        self._function = eval(self._function_str)

        error_function = lambda P, X, y: self._function(P, X) - y
        self.model_args = list(np.ones((num_vars)))
        try:
            res = optimize.least_squares(
                error_function, self.model_args, args=(fit_parameters, data), xtol=2e-15
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

    def is_predictable(self, param_list=None):
        """
        Return whether the model function can be evaluated on the given parameter values.
        """
        return True

    def eval(self, param_list=None):
        """
        Evaluate model function with specified param/arg values.

        Far a Staticfunction, this is just the static value

        """
        if param_list is None:
            return self.value
        actual_param_list = list()
        for i, param in enumerate(param_list):
            if not self.ignore_index[i]:
                if i in self.categorial_to_index:
                    try:
                        actual_param_list.append(self.categorial_to_index[i][param])
                    except KeyError:
                        # param was not part of training data. substitute an unused scalar.
                        # Note that all param values which were not part of training data map to the same scalar this way.
                        # This should be harmless.
                        actual_param_list.append(
                            max(self.categorial_to_index[i].values()) + 1
                        )
                else:
                    actual_param_list.append(int(param))
        try:
            return self._function(self.model_args, actual_param_list)
        except FloatingPointError as e:
            logger.error(
                f"{e} when predicting {self._function_str}({self.model_args}, {actual_param_list}) for {param_list}, returning static value"
            )
            return self.value
        except TypeError as e:
            logger.error(
                f"{e} when predicting {self._function_str}({self.model_args}, {actual_param_list}) for {param_list}"
            )
            raise

    def get_complexity_score(self):
        return len(self.model_args)

    def to_dot(self, pydot, graph, feature_names, parent=None):
        model_function = self.model_function
        for i, arg in enumerate(self.model_args):
            model_function = model_function.replace(
                f"regression_arg({i})", f"{arg:.2f}"
            )
        graph.add_node(
            pydot.Node(str(id(self)), label=model_function, shape="rectangle")
        )

    def to_json(self, **kwargs):
        ret = super().to_json(**kwargs)
        ret.update(
            {
                "type": "analytic",
                "functionStr": self.model_function,
                "argCount": self._num_args,
                "parameterNames": self.parameter_names,
                "regressionModel": list(self.model_args),
            }
        )
        return ret


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
        **kwargs,
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
        super().__init__(value, **kwargs)
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
            logger.debug("Insufficient amount of valid parameter keys, cannot fit")

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
        try:
            return self._function(self.model_args, param_list)
        except FloatingPointError as e:
            logger.error(
                f"{e} when predicting {self._function_str}({param_list}), returning static value"
            )
            return self.value

    def get_complexity_score(self):
        return len(self.model_args)

    def webconf_function_map(self):
        js_buf = self.model_function
        for i in range(len(self.model_args)):
            js_buf = js_buf.replace(f"regression_arg({i})", str(self.model_args[i]))
        for parameter_name in self._parameter_names:
            js_buf = js_buf.replace(
                f"parameter({parameter_name})", f"""param["{parameter_name}"]"""
            )
        for arg_num in range(self._num_args):
            js_buf = js_buf.replace(f"function_arg({arg_num})", f"args[{arg_num}]")
        js_buf = "(param, args) => " + js_buf.replace("np.", "Math.")
        return [(f'"{self.model_function}"', js_buf)]

    def to_json(self, **kwargs):
        ret = super().to_json(**kwargs)
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

    def to_dot(self, pydot, graph, feature_names, parent=None):
        model_function = self.model_function
        for i, arg in enumerate(self.model_args):
            model_function = model_function.replace(
                f"regression_arg({i})", f"{arg:.2f}"
            )
        graph.add_node(
            pydot.Node(str(id(self)), label=model_function, shape="rectangle")
        )

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
        "square": lambda x: x**2,
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
                repr_str="β₀ + β₁ * x",
            ),
            "logarithmic": ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * np.log(model_param),
                lambda model_param: model_param > 0,
                2,
                repr_str="β₀ + β₁ * np.log(x)",
            ),
            "logarithmic1": ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * np.log(model_param + 1),
                lambda model_param: model_param > -1,
                2,
                repr_str="β₀ + β₁ * np.log(x+1)",
            ),
            "exponential": ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * np.exp(model_param),
                lambda model_param: model_param <= 64,
                2,
                repr_str="β₀ + β₁ * np.exp(x)",
            ),
            #'polynomial' : lambda reg_param, model_param: reg_param[0] + reg_param[1] * model_param + reg_param[2] * model_param ** 2,
            "square": ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * model_param**2,
                lambda model_param: True,
                2,
                repr_str="β₀ + β₁ * x²",
            ),
            "inverse": ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] / model_param,
                lambda model_param: model_param != 0,
                2,
                repr_str="β₀ + β₁ * 1/x",
            ),
            "sqrt": ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * np.sqrt(model_param),
                lambda model_param: model_param >= 0,
                2,
                repr_str="β₀ + β₁ * np.sqrt(x)",
            ),
            # "num0_8": ParamFunction(
            #    lambda reg_param, model_param: reg_param[0]
            #    + reg_param[1] * analytic._num0_8(model_param),
            #    lambda model_param: True,
            #    2,
            # ),
            # "num0_16": ParamFunction(
            #    lambda reg_param, model_param: reg_param[0]
            #    + reg_param[1] * analytic._num0_16(model_param),
            #    lambda model_param: True,
            #    2,
            # ),
            # "num1": ParamFunction(
            #    lambda reg_param, model_param: reg_param[0]
            #    + reg_param[1] * analytic._num1(model_param),
            #    lambda model_param: True,
            #    2,
            # ),
        }

        if safe_functions_enabled or bool(
            int(os.getenv("DFATOOL_REGRESSION_SAFE_FUNCTIONS", "0"))
        ):
            functions.pop("logarithmic1")
            functions.pop("logarithmic")
            functions["safe_log"] = ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * analytic._safe_log(model_param),
                lambda model_param: True,
                2,
                repr_str="β₀ + β₁ * safe_log(x)",
            )
            functions.pop("inverse")
            functions["safe_inv"] = ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * analytic._safe_inv(model_param),
                lambda model_param: True,
                2,
                repr_str="β₀ + β₁ * safe(1/x)",
            )
            functions.pop("sqrt")
            functions["safe_sqrt"] = ParamFunction(
                lambda reg_param, model_param: reg_param[0]
                + reg_param[1] * analytic._safe_sqrt(model_param),
                lambda model_param: True,
                2,
                repr_str="β₀ + β₁ * safe_sqrt(x)",
            )

        if bool(int(os.getenv("DFATOOL_FIT_LINEAR_ONLY", "0"))):
            functions = {"linear": functions["linear"]}

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
