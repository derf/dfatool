#!/usr/bin/env python3

import json
import numpy as np
import re
import logging
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def running_mean(x: np.ndarray, N: int) -> np.ndarray:
    """
    Compute `N` elements wide running average over `x`.

    :param x: 1-Dimensional NumPy array
    :param N: how many items to average
    """
    # FIXME np.insert(x, 0, [x[0] for i in range(N/2)])
    # FIXME np.insert(x, -1, [x[-1] for i in range(N/2)])
    # (dabei ungerade N beachten)
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def human_readable(value, unit):
    for prefix, factor in (
        ("p", 1e-12),
        ("n", 1e-9),
        ("µ", 1e-6),
        ("m", 1e-3),
        ("", 1),
        ("k", 1e3),
    ):
        if value < 1e3 * factor:
            return "{:.2f} {}{}".format(value * (1 / factor), prefix, unit)
    return "{:.2f} {}".format(value, unit)


def is_numeric(n):
    """Check if `n` is numeric (i.e., it can be converted to float)."""
    if n is None:
        return False
    try:
        float(n)
        return True
    except ValueError:
        return False


def is_power_of_two(n):
    """Check if `n` is a power of two (1, 2, 4, 8, 16, ...)."""
    return n > 0 and (n & (n - 1)) == 0


def float_or_nan(n):
    """Convert `n` to float (if numeric) or NaN."""
    if n is None:
        return np.nan
    try:
        return float(n)
    except ValueError:
        return np.nan


def soft_cast_int(n):
    """
    Convert `n` to int (if numeric) or return it as-is.

    If `n` is empty, returns None.
    If `n` is not numeric, it is left unchanged.
    """
    if n is None or n == "":
        return None
    try:
        return int(n)
    except ValueError:
        return n


def soft_cast_float(n):
    """
    Convert `n` to float (if numeric) or return it as-is.

    If `n` is empty, returns None.
    If `n` is not numeric, it is left unchanged.
    """
    if n is None or n == "":
        return None
    try:
        return float(n)
    except ValueError:
        return n


def flatten(somelist):
    """
    Flatten a list.

    Example: flatten([[1, 2], [3], [4, 5]]) -> [1, 2, 3, 4, 5]
    """
    return [item for sublist in somelist for item in sublist]


def parse_conf_str(conf_str):
    """
    Parse a configuration string `k1=v1,k2=v2`... and return a dict `{'k1': v1, 'k2': v2}`...

    Values are casted to float if possible and kept as-is otherwise.
    """
    conf_dict = dict()
    for option in conf_str.split(","):
        key, value = option.split("=")
        conf_dict[key] = soft_cast_float(value)
    return conf_dict


def remove_index_from_tuple(parameters, index):
    """
    Remove the element at `index` from tuple `parameters`.

    :param parameters: tuple
    :param index: index of element which is to be removed
    :returns: parameters tuple without the element at index
    """
    return (*parameters[:index], *parameters[index + 1 :])


def param_slice_eq(a, b, index):
    """
    Check if by_param keys a and b are identical, ignoring the parameter at index.

    parameters:
    a, b -- (state/transition name, [parameter0 value, parameter1 value, ...])
    index -- parameter index to ignore (0 -> parameter0, 1 -> parameter1, etc.)

    Returns True iff a and b have the same state/transition name, and all
    parameters at positions != index are identical.

    example:
    ('foo', [1, 4]), ('foo', [2, 4]), 0 -> True
    ('foo', [1, 4]), ('foo', [2, 4]), 1 -> False

    """
    if (*a[:index], *a[index + 1 :]) == (*b[:index], *b[index + 1 :]):
        return True
    return False


def match_parameter_values(input_param: dict, match_param: dict):
    """
    Check whether one of the paramaters in `input_param` has the same value in `match_param`.

    :param input_param: parameter dict of a state/transition/... measurement
    :param match_param: parameter value filter
    :returns: True if for all parameters k in match_param: input_param[k] == match_param[k], or if match_param is None.
    """
    if match_param is None:
        return True
    for k, v in match_param.items():
        if k in input_param and input_param[k] != v:
            return False
    return True


def partition_by_param(data, param_values, ignore_parameters=list()):
    ret = dict()
    for i, parameters in enumerate(param_values):
        # ensure that parameters[param_index] = None does not affect the "param_values" entries passed to this function
        parameters = list(parameters)
        for param_index in ignore_parameters:
            parameters[param_index] = None
        param_key = tuple(parameters)
        if param_key not in ret:
            ret[param_key] = list()
        ret[param_key].append(data[i])
    return ret


def param_dict_to_list(param_dict, parameter_names, default=None):
    """
    Convert {"foo": 1, "bar": 2}, ["bar", "foo", "quux"] to [2, 1, None]
    """
    ret = list()
    for parameter_name in parameter_names:
        ret.append(param_dict.get(parameter_name, None))
    return ret


def observations_to_by_name(observations: list, attributes: list):
    """
    Convert observation list to by_name dictionary for AnalyticModel analysis

    :param observations: list of dicts, each representing one measurement. dict keys:
        "name": name of observed state/transition/...
        "param": {"parameter name": parameter value, ...} dict
    :param attributes: observed attributes (i.e., actual measurements). Each measurement dict must have an
        entry holding the data value for each attribute. It should not be None.

    :returns: tuple (by_name, parameter_names) which can be passed to AnalyticModel
    """
    parameter_names = set()
    by_name = dict()
    for observation in observations:
        parameter_names.update(observation["param"].keys())
        name = observation["name"]
        if name not in by_name:
            by_name[name] = {"attributes": attributes, "param": list()}
            for attribute in attributes:
                by_name[name][attribute] = list()
    parameter_names = sorted(parameter_names)
    for observation in observations:
        name = observation["name"]
        by_name[name]["param"].append(
            param_dict_to_list(observation["param"], parameter_names)
        )
        for attribute in attributes:
            by_name[name][attribute].append(observation[attribute])
    for name in by_name:
        for attribute in attributes:
            by_name[name][attribute] = np.array(by_name[name][attribute])
    return by_name, parameter_names


def by_name_to_by_param(by_name: dict):
    """
    Convert aggregation by name to aggregation by name and parameter values.
    """
    by_param = dict()
    for name in by_name.keys():
        for i, parameters in enumerate(by_name[name]["param"]):
            param_key = (name, tuple(parameters))
            if param_key not in by_param:
                by_param[param_key] = dict()
                for key in by_name[name].keys():
                    by_param[param_key][key] = list()
                by_param[param_key]["attributes"] = by_name[name]["attributes"]
                # special case for PTA models
                if "isa" in by_name[name]:
                    by_param[param_key]["isa"] = by_name[name]["isa"]
            for attribute in by_name[name]["attributes"]:
                by_param[param_key][attribute].append(by_name[name][attribute][i])
            if "supports" in by_name[name]:
                for support in by_name[name]["supports"]:
                    by_param[param_key][support].append(by_name[name][support][i])
            # Required for match_parameter_valuse in _try_fits
            by_param[param_key]["param"].append(by_name[name]["param"][i])
    return by_param


def by_param_to_by_name(by_param: dict) -> dict:
    """
    Convert aggregation by name and parameter values to aggregation by name only.
    """
    by_name = dict()
    for param_key in by_param.keys():
        name, _ = param_key
        if name not in by_name:
            by_name[name] = dict()
            for key in by_param[param_key].keys():
                by_name[name][key] = list()
            by_name[name]["attributes"] = by_param[param_key]["attributes"]
            # special case for PTA models
            if "isa" in by_param[param_key]:
                by_name[name]["isa"] = by_param[param_key]["isa"]
        for attribute in by_name[name]["attributes"]:
            by_name[name][attribute].extend(by_param[param_key][attribute])
        if "supports" in by_param[param_key]:
            for support in by_param[param_key]["supports"]:
                by_name[name][support].extend(by_param[param_key][support])
        by_name[name]["param"].extend(by_param[param_key]["param"])
    for name in by_name.keys():
        for attribute in by_name[name]["attributes"]:
            by_name[name][attribute] = np.array(by_name[name][attribute])
    return by_name


def filter_aggregate_by_param(aggregate, parameters, parameter_filter):
    """
    Remove entries which do not have certain parameter values from `aggregate`.

    :param aggregate: aggregated measurement data, must be a dict conforming to
        aggregate[state or transition name]['param'] = (first parameter value, second parameter value, ...)
        and
        aggregate[state or transition name]['attributes'] = [list of keys with measurement data, e.g. 'power' or 'duration']
    :param parameters: list of parameters, used to map parameter index to parameter name. parameters=['foo', ...] means 'foo' is the first parameter
    :param parameter_filter: [[name, value], [name, value], ...] list of parameter values to keep, all others are removed. Values refer to normalizad parameter data.
    """
    for param_name_and_value in parameter_filter:
        param_index = parameters.index(param_name_and_value[0])
        param_value = soft_cast_int(param_name_and_value[1])
        names_to_remove = set()
        for name in aggregate.keys():
            indices_to_keep = list(
                map(lambda x: x[param_index] == param_value, aggregate[name]["param"])
            )
            aggregate[name]["param"] = list(
                map(
                    lambda iv: iv[1],
                    filter(
                        lambda iv: indices_to_keep[iv[0]],
                        enumerate(aggregate[name]["param"]),
                    ),
                )
            )
            if len(indices_to_keep) == 0:
                logger.debug("??? {}->{}".format(parameter_filter, name))
                names_to_remove.add(name)
            else:
                for attribute in aggregate[name]["attributes"]:
                    aggregate[name][attribute] = aggregate[name][attribute][
                        indices_to_keep
                    ]
                    if len(aggregate[name][attribute]) == 0:
                        names_to_remove.add(name)
        for name in names_to_remove:
            aggregate.pop(name)


def detect_outliers_in_aggregate(aggregate, z_limit=10, remove_outliers=False):
    for name in aggregate.keys():
        indices_to_remove = set()
        attributes = list()
        for attribute in aggregate[name]["attributes"]:
            data = aggregate[name][attribute]
            z_scores = (data - np.mean(data)) / np.std(data)
            outliers = np.abs(z_scores) > z_limit
            if np.any(outliers) and remove_outliers:
                indices_to_remove = indices_to_remove.union(
                    np.arange(len(outliers))[outliers]
                )
                attributes.append(attribute)
            elif np.any(outliers):
                logger.info(
                    f"{name} {attribute} has {len(z_scores[outliers])} outliers"
                )
        if indices_to_remove:
            # Assumption: len(aggregate[name][attribute]) is the same for each
            # attribute.
            logger.info(
                f"Removing outliers {indices_to_remove} from {name}. Affected attributes: {attributes}"
            )
            indices_to_keep = map(
                lambda x: x not in indices_to_remove, np.arange(len(outliers))
            )
            indices_to_keep = np.array(list(indices_to_keep))
            for attribute in aggregate[name]["attributes"]:
                aggregate[name][attribute] = aggregate[name][attribute][indices_to_keep]
            aggregate[name]["param"] = list(
                map(
                    lambda iv: iv[1],
                    filter(
                        lambda iv: indices_to_keep[iv[0]],
                        enumerate(aggregate[name]["param"]),
                    ),
                )
            )


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


class OptionalTimingAnalysis:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.wrapped_lines = list()
        self.index = 1

    def get_header(self):
        ret = ""
        if self.enabled:
            ret += "#define TIMEIT(index, functioncall) "
            ret += "counter.start(); "
            ret += "functioncall; "
            ret += "counter.stop();"
            ret += 'kout << endl << index << " :: " << counter.value << "/" << counter.overflow << endl;\n'
        return ret

    def wrap_codeblock(self, codeblock):
        if not self.enabled:
            return codeblock
        lines = codeblock.split("\n")
        ret = list()
        for line in lines:
            if re.fullmatch(".+;", line):
                ret.append("TIMEIT( {:d}, {} )".format(self.index, line))
                self.wrapped_lines.append(line)
                self.index += 1
            else:
                ret.append(line)
        return "\n".join(ret)
