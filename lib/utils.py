#!/usr/bin/env python3

import json
import numpy as np
import os
import re
import logging
from contextlib import contextmanager
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


@contextmanager
def cd(path):
    old_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_dir)


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


class Logfile:
    def __init__(self):
        pass

    def kv_to_param(self, kv_str, cast):
        try:
            key, value = kv_str.split("=")
            value = cast(value)
            return key, value
        except ValueError:
            logger.warning(f"Invalid key-value pair: {kv_str}")
            raise

    def kv_to_param_f(self, kv_str):
        return self.kv_to_param(kv_str, soft_cast_float)

    def kv_to_param_i(self, kv_str):
        return self.kv_to_param(kv_str, soft_cast_int)

    def load(self, f):
        observations = list()
        for lineno, line in enumerate(f):
            m = re.search(r"\[::\] *([^|]*?) *[|] *([^|]*?) *[|] *(.*)", line)
            if m:
                name_str = m.group(1)
                param_str = m.group(2)
                attr_str = m.group(3)
                try:
                    param = dict(map(self.kv_to_param_i, param_str.split()))
                    attr = dict(map(self.kv_to_param_f, attr_str.split()))
                    observations.append(
                        {
                            "name": name_str,
                            "param": param,
                            "attribute": attr,
                        }
                    )
                except ValueError:
                    logger.warning(
                        f"Error parsing {filename}: invalid key-value pair in line {lineno+1}",
                        file=sys.stderr,
                    )
                    logger.warning(f"Offending entry:\n{line}", file=sys.stderr)
                    raise

        return observations

    def dump(self, observations, f):
        for observation in observations:
            name = observation["name"]
            param = observation["param"]
            attr = observation["attribute"]

            param_str = " ".join(
                map(
                    lambda kv: f"{kv[0]}={kv[1]}",
                    sorted(param.items(), key=lambda kv: kv[0]),
                )
            )
            attr_str = " ".join(
                map(
                    lambda kv: f"{kv[0]}={kv[1]}",
                    sorted(attr.items(), key=lambda kv: kv[0]),
                )
            )

            print(f"[::] {name} | {param_str} | {attr_str}", file=f)


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


def remove_indexes_from_tuple(parameters, indexes):
    """
    Remove the elements at `indexes` from tuple `parameters`.

    :param parameters: tuple
    :param indexes: list or tuple: indexes of element which are to be removed
    :returns: parameters tuple without the elements at indexes
    """
    indexes = sorted(indexes)
    ret = list()
    last_index = 0
    for index in indexes:
        ret.extend(parameters[last_index:index])
        last_index = index + 1
    ret.extend(parameters[last_index:])
    return tuple(ret)


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


def param_to_ndarray(
    param_tuples, with_nan=True, categorical_to_scalar=False, ignore_indexes=list()
):
    has_nan = dict()
    has_non_numeric = dict()
    distinct_values = dict()
    category_to_scalar = dict()

    logger.debug(
        f"converting param_to_ndarray(with_nan={with_nan}, categorical_to_scalar={categorical_to_scalar}, ignore_indexes={ignore_indexes})"
    )

    for param_tuple in param_tuples:
        for i, param in enumerate(param_tuple):
            if not is_numeric(param):
                if param is None:
                    has_nan[i] = True
                else:
                    has_non_numeric[i] = True
                if categorical_to_scalar and param is not None:
                    if not i in distinct_values:
                        distinct_values[i] = set()
                    distinct_values[i].add(param)

    for i, paramset in distinct_values.items():
        distinct_values[i] = sorted(paramset)
        category_to_scalar[i] = dict()
        for j, param_value in enumerate(distinct_values[i]):
            category_to_scalar[i][param_value] = j

    ignore_index = dict()
    for i in range(len(param_tuples[0])):
        if has_non_numeric.get(i, False) and not categorical_to_scalar:
            ignore_index[i] = True
        elif not with_nan and has_nan.get(i, False):
            ignore_index[i] = True
        else:
            ignore_index[i] = False

    for i in ignore_indexes:
        ignore_index[i] = True

    ret_tuples = list()
    for param_tuple in param_tuples:
        ret_tuple = list()
        for i, param in enumerate(param_tuple):
            if not ignore_index[i]:
                if i in category_to_scalar and not is_numeric(param):
                    ret_tuple.append(category_to_scalar[i][param])
                elif categorical_to_scalar:
                    ret_tuple.append(soft_cast_int(param))
                else:
                    ret_tuple.append(param)
        ret_tuples.append(ret_tuple)
    return np.asarray(ret_tuples), category_to_scalar, ignore_index


def param_dict_to_list(param_dict, parameter_names, default=None):
    """
    Convert {"foo": 1, "bar": 2}, ["bar", "foo", "quux"] to [2, 1, None]
    """
    ret = list()
    for parameter_name in parameter_names:
        ret.append(param_dict.get(parameter_name, None))
    return ret


def observations_enum_to_bool(observations: list, kconfig=False):
    """
    Convert enum / categorical observations to boolean-only ones.
    'observations' is altered in-place.

    DEPRECATED.
    """
    distinct_param_values = dict()
    replace_map = dict()

    for observation in observations:
        for k, v in observation["param"].items():
            if not k in distinct_param_values:
                distinct_param_values[k] = set()
            if v is not None:
                distinct_param_values[k].add(v)

    for param_name, distinct_values in distinct_param_values.items():
        if len(distinct_values) > 2 and not all(
            map(lambda x: x is None or is_numeric(x), distinct_values)
        ):
            replace_map[param_name] = distinct_values

    for observation in observations:
        binary_keys = set()
        for k, v in replace_map.items():
            enum_value = observation["param"].pop(k)
            for binary_key in v:
                if kconfig:
                    if enum_value == binary_key:
                        observation["param"][binary_key] = "y"
                    else:
                        observation["param"][binary_key] = "n"
                else:
                    observation["param"][binary_key] = int(enum_value == binary_key)
                if binary_key in binary_keys:
                    print(f"Error: key '{binary_key}' is not unique")
                binary_keys.add(binary_key)


def ignore_param(by_name: dict, parameter_names: list, ignored_parameters: list):
    ignored_indexes = list()
    unpoppable_params = list()
    for param_name in sorted(ignored_parameters):
        try:
            ignored_indexes.append(parameter_names.index(param_name))
        except ValueError:
            unpoppable_params.append(param_name)

    assert ignored_indexes == sorted(ignored_indexes)
    ignored_indexes = sorted(ignored_indexes, reverse=True)

    for name in by_name:
        for param in by_name[name]["param"]:
            for ignored_index in ignored_indexes:
                param.pop(ignored_index)

    for ignored_index in ignored_indexes:
        parameter_names.pop(ignored_index)

    if unpoppable_params:
        logger.info(
            f"ignore_param: Parameters {unpoppable_params} were not part of the observations to begin with"
        )


def observation_dict_to_by_name(observation):
    parameter_names = observation["param_names"]
    by_name = observation["by_name"]
    assert parameter_names == sorted(parameter_names)
    for name in by_name:
        for entry in by_name[name]["param"]:
            if len(entry) != len(parameter_names):
                logger.error(
                    f"by_name[{name}] has an entry with {len(entry)} parameters. I expect {len(parameter_names)} parameters."
                )
                assert len(entry) == len(parameter_names)
        for attribute in by_name[name]["attributes"]:
            by_name[name][attribute] = np.array(by_name[name][attribute])
    return by_name, parameter_names


def observations_to_by_name(observations):
    """
    Convert observation list to by_name dictionary for AnalyticModel analysis

    :param observations: list of dicts, each representing one measurement. dict keys:
        "name": name of observed state/transition/...
        "param": {"parameter name": parameter value, ...},
        "attribute:" {"attribute name": attribute value, ...}

    :param attributes: observed attributes (i.e., ground truth). Each measurement dict must have an
        entry holding the data value for each attribute. It should not be None.

    :returns: tuple (by_name, parameter_names) which can be passed to AnalyticModel
    """
    if type(observations) is dict:
        return observation_dict_to_by_name(observations)
    parameter_names = set()
    attributes_by_name = dict()
    by_name = dict()
    for observation in observations:
        if observation["name"] not in attributes_by_name:
            attributes_by_name[observation["name"]] = set()
        parameter_names.update(observation["param"].keys())
        attributes_by_name[observation["name"]].update(observation["attribute"].keys())
        name = observation["name"]
        if name not in by_name:
            attributes = list(attributes_by_name[observation["name"]])
            by_name[name] = {"attributes": attributes, "param": list()}
            for attribute in attributes:
                by_name[name][attribute] = list()
    parameter_names = sorted(parameter_names)
    for observation in observations:
        name = observation["name"]
        by_name[name]["param"].append(
            param_dict_to_list(observation["param"], parameter_names)
        )
        for attribute in attributes_by_name[name]:
            if attribute not in observation["attribute"]:
                raise ValueError(
                    f"""Attribute "{attribute}" missing in observation "{name}". Parameters = {observation["param"]}"""
                )
            if observation["attribute"][attribute] is None:
                raise ValueError(
                    f"""Attribute "{attribute}" of observation "{name}" is None. This is not allowed. Parameters = {observation["param"]}"""
                )
            by_name[name][attribute].append(observation["attribute"][attribute])
    for name in by_name:
        for attribute in attributes_by_name[name]:
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


def normalize_nfp_in_aggregate(aggregate, nfp_norm):
    for name in aggregate.keys():
        for new_name, old_name, norm_function in nfp_norm:
            if old_name in aggregate[name]["attributes"]:
                aggregate[name][new_name] = norm_function(aggregate[name].pop(old_name))
                aggregate[name]["attributes"].remove(old_name)
                aggregate[name]["attributes"].append(new_name)


def shift_param_in_aggregate(aggregate, parameters, parameter_shift):
    """
    Remove entries which do not have certain parameter values from `aggregate`.

    :param aggregate: aggregated measurement data, must be a dict conforming to
        aggregate[state or transition name]['param'] = (first parameter value, second parameter value, ...)
        and
        aggregate[state or transition name]['attributes'] = [list of keys with measurement data, e.g. 'power' or 'duration']
    :param parameters: list of parameters, used to map parameter index to parameter name. parameters=['foo', ...] means 'foo' is the first parameter
    :param parameter_shift: [[name, function], [name, function], ...] list of parameter values to alter.
    """
    for param_name, param_shift_function in parameter_shift:
        if param_name == "*":
            for name in aggregate.keys():
                for param_list in aggregate[name]["param"]:
                    for param_index in range(len(param_list)):
                        param_list[param_index] = param_shift_function(
                            param_list[param_index]
                        )
        else:
            param_index = parameters.index(param_name)
            for name in aggregate.keys():
                for param_list in aggregate[name]["param"]:
                    if param_list[param_index] is not None:
                        param_list[param_index] = param_shift_function(
                            param_list[param_index]
                        )


def filter_aggregate_by_observation(aggregate, observation_filter):
    if observation_filter is None:
        return
    to_pop = dict()
    for name in aggregate.keys():
        to_pop[name] = list()
        for attribute in aggregate[name]["attributes"]:
            if (name, attribute) not in observation_filter:
                to_pop[name].append(attribute)
    for name, attributes in to_pop.items():
        for attribute in attributes:
            aggregate[name]["attributes"].remove(attribute)
            aggregate[name].pop(attribute)
        if len(aggregate[name]["attributes"]) == 0:
            aggregate.pop(name)


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
    for param_name, condition, param_value in parameter_filter:
        param_index = parameters.index(param_name)
        param_value = soft_cast_int(param_value)
        names_to_remove = set()

        if condition == "<":
            condf = lambda x: x[param_index] < param_value
        elif condition == "≤":
            condf = lambda x: x[param_index] <= param_value
        elif condition == "=":
            condf = lambda x: x[param_index] == param_value
        elif condition == "≠":
            condf = lambda x: x[param_index] != param_value
        elif condition == "≥":
            condf = lambda x: x[param_index] >= param_value
        elif condition == ">":
            condf = lambda x: x[param_index] > param_value
        elif condition == "∈":
            param_values = tuple(map(soft_cast_int, param_value.split(",")))
            condf = lambda x: x[param_index] in param_values

        for name in aggregate.keys():
            indices_to_keep = list(map(condf, aggregate[name]["param"]))
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


def aggregate_measures(predicted: float, ground_truth: list) -> dict:
    """
    Calculate error measures for a single predicted value compared to a ground truth list

    arguments:
    predicted -- predicted value, i.e. model output (float or int)
    ground_truth -- real-world / reference values (list of float or int)

    return value:
    See regression_measures
    """
    predicted_array = np.array([predicted] * len(ground_truth))
    return regression_measures(predicted_array, np.array(ground_truth))


def regression_measures(predicted: np.ndarray, ground_truth: np.ndarray):
    """
    Calculate error measures by comparing predicted values to ground truth.

    arguments:
    predicted -- model output (np.ndarray)
    ground_truth -- real-world / reference values (np.ndarray)

    Returns a dict containing the following measures:
    mae -- Mean Absolute Error
    mape -- Mean Absolute Percentage Error,
            if all items in ground_truth are non-zero (NaN otherwise)
    smape -- Symmetric Mean Absolute Percentage Error,
             if no 0,0-pairs are present in ground_truth and predicted (NaN otherwise)
    p50 -- Median Absolute Error (as in: the median of the list of absolute
           prediction errors aka. 50th percentile error)
    p90 -- 90th percentile absolute error
    p95 -- 95th percentile absolute error
    p99 -- 99th percentile absolute error
    msd -- Mean Square Deviation
    rmsd -- Root Mean Square Deviation
    ssr -- Sum of Squared Residuals
    rsq -- R^2 measure, see sklearn.metrics.r2_score
    count -- Number of values
    """
    if type(predicted) != np.ndarray:
        raise ValueError(
            "first arg ('predicted') must be ndarray, is {}".format(type(predicted))
        )
    if type(ground_truth) != np.ndarray:
        raise ValueError(
            "second arg ('ground_truth') must be ndarray, is {}".format(
                type(ground_truth)
            )
        )

    deviations = predicted - ground_truth
    if len(deviations) == 0:
        return {}

    p50, p90, p95, p99 = np.percentile(np.abs(deviations), (50, 90, 95, 99))
    measures = {
        "mae": np.mean(np.abs(deviations), dtype=np.float64),
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "msd": np.mean(deviations**2, dtype=np.float64),
        "rmsd": np.sqrt(np.mean(deviations**2), dtype=np.float64),
        "ssr": np.sum(deviations**2, dtype=np.float64),
        "rsq": r2_score(ground_truth, predicted),
        "count": len(ground_truth),
    }

    if np.all(ground_truth != 0):
        # MAPE is generalle considered to be a bad metric
        measures["mape"] = np.mean(np.abs(deviations / ground_truth)) * 100
    else:
        measures["mape"] = np.nan

    if np.all(np.abs(predicted) + np.abs(ground_truth) != 0):
        measures["smape"] = (
            np.mean(
                np.abs(deviations) / ((np.abs(predicted) + np.abs(ground_truth)) / 2)
            )
            * 100
        )
    else:
        measures["smape"] = np.nan

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
