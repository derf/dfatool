import json
import numpy as np
import re
import logging

arg_support_enabled = True
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
        (u"µ", 1e-6),
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
