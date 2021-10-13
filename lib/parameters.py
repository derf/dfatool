#!/usr/bin/env python3
import itertools
import logging
import numpy as np
import os
from collections import OrderedDict
from copy import deepcopy
from multiprocessing import Pool
import dfatool.functions as df
from .paramfit import ParamFit
from .utils import remove_index_from_tuple, is_numeric
from .utils import filter_aggregate_by_param, partition_by_param

logger = logging.getLogger(__name__)


def distinct_param_values(param_tuples):
    """
    Return the distinct values of each parameter in param_tuples.

    E.g. if param_tuples contains the distinct entries (1, 1), (1, 2), (1, 3), (0, 3),
    this function returns [[1, 0], [1, 2, 3]].

    Note that this function deliberately also consider None
    (uninitialized parameter with unknown value) as a distinct value. Benchmarks
    and drivers must ensure that a parameter is only None when its value is
    not important yet, e.g. a packet length parameter must only be None when
    write() or similar has not been called yet. Other parameters should always
    be initialized when leaving UNINITIALIZED.
    """
    distinct_values = [OrderedDict() for i in range(len(param_tuples[0]))]
    for param_tuple in param_tuples:
        for i in range(len(param_tuple)):
            distinct_values[i][param_tuple[i]] = True

    # Convert sets to lists
    distinct_values = list(map(lambda x: list(x.keys()), distinct_values))
    return distinct_values


def _depends_on_param(corr_param, std_param, std_lut):
    # if self.use_corrcoef:
    if False:
        return corr_param > 0.1
    elif std_param == 0:
        # In general, std_param_lut < std_by_param. So, if std_by_param == 0, std_param_lut == 0 follows.
        # This means that the variation of param does not affect the model quality -> no influence
        return False
    return std_lut / std_param < 0.5


def _mean_std_by_param(n_by_param, all_param_values, param_index):
    """
    Calculate the mean standard deviation for a static model where all parameters but `param_index` are constant.

    :param n_by_param: measurements of a specific model attribute partitioned by parameter values.
        Example: `{(0, 2): [2], (0, 4): [4], (0, 6): [6]}`
    :param all_param_values: distinct values of each parameter.
        E.g. for two parameters, the first being None, FOO, or BAR, and the second being 1, 2, 3, or 4, the argument is
        `[[None, 'FOO', 'BAR'], [1, 2, 3, 4]]`.
    :param param_index: index of variable parameter
    :returns: mean stddev
        *mean stddev* is the mean standard deviation of all measurements where parameter `param_index` is dynamic and all other parameters are fixed.
        E.g., if parameters are a, b, c ∈ {1,2,3} and 'index' corresponds to b, then
        this function returns the mean of the standard deviations of (a=1, b=*, c=1),
        (a=1, b=*, c=2), and so on.
    """
    param_values = list(remove_index_from_tuple(all_param_values, param_index))
    partitions = list()

    for param_value in itertools.product(*param_values):
        param_partition = list()
        std_list = list()
        for k, v in n_by_param.items():
            if (*k[:param_index], *k[param_index + 1 :]) == param_value:
                param_partition.extend(v)

        if len(param_partition) > 1:
            partitions.append(param_partition)

    if len(partitions) == 0:
        return 0.0

    return np.mean([np.std(partition) for partition in partitions])


def _corr_by_param(attribute_data, param_values, param_index):
    """
    Return correlation coefficient (`np.corrcoef`) of `attribute_data` <-> `param_values[param_index]`

    A correlation coefficient close to 1 indicates that the attribute likely depends on the value of the parameter denoted by `param_index`, if it is nearly 0, it likely does not depend on it.

    If any value of `param_index` is not numeric (i.e., can not be parsed as float), this function returns 0.

    :param attribute_data: list or 1-D numpy array of measurements
    :param param_values: list of parameter values
    :param param_index: index of parameter in `by_name[*]['param']`
    """
    if _all_params_are_numeric(param_values, param_index):
        param_values = np.array(list((map(lambda x: x[param_index], param_values))))
        try:
            return np.corrcoef(attribute_data, param_values)[0, 1]
        except FloatingPointError:
            # Typically happens when all parameter values are identical.
            # Building a correlation coefficient is pointless in this case
            # -> assume no correlation
            return 0.0
        except ValueError:
            logger.error(
                "ValueError in _corr_by_param(param_index={})".format(param_index)
            )
            logger.error(
                "while executing np.corrcoef({}, {}))".format(
                    attribute_data, param_values
                )
            )
            raise
    else:
        return 0.0


def _compute_param_statistics(
    data,
    param_names,
    param_tuples,
    arg_count=None,
    use_corrcoef=False,
    codependent_params=list(),
):
    """
    Compute standard deviation and correlation coefficient on parameterized data partitions.

    It is strongly recommended to vary all parameter values evenly.
    For instance, given two parameters, providing only the combinations
    (1, 1), (5, 1), (7, 1,) (10, 1), (1, 2), (1, 6) will lead to bogus results.
    It is better to provide (1, 1), (5, 1), (1, 2), (5, 2), ... (i.e. a cross product of all individual parameter values)

    arguments:
    data -- measurement data (ground truth). Must be a list or 1-D numpy array.
    param_names -- list of parameter names
    param_tuples -- list of parameter values corresponding to the order in param_names
    arg_count -- dict providing the number of functions args ("local parameters") for each function.
    use_corrcoef -- use correlation coefficient instead of stddev heuristic for parameter detection

    :returns: a dict with the following content:
    std_static -- static parameter-unaware model error: stddev of data
    std_param_lut -- static parameter-aware model error: mean stddev of data[*]
    std_by_param -- static parameter-aware model error ignoring a single parameter.
        dictionary with one key per parameter. The value is the mean stddev
        of measurements where all other parameters are fixed and the parameter
        in question is variable. E.g. std_by_param['X'] is the mean stddev of
        n_by_param[(X=*, Y=..., Z=...)].
    std_by_arg -- same, but ignoring a single function argument
        Only set if arg_count is non-zero, empty list otherwise.
    corr_by_param -- correlation coefficient
    corr_by_arg -- same, but ignoring a single function argument
        Only set if arg_count is non-zero, empty list otherwise.
    depends_on_param -- dict(parameter_name -> Bool). True if /attribute/ behaviour probably depends on /parameter_name/
    depends_on_arg -- list(bool). Same, but for function arguments, if any.
    """
    ret = dict()

    ret["by_param"] = by_param = partition_by_param(data, param_tuples)

    ret["use_corrcoef"] = use_corrcoef
    ret["_parameter_names"] = param_names

    ret["distinct_values_by_param_index"] = distinct_param_values(param_tuples)

    ret["distinct_values_by_param_name"] = dict()
    for i, param in enumerate(param_names):
        ret["distinct_values_by_param_name"][param] = ret[
            "distinct_values_by_param_index"
        ][i]

    ret["std_static"] = np.std(data)
    # TODO Gewichtung? Parameterkombinationen mit wenig verfügbaren Messdaten werden
    # genau so behandelt wie welchemit vielen verfügbaren Messdaten, in
    # std_static haben sie dagegen weniger Gewicht
    ret["std_param_lut"] = np.mean([np.std(v) for v in by_param.values()])

    ret["std_by_param"] = dict()

    ret["std_by_arg"] = list()

    ret["corr_by_param"] = dict()
    ret["corr_by_arg"] = list()

    ret["_depends_on_param"] = dict()
    ret["_depends_on_arg"] = list()

    np.seterr("raise")

    for param_idx, param in enumerate(param_names):
        if param_idx < len(codependent_params) and codependent_params[param_idx]:
            by_param = partition_by_param(
                data, param_tuples, ignore_parameters=codependent_params[param_idx]
            )
            distinct_values = ret["distinct_values_by_param_index"].copy()
            for codependent_param_index in codependent_params[param_idx]:
                distinct_values[codependent_param_index] = [None]
        else:
            by_param = ret["by_param"]
            distinct_values = ret["distinct_values_by_param_index"]
        mean_std = _mean_std_by_param(by_param, distinct_values, param_idx)
        ret["std_by_param"][param] = mean_std
        ret["corr_by_param"][param] = _corr_by_param(data, param_tuples, param_idx)

        ret["_depends_on_param"][param] = _depends_on_param(
            ret["corr_by_param"][param],
            ret["std_by_param"][param],
            ret["std_param_lut"],
        )

    if arg_count:
        for arg_index in range(arg_count):
            param_idx = len(param_names) + arg_index
            if param_idx < len(codependent_params) and codependent_params[param_idx]:
                by_param = partition_by_param(
                    data, param_tuples, ignore_parameters=codependent_params[param_idx]
                )
                distinct_values = ret["distinct_values_by_param_index"].copy()
                for codependent_param_index in codependent_params[param_idx]:
                    distinct_values[codependent_param_index] = [None]
            else:
                by_param = ret["by_param"]
                distinct_values = ret["distinct_values_by_param_index"]
            mean_std = _mean_std_by_param(by_param, distinct_values, param_idx)
            ret["std_by_arg"].append(mean_std)
            ret["corr_by_arg"].append(_corr_by_param(data, param_tuples, param_idx))

            if False:
                ret["_depends_on_arg"].append(ret["corr_by_arg"][arg_index] > 0.1)
            elif ret["std_by_arg"][arg_index] == 0:
                # In general, std_param_lut < std_by_arg. So, if std_by_arg == 0, std_param_lut == 0 follows.
                # This means that the variation of arg does not affect the model quality -> no influence
                ret["_depends_on_arg"].append(False)
            else:
                ret["_depends_on_arg"].append(
                    ret["std_param_lut"] / ret["std_by_arg"][arg_index] < 0.5
                )

    return ret


def codependent_param_dict(param_values):
    """
    Detect pairs of codependent parameters in param_values.

    The parameter values are first normalized to integer values (e.g. 1, 7, 33 -> 0, 1, 2 and "foo", None, "Hello" -> 0, 1, 2).
    In essence, a pair of parameters (p1, p2) is codepenent if p2 changes only if p1 changes. This is calculated as follows:
    A pair of parameters (p1, p2) is codependent if, for each normalized value of p1, there is only one normalized value of p2 in the set of measurements with
    parameter 1 == p1. Essentially, this means that the mean standard deviation of parameter 2 values for each subset of measurements with a constant parameter
    1 value is zero.

    :param param_values: List of parameter values. Each list entry contains a list of parameter values for one measurement:
        ((param 1 value 1, param 2 value 1, ...), (param 1 value 2, param 2 value 2, ...), ...)
    :returns: dict of codependent parameter pairs. dict[(param 1 index, param 2 index)] is True iff param 1 and param 2 are codependent.
    """
    lut = [dict() for i in param_values[0]]
    for param_index in range(len(param_values[0])):
        uniqs = set(map(lambda param_tuple: param_tuple[param_index], param_values))
        for uniq_index, uniq in enumerate(uniqs):
            lut[param_index][uniq] = uniq_index

    normed_param_values = list()
    for param_tuple in param_values:
        normed_param_values.append(
            tuple(map(lambda ipv: lut[ipv[0]][ipv[1]], enumerate(param_tuple)))
        )

    normed_param_values = np.array(normed_param_values)

    std_by_param = list()
    std_by_param_pair = dict()

    ret = dict()

    for param1_i in range(len(lut)):
        std_by_param.append(np.std(normed_param_values[:, param1_i]))
        for param2_i in range(param1_i + 1, len(lut)):
            stds = list()
            for param1_value in range(len(lut[param1_i])):
                tt = normed_param_values[:, param1_i] == param1_value
                values = normed_param_values[tt, param2_i]
                if len(values) <= 1:
                    stds.append(0.0)
                else:
                    stds.append(np.std(values))
            std_by_param_pair[(param1_i, param2_i)] = np.mean(stds)

    for param1_i in range(len(lut)):
        for param2_i in range(param1_i + 1, len(lut)):
            if std_by_param[param1_i] > 0 and std_by_param[param2_i] > 0:
                if std_by_param_pair[(param1_i, param2_i)] == 0:
                    ret[(param1_i, param2_i)] = True

    return ret


def _compute_param_statistics_parallel(arg):
    return {"key": arg["key"], "dict": _compute_param_statistics(*arg["args"])}


def _all_params_are_numeric(data, param_idx):
    """Check if all `data['param'][*][param_idx]` elements are numeric, as reported by `utils.is_numeric`."""
    param_values = list(map(lambda x: x[param_idx], data))
    return all(map(is_numeric, param_values))


class ParallelParamStats:
    def __init__(self):
        self.queue = list()
        self.map = dict()

    def enqueue(self, key, attr):
        """
        Enqueue data series for statistics calculation.

        :param key: entry key used for retrieval. attr is stored in self.map[key]
            and extended with "by_param" and "stats" attributes once compute() has been called.
        :param attr: ModelAttribute instance. Edited in-place by compute()
        """
        self.queue.append(
            {
                "key": key,
                "args": [
                    attr.data,
                    attr.param_names,
                    attr.param_values,
                    attr.arg_count,
                    False,
                    attr.codependent_params,
                ],
            }
        )
        self.map[key] = attr

    def compute(self):
        """
        Compute statistics for previously enqueue ModelAttribute data.

        Statistics are computed in parallel with one process per core. Results are written to each ModelAttribute wich was passed via enqueue().
        """
        with Pool() as pool:
            results = pool.map(_compute_param_statistics_parallel, self.queue)

        for result in results:
            self.map[result["key"]].by_param = result["dict"].pop("by_param")
            self.map[result["key"]].stats = ParamStats(result["dict"])


class ParamStats:
    """
    Statistics.

    :var std_static: static parameter-unaware model error (standard deviation of raw data)
    :var std_param_lut:  static parameter-aware model error (mean standard deviation of data partitioned by parameter)
    :var std_by_param: static parameter-aware model error ignoring a single parameter.
        dict mapping parameter name -> mean std of data partitioned so that all parameters but "parameter name" are constant.
    :var sty_by_arg: list, argument index -> mean std of data partition so that all parameters but "argument index" are constant
    :var _depends_on_param: dict, parameter name -> bool, True if the modeled behaviour likely depends on parameter name
    :var _depends_on_arg: list, argument index -> bool, True if the modeled behaviour likely depends on argument index
    """

    def __init__(self, data):
        self.__dict__.update(data)

    @classmethod
    def compute_for_attr(cls, attr, use_corrcoef=False):
        res = _compute_param_statistics(
            attr.data,
            attr.param_names,
            attr.param_values,
            arg_count=attr.arg_count,
            use_corrcoef=use_corrcoef,
            codependent_params=attr.codependent_params,
        )
        attr.by_param = res.pop("by_param")
        attr.stats = cls(res)

    def can_be_fitted(self) -> bool:
        """
        Return whether a sufficient amount of distinct numeric parameter values is available, allowing a parameter-aware model to be generated.
        """
        for param in self._parameter_names:
            if (
                len(
                    list(
                        filter(
                            lambda n: is_numeric(n),
                            self.distinct_values_by_param_name[param],
                        )
                    )
                )
                > 2
            ):
                logger.debug(
                    "can be fitted for param {} on {}".format(
                        param,
                        list(
                            filter(
                                lambda n: is_numeric(n),
                                self.distinct_values_by_param_name[param],
                            )
                        ),
                    )
                )
                return True
        return False

    def _generic_param_independence_ratio(self):
        """
        Return the heuristic ratio of parameter independence.

        This is not supported if the correlation coefficient is used.
        A value close to 1 means no influence, a value close to 0 means high probability of influence.
        """
        if self.use_corrcoef:
            # not supported
            raise ValueError
        if self.std_static == 0:
            return 0
        return self.std_param_lut / self.std_static

    def generic_param_dependence_ratio(self):
        """
        Return the heuristic ratio of parameter dependence.

        This is not supported if the correlation coefficient is used.
        A value close to 0 means no influence, a value close to 1 means high probability of influence.
        """
        return 1 - self._generic_param_independence_ratio()

    def _param_independence_ratio(self, param: str) -> float:
        """
        Return the heuristic ratio of parameter independence for param.

        A value close to 1 means no influence, a value close to 0 means high probability of influence.
        """
        if self.use_corrcoef:
            return 1 - np.abs(self.corr_by_param[param])
        if self.std_by_param[param] == 0:
            # if self.std_param_lut != 0:
            #    raise RuntimeError(f"wat: std_by_param[{param}]==0, but std_param_lut=={self.std_param_lut} ≠ 0")
            # In general, std_param_lut < std_by_param. So, if std_by_param == 0, std_param_lut == 0 follows.
            # This means that the variation of param does not affect the model quality -> no influence, return 1
            return 1.0

        return self.std_param_lut / self.std_by_param[param]

    def param_dependence_ratio(self, param: str) -> float:
        """
        Return the heuristic ratio of parameter dependence for param.

        A value close to 0 means no influence, a value close to 1 means high probability of influence.

        :param param: parameter name

        :returns: parameter dependence (float between 0 == no influence and 1 == high probability of influence)
        """
        return 1 - self._param_independence_ratio(param)

    def _arg_independence_ratio(self, arg_index):
        if self.use_corrcoef:
            return 1 - np.abs(self.corr_by_arg[arg_index])
        if self.std_by_arg[arg_index] == 0:
            if self.std_param_lut != 0:
                raise RuntimeError(
                    f"wat: std_by_arg[{arg_index}]==0, but std_param_lut=={self.std_param_lut} ≠ 0"
                )
            # In general, std_param_lut < std_by_arg. So, if std_by_arg == 0, std_param_lut == 0 follows.
            # This means that the variation of arg does not affect the model quality -> no influence, return 1
            return 1
        return self.std_param_lut / self.std_by_arg[arg_index]

    def arg_dependence_ratio(self, arg_index: int) -> float:
        return 1 - self._arg_independence_ratio(arg_index)

    # This heuristic is very similar to the "function is not much better than
    # median" checks in get_fitted. So far, doing it here as well is mostly
    # a performance and not an algorithm quality decision.
    # --df, 2018-04-18
    def depends_on_param(self, param):
        """Return whether attribute of state_or_trans depens on param."""
        return self._depends_on_param[param]

    # See notes on depends_on_param
    def depends_on_arg(self, arg_index):
        """Return whether attribute of state_or_trans depens on arg_index."""
        return self._depends_on_arg[arg_index]


class ModelAttribute:
    """
    A ModelAttribute instance handles a single model attribute, e.g. TX state power or something() function call duration, and corresponding benchmark data.

    It provides three models:
    - a static model (`mean`, `median`) as lower bound of model accuracy
    - a LUT model (`by_param`) as upper bound of model accuracy
    - a fitted model (`model_function`, a `ModelFunction` instance)
    """

    def __init__(
        self,
        name,
        attr,
        data,
        param_values,
        param_names,
        arg_count=0,
        codependent_param=dict(),
    ):

        # Data for model generation
        self.data = np.array(data)

        # Meta data
        self.name = name
        self.attr = attr
        self.param_values = param_values
        self.param_names = sorted(param_names)
        self.arg_count = arg_count

        self.log_param_names = self.param_names + list(
            map(lambda i: f"arg{i}", range(arg_count))
        )

        # Co-dependent parameters. If (param1_index, param2_index) in codependent_param, they are codependent.
        # In this case, only one of them must be used for parameter-dependent model attribute detection and modeling
        self.codependent_param_pair = codependent_param
        self.codependent_params = [list() for x in self.log_param_names]
        self.ignore_param = dict()

        # Static model used as lower bound of model accuracy
        if data is not None:
            self.mean = np.mean(data)
            self.median = np.median(data)
        else:
            self.mean = None
            self.median = None

        # LUT model used as upper bound of model accuracy
        self.by_param = None  # set via ParallelParamStats

        # Split (decision tree) information
        self.split = None

        # param model override
        self.function_override = None

        # The best model we have. May be Static, Split, or Param (and later perhaps Substate)
        self.model_function = None

        self._check_codependent_param()

        # There must be at least 3 distinct data values (≠ None) if an analytic model
        # is to be fitted. For 2 (or less) values, decision trees are better.
        self.min_values_for_analytic_model = 3

    def __repr__(self):
        mean = np.mean(self.data)
        return f"ModelAttribute<{self.name}, {self.attr}, mean={mean}>"

    def to_json(self, **kwargs):
        ret = {
            "paramNames": self.param_names,
            "argCount": self.arg_count,
            "modelFunction": self.model_function.to_json(**kwargs),
        }
        return ret

    def to_dref(self, unit=None):
        ret = {"mean": (self.mean, unit), "median": (self.median, unit)}
        return ret

    def webconf_function_map(self):
        return self.model_function.webconf_function_map()

    @classmethod
    def from_json(cls, name, attr, data):
        param_names = data["paramNames"]
        arg_count = data["argCount"]

        self = cls(name, attr, None, None, param_names, arg_count)

        self.model_function = df.ModelFunction.from_json(data["modelFunction"])

        return self

    def _check_codependent_param(self):
        for (
            (param1_index, param2_index),
            is_codependent,
        ) in self.codependent_param_pair.items():
            if not is_codependent:
                continue
            param1_values = map(lambda pv: pv[param1_index], self.param_values)
            param1_numeric_count = sum(map(is_numeric, param1_values))
            param2_values = map(lambda pv: pv[param2_index], self.param_values)
            param2_numeric_count = sum(map(is_numeric, param2_values))
            # codependent parameter removal is only sensible for numeric parameters. For others (e.g. enums or boolean kconfig switches), dtree modeling
            # automatically leaves out unimportant parameters.
            if param1_numeric_count >= param2_numeric_count > 0:
                self.ignore_param[param2_index] = True
                self.codependent_params[param1_index].append(param2_index)
                logger.info(
                    f"{self.name} {self.attr}: parameters ({self.log_param_names[param1_index]}, {self.log_param_names[param2_index]}) are codependent. Ignoring {self.log_param_names[param2_index]}"
                )
            elif param2_numeric_count >= param1_numeric_count > 0:
                self.ignore_param[param1_index] = True
                self.codependent_params[param2_index].append(param1_index)
                logger.info(
                    f"{self.name} {self.attr}: parameters ({self.log_param_names[param1_index]}, {self.log_param_names[param2_index]}) are codependent. Ignoring {self.log_param_names[param1_index]}"
                )

    def get_static(self, use_mean=False):
        if use_mean:
            return self.mean
        return self.median

    def get_lut(self, param, use_mean=False):
        if use_mean:
            return np.mean(self.by_param[param])
        return np.median(self.by_param[param])

    def get_data_for_paramfit(self, safe_functions_enabled=False):
        if self.split:
            return self.get_data_for_paramfit_split(
                safe_functions_enabled=safe_functions_enabled
            )
        else:
            return self.get_data_for_paramfit_this(
                safe_functions_enabled=safe_functions_enabled
            )

    def get_data_for_paramfit_split(self, safe_functions_enabled=False):
        # currently unused
        split_param_index, child_by_param_value = self.split
        ret = list()
        for param_value, child in child_by_param_value.items():
            child_ret = child.get_data_for_paramfit(
                safe_functions_enabled=safe_functions_enabled
            )
            for key, param, args, kwargs in child_ret:
                ret.append((key[:2] + (param_value,) + key[2:], param, args, kwargs))
        return ret

    def _by_param_for_index(self, param_index):
        if not self.codependent_params[param_index]:
            return self.by_param
        new_param_values = list()
        for param_tuple in self.param_values:
            for i in self.codependent_params[param_index]:
                param_tuple[i] = None
            new_param_values.append(param_tuple)
        return partition_by_param(self.data, new_param_values)

    def depends_on_any_param(self):
        for param_index, param_name in enumerate(self.param_names):
            if (
                self.stats.depends_on_param(param_name)
                and not param_index in self.ignore_param
            ):
                return True
        return False

    def all_relevant_parameters_are_none_or_numeric(self):
        for param_index, param_name in enumerate(self.param_names):
            if (
                self.stats.depends_on_param(param_name)
                and not param_index in self.ignore_param
            ):
                param_values = list(map(lambda x: x[param_index], self.param_values))
                if not all(map(lambda n: n is None or is_numeric(n), param_values)):
                    return False
                distinct_values = self.stats.distinct_values_by_param_index[param_index]
                if (
                    None in distinct_values
                    and len(distinct_values) - 1 < self.min_values_for_analytic_model
                ) or len(distinct_values) < self.min_values_for_analytic_model:
                    return False
        return True

    def get_data_for_paramfit_this(self, safe_functions_enabled=False):
        ret = list()
        for param_index, param_name in enumerate(self.param_names):
            if (
                self.stats.depends_on_param(param_name)
                and not param_index in self.ignore_param
            ):
                by_param = self._by_param_for_index(param_index)
                ret.append(
                    (
                        (self.name, self.attr),
                        param_name,
                        (by_param, param_index, safe_functions_enabled),
                        dict(),
                    )
                )
        if self.arg_count:
            for arg_index in range(self.arg_count):
                param_index = len(self.param_names) + arg_index
                if (
                    self.stats.depends_on_arg(arg_index)
                    and not param_index in self.ignore_param
                ):
                    by_param = self._by_param_for_index(param_index)
                    ret.append(
                        (
                            (self.name, self.attr),
                            arg_index,
                            (by_param, param_index, safe_functions_enabled),
                            dict(),
                        )
                    )

        return ret

    def fit_override_function(self):
        function_str = self.function_override
        x = df.AnalyticFunction(
            self.median,
            function_str,
            self.param_names,
            self.arg_count,
            # fit_by_param=fit_result,
        )
        x.fit(self.by_param)
        if x.fit_success:
            self.model_function = x
        else:
            logger.warning(f"Fit of user-defined model function {function_str} failed.")

    def set_data_from_paramfit(self, paramfit, prefix=tuple()):
        if self.split:
            self.set_data_from_paramfit_split(paramfit, prefix)
        else:
            self.set_data_from_paramfit_this(paramfit, prefix)

    def set_data_from_paramfit_split(self, paramfit, prefix):
        # currently unused
        split_param_index, child_by_param_value = self.split
        function_map = {
            "split_by": split_param_index,
            "child": dict(),
            "child_static": dict(),
        }
        function_child = dict()
        info_child = dict()
        for param_value, child in child_by_param_value.items():
            child.set_data_from_paramfit(paramfit, prefix + (param_value,))
            function_child[param_value] = child.model_function
        self.model_function = df.SplitFunction(
            self.median, split_param_index, function_child
        )

    def set_data_from_paramfit_this(self, paramfit, prefix):
        fit_result = paramfit.get_result((self.name, self.attr) + prefix)
        if self.model_function is None:
            self.model_function = df.StaticFunction(self.median)
        if os.getenv("DFATOOL_NO_PARAM"):
            pass
        elif len(fit_result.keys()):
            x = df.analytic.function_powerset(
                fit_result, self.param_names, self.arg_count
            )
            x.value = self.median
            x.fit(self.by_param)

            if x.fit_success:
                self.model_function = x

    def build_dtree(self, parameters, data, with_function_leaves=False, threshold=100):
        """
        Build a Decision Tree on `param` / `data` for kconfig models.

        :param this_symbols: parameter names
        :param this_data: list of measurements. Each entry is a (param vector, mearusements vector) tuple.
            param vector holds parameter values (same order as parameter names). mearuserements vector holds measurements.
        :param data_index: Index in measurements vector to use for model generation. Default 0.
        :param threshold: Return a StaticFunction leaf node if std(data[data_index]) < threshold. Default 100.

        :returns: SplitFunction or StaticFunction
        """
        self.model_function = self._build_dtree(
            parameters, data, with_function_leaves, threshold
        )

    def _build_dtree(
        self, parameters, data, with_function_leaves=False, threshold=100, level=0
    ):
        """
        Build a Decision Tree on `param` / `data` for kconfig models.

        :param this_symbols: parameter names
        :param this_data: list of measurements. Each entry is a (param vector, mearusements vector) tuple.
            param vector holds parameter values (same order as parameter names). mearuserements vector holds measurements.
        :param data_index: Index in measurements vector to use for model generation. Default 0.
        :param threshold: Return a StaticFunction leaf node if std(data[data_index]) < threshold. Default 100.

        :returns: SplitFunction or StaticFunction
        """

        # TODO remove data entries which are None (and remove corresponding parameters, too!)

        param_count = len(self.param_names) + self.arg_count
        if param_count == 0 or np.std(data) < threshold:
            return df.StaticFunction(np.mean(data))
            # sf.value_error["std"] = np.std(data)

        mean_stds = list()
        for param_index in range(param_count):

            if param_index in self.ignore_param:
                mean_stds.append(np.inf)
                continue

            unique_values = list(set(map(lambda p: p[param_index], parameters)))

            if None in unique_values:
                # param is a choice and undefined in some configs. Do not split on it.
                mean_stds.append(np.inf)
                continue

            if (
                with_function_leaves
                and len(unique_values) >= self.min_values_for_analytic_model
                and all(map(lambda x: type(x) is int, unique_values))
            ):
                # param can be modeled as a function. Do not split on it.
                mean_stds.append(np.inf)
                continue

            child_indexes = list()
            for value in unique_values:
                child_indexes.append(
                    list(
                        filter(
                            lambda i: parameters[i][param_index] == value,
                            range(len(parameters)),
                        )
                    )
                )

            if len(list(filter(len, child_indexes))) <= 1:
                # this param only has a single value. there's no point in splitting.
                mean_stds.append(np.inf)
                continue

            children = list()
            for child in child_indexes:
                children.append(np.std(list(map(lambda i: data[i], child))))

            if np.any(np.isnan(children)):
                mean_stds.append(np.inf)
            else:
                mean_stds.append(np.mean(children))

        if np.all(np.isinf(mean_stds)):
            # all children have the same configuration. We shouldn't get here due to the threshold check above...
            if with_function_leaves:
                # try generating a function. if it fails, model_function is a StaticFunction.
                ma = ModelAttribute(
                    "tmp",
                    "tmp",
                    data,
                    parameters,
                    self.param_names,
                    arg_count=self.arg_count,
                )
                ParamStats.compute_for_attr(ma)
                paramfit = ParamFit(parallel=False)
                for key, param, args, kwargs in ma.get_data_for_paramfit():
                    paramfit.enqueue(key, param, args, kwargs)
                paramfit.fit()
                ma.set_data_from_paramfit(paramfit)
                return ma.model_function
            # else:
            #    logging.warning(
            #        f"While building DTree for configurations {parameters}: Children have identical configuration, but high stddev ({np.std(data)}). Falling back to Staticfunction"
            #    )
            return df.StaticFunction(np.mean(data))

        symbol_index = np.argmin(mean_stds)
        unique_values = list(set(map(lambda p: p[symbol_index], parameters)))

        child = dict()

        for value in unique_values:
            indexes = list(
                filter(
                    lambda i: parameters[i][symbol_index] == value,
                    range(len(parameters)),
                )
            )
            child_parameters = list(map(lambda i: parameters[i], indexes))
            child_data = list(map(lambda i: data[i], indexes))
            if len(child_data):
                child[value] = self._build_dtree(
                    child_parameters,
                    child_data,
                    with_function_leaves,
                    threshold,
                    level + 1,
                )

        assert len(child.values()) >= 2

        return df.SplitFunction(np.mean(data), symbol_index, child)
