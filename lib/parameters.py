#!/usr/bin/env python3
import itertools
import logging
import numpy as np
import os
import warnings
from collections import OrderedDict
from copy import deepcopy
from multiprocessing import Pool
import dfatool.functions as df
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


def _reduce_param_matrix(matrix: np.ndarray, parameter_names: list) -> list:
    """
    :param matrix: parameter dependence matrix, M[(...)] == 1 iff (model attribute) is influenced by (parameter) for other parameter value indxe == (...)
    :param parameter_names: names of parameters in the order in which they appear in the matrix index. The first entry corresponds to the first axis, etc.
    :returns: parameters which determine whether (parameter) has an effect on (model attribute). If a parameter is not part of this list, its value does not
        affect (parameter)'s influence on (model attribute) -- it either always or never has an influence
    """
    if np.all(matrix == True) or np.all(matrix == False):
        return list()

    # Diese Abbruchbedingung scheint noch nicht so schlau zu sein...
    # Mit wird zu viel rausgefiltert (z.B. auto_ack! -> max_retry_count in "bin/analyze-timing.py ../data/20190815_122531_nRF24_no-rx.json" nicht erkannt)
    # Ohne wird zu wenig rausgefiltert (auch ganz viele Abhängigkeiten erkannt, bei denen eine Parameter-Abhängigketi immer unabhängig vom Wert der anderen Parameter besteht)
    # if not is_power_of_two(np.count_nonzero(matrix)):
    #    # cannot be reliably reduced to a list of parameters
    #    return list()

    if np.count_nonzero(matrix) == 1:
        influential_parameters = list()
        for i, parameter_name in enumerate(parameter_names):
            if matrix.shape[i] > 1:
                influential_parameters.append(parameter_name)
        return influential_parameters

    for axis in range(matrix.ndim):
        candidate = _reduce_param_matrix(
            np.all(matrix, axis=axis), remove_index_from_tuple(parameter_names, axis)
        )
        if len(candidate):
            return candidate

    return list()


def _std_by_param(n_by_param, all_param_values, param_index):
    """
    Calculate standard deviations for a static model where all parameters but `param_index` are constant.

    :param n_by_param: measurements of a specific model attribute partitioned by parameter values.
        Example: `{(0, 2): [2], (0, 4): [4], (0, 6): [6]}`
    :param all_param_values: distinct values of each parameter.
        E.g. for two parameters, the first being None, FOO, or BAR, and the second being 1, 2, 3, or 4, the argument is
        `[[None, 'FOO', 'BAR'], [1, 2, 3, 4]]`.
    :param param_index: index of variable parameter
    :returns: (stddev matrix, mean stddev, LUT matrix)
        *stddev matrix* is an ((number of parameters)-1)-dimensional matrix giving the standard deviation of each individual parameter variation partition.
        E.g. for param_index == 2 and 4 parameters, stddev matrix[a][b][d] is the stddev of
        measurements with param0 == all_param_values[0][a],
        param1 == all_param_values[1][b], param2 variable, and
        param3 == all_param_values[3][d].
        *mean stddev* is the mean standard deviation of all measurements where parameter `param_index` is dynamic and all other parameters are fixed.
        E.g., if parameters are a, b, c ∈ {1,2,3} and 'index' corresponds to b, then
        this function returns the mean of the standard deviations of (a=1, b=*, c=1),
        (a=1, b=*, c=2), and so on.
        *LUT matrix* is an ((number of parameters)-1)-dimensional matrix giving the mean standard deviation of individual partitions with entirely constant parameters.
        E.g. for param_index == 2 and 4 parameters, LUT matrix[a][b][d] is the mean of
        stddev(param0 -> a, param1 -> b, param2 -> first distinct value, param3 -> d),
        stddev(param0 -> a, param1 -> b, param2 -> second distinct value, param3 -> d),
        and so on.
    """
    param_values = list(remove_index_from_tuple(all_param_values, param_index))
    info_shape = tuple(map(len, param_values))

    # We will calculate the mean over the entire matrix later on. As we cannot
    # guarantee that each entry will be filled in this loop (e.g. transitions
    # whose arguments are combined using 'zip' rather than 'cartesian' always
    # have missing parameter combinations), we pre-fill it with NaN and use
    # np.nanmean to skip those when calculating the mean.
    stddev_matrix = np.full(info_shape, np.nan)
    lut_matrix = np.full(info_shape, np.nan)

    for param_value in itertools.product(*param_values):
        param_partition = list()
        std_list = list()
        for k, v in n_by_param.items():
            if (*k[:param_index], *k[param_index + 1 :]) == param_value:
                param_partition.extend(v)
                std_list.append(np.std(v))

        if len(param_partition) > 1:
            matrix_index = list(range(len(param_value)))
            for i in range(len(param_value)):
                matrix_index[i] = param_values[i].index(param_value[i])
            matrix_index = tuple(matrix_index)
            stddev_matrix[matrix_index] = np.std(param_partition)
            lut_matrix[matrix_index] = np.mean(std_list)
        # This can (and will) happen in normal operation, e.g. when a transition's
        # arguments are combined using 'zip' rather than 'cartesian'.
        # elif len(param_partition) == 1:
        #    vprint(verbose, '[W] parameter value partition for {} contains only one element -- skipping'.format(param_value))
        # else:
        #    vprint(verbose, '[W] parameter value partition for {} is empty'.format(param_value))

    if np.all(np.isnan(stddev_matrix)):
        warnings.warn(
            "parameter #{} has no data partitions. stddev_matrix = {}".format(
                param_index, stddev_matrix
            )
        )
        return stddev_matrix, 0.0

    return (
        stddev_matrix,
        np.nanmean(stddev_matrix),
        lut_matrix,
    )  # np.mean([np.std(partition) for partition in partitions])


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
    data, param_names, param_tuples, arg_count=None, use_corrcoef=False
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
    ret["std_by_param_values"] = dict()
    ret["lut_by_param_values"] = dict()

    ret["std_by_arg"] = list()
    ret["std_by_arg_values"] = list()
    ret["lut_by_arg_values"] = list()

    ret["corr_by_param"] = dict()
    ret["corr_by_arg"] = list()

    ret["_depends_on_param"] = dict()
    ret["_depends_on_arg"] = list()

    np.seterr("raise")

    for param_idx, param in enumerate(param_names):
        std_matrix, mean_std, lut_matrix = _std_by_param(
            by_param, ret["distinct_values_by_param_index"], param_idx
        )
        ret["std_by_param"][param] = mean_std
        ret["std_by_param_values"][param] = std_matrix
        ret["lut_by_param_values"][param] = lut_matrix
        ret["corr_by_param"][param] = _corr_by_param(data, param_tuples, param_idx)

        ret["_depends_on_param"][param] = _depends_on_param(
            ret["corr_by_param"][param],
            ret["std_by_param"][param],
            ret["std_param_lut"],
        )

    if arg_count:
        for arg_index in range(arg_count):
            std_matrix, mean_std, lut_matrix = _std_by_param(
                by_param,
                ret["distinct_values_by_param_index"],
                len(param_names) + arg_index,
            )
            ret["std_by_arg"].append(mean_std)
            ret["std_by_arg_values"].append(std_matrix)
            ret["lut_by_arg_values"].append(lut_matrix)
            ret["corr_by_arg"].append(
                _corr_by_param(data, param_tuples, len(param_names) + arg_index)
            )

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


def _compute_param_statistics_parallel(arg):
    return {"key": arg["key"], "dict": _compute_param_statistics(*arg["args"])}


def _all_params_are_numeric(data, param_idx):
    """Check if all `data['param'][*][param_idx]` elements are numeric, as reported by `utils.is_numeric`."""
    param_values = list(map(lambda x: x[param_idx], data))
    if len(list(filter(is_numeric, param_values))) == len(param_values):
        return True
    return False


def prune_dependent_parameters(by_name, parameter_names, correlation_threshold=0.5):
    """
    Remove dependent parameters from aggregate.

    :param by_name: measurements partitioned by state/transition/... name and attribute, edited in-place.
        by_name[name][attribute] must be a list or 1-D numpy array.
        by_name[stanamete_or_trans]['param'] must be a list of parameter values.
        Other dict members are left as-is
    :param parameter_names: List of parameter names in the order they are used in by_name[name]['param'], edited in-place.
    :param correlation_threshold: Remove parameter if absolute correlation exceeds this threshold (default: 0.5)

    Model generation (and its components, such as relevant parameter detection and least squares optimization) only works if input variables (i.e., parameters)
    are independent of each other. This function computes the correlation coefficient for each pair of parameters and removes those which depend on each other.
    For each pair of dependent parameters, the lexically greater one is removed (e.g. "a" and "b" -> "b" is removed).
    """

    parameter_indices_to_remove = list()
    for parameter_combination in itertools.product(
        range(len(parameter_names)), range(len(parameter_names))
    ):
        index_1, index_2 = parameter_combination
        if index_1 >= index_2:
            continue
        parameter_values = [list(), list()]  # both parameters have a value
        parameter_values_1 = list()  # parameter 1 has a value
        parameter_values_2 = list()  # parameter 2 has a value
        for name in by_name:
            for measurement in by_name[name]["param"]:
                value_1 = measurement[index_1]
                value_2 = measurement[index_2]
                if is_numeric(value_1):
                    parameter_values_1.append(value_1)
                if is_numeric(value_2):
                    parameter_values_2.append(value_2)
                if is_numeric(value_1) and is_numeric(value_2):
                    parameter_values[0].append(value_1)
                    parameter_values[1].append(value_2)
        if len(parameter_values[0]):
            # Calculating the correlation coefficient only makes sense when neither value is constant
            if np.std(parameter_values_1) != 0 and np.std(parameter_values_2) != 0:
                correlation = np.corrcoef(parameter_values)[0][1]
                if (
                    correlation != np.nan
                    and np.abs(correlation) > correlation_threshold
                ):
                    logger.debug(
                        "Parameters {} <-> {} are correlated with coefficcient {}".format(
                            parameter_names[index_1],
                            parameter_names[index_2],
                            correlation,
                        )
                    )
                    if len(parameter_values_1) < len(parameter_values_2):
                        index_to_remove = index_1
                    else:
                        index_to_remove = index_2
                    logger.debug(
                        "    Removing parameter {}".format(
                            parameter_names[index_to_remove]
                        )
                    )
                    parameter_indices_to_remove.append(index_to_remove)
    remove_parameters_by_indices(by_name, parameter_names, parameter_indices_to_remove)


def remove_parameters_by_indices(by_name, parameter_names, parameter_indices_to_remove):
    """
    Remove parameters listed in `parameter_indices` from aggregate `by_name` and `parameter_names`.

    :param by_name: measurements partitioned by state/transition/... name and attribute, edited in-place.
        by_name[name][attribute] must be a list or 1-D numpy array.
        by_name[stanamete_or_trans]['param'] must be a list of parameter values.
        Other dict members are left as-is
    :param parameter_names: List of parameter names in the order they are used in by_name[name]['param'], edited in-place.
    :param parameter_indices_to_remove: List of parameter indices to be removed
    """

    # Start removal from the end of the list to avoid renumbering of list elemenets
    for parameter_index in sorted(parameter_indices_to_remove, reverse=True):
        for name in by_name:
            for measurement in by_name[name]["param"]:
                measurement.pop(parameter_index)
        parameter_names.pop(parameter_index)


class ParallelParamStats:
    def __init__(self):
        self.queue = list()
        self.map = dict()

    def enqueue(self, key, attr):
        self.queue.append(
            {
                "key": key,
                "args": [
                    attr.data,
                    attr.param_names,
                    attr.param_values,
                    attr.arg_count,
                ],
            }
        )
        self.map[key] = attr

    def compute(self):
        """
        Fit functions on previously enqueue data.

        Fitting is one in parallel with one process per core.

        Results can be accessed using the public ParallelParamFit.results object.
        """
        with Pool() as pool:
            results = pool.map(_compute_param_statistics_parallel, self.queue)

        for result in results:
            self.map[result["key"]].by_param = result["dict"].pop("by_param")
            self.map[result["key"]].stats = ParamStats(result["dict"])


class ParamStats:
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
    def __init__(self, name, attr, data, param_values, param_names, arg_count=0):

        # Data for model generation
        self.data = np.array(data)

        # Meta data
        self.name = name
        self.attr = attr
        self.param_values = param_values
        self.param_names = sorted(param_names)
        self.arg_count = arg_count

        # Static model used as lower bound of model accuracy
        self.mean = np.mean(data)
        self.median = np.median(data)

        # LUT model used as upper bound of model accuracy
        self.by_param = None  # set via ParallelParamStats

        # Split (decision tree) information
        self.split = None

        # param model override
        self.function_override = None

        # The best model we have. May be Static, Split, or Param (and later perhaps Substate)
        self.model_function = None

    def __repr__(self):
        mean = np.mean(self.data)
        return f"ModelAttribute<{self.name}, {self.attr}, mean={mean}>"

    def to_json(self):
        ret = {
            "paramNames": self.param_names,
            "argCount": self.arg_count,
            "modelFunction": self.model_function.to_json(),
        }
        return ret

    @staticmethod
    def from_json(cls, name, attr, data):
        param_names = data["paramNames"]
        arg_count = data["argCount"]

        self = cls(name, attr, None, None, param_names, arg_count)

        self.model_function = df.ModelFunction.from_json(data["modelFunction"])

        return self

    def get_static(self, use_mean=False):
        if use_mean:
            return self.mean
        return self.median

    def get_lut(self, param, use_mean=False):
        if use_mean:
            return np.mean(self.by_param[param])
        return np.median(self.by_param[param])

    def build_dtree(self):
        split_param_index = self.get_split_param_index()
        if split_param_index is None:
            return

        distinct_values = self.stats.distinct_values_by_param_index[split_param_index]
        tt1 = list(
            map(
                lambda i: self.param_values[i][split_param_index] == distinct_values[0],
                range(len(self.param_values)),
            )
        )
        tt2 = np.invert(tt1)

        pv1 = list()
        pv2 = list()

        for i, param_tuple in enumerate(self.param_values):
            if tt1[i]:
                pv1.append(param_tuple)
            else:
                pv2.append(param_tuple)

        # print(
        #    f">>> split {self.name} {self.attr} by param #{split_param_index}"
        # )

        child1 = ModelAttribute(
            self.name, self.attr, self.data[tt1], pv1, self.param_names, self.arg_count
        )
        child2 = ModelAttribute(
            self.name, self.attr, self.data[tt2], pv2, self.param_names, self.arg_count
        )

        ParamStats.compute_for_attr(child1)
        ParamStats.compute_for_attr(child2)

        child1.build_dtree()
        child2.build_dtree()

        self.split = (
            split_param_index,
            {distinct_values[0]: child1, distinct_values[1]: child2},
        )

        # print(
        #    f"<<< split {self.name} {self.attr} by param #{split_param_index}"
        # )

    # None -> kein split notwendig
    # andernfalls: Parameter-Index, anhand dessen eine Decision Tree-Ebene aufgespannt wird
    # (Kinder sind wiederum ModelAttributes, in denen dieser Parameter konstant ist)
    def get_split_param_index(self):
        if not self.param_names:
            return None
        std_by_param = list()
        for param_index, param_name in enumerate(self.param_names):
            distinct_values = self.stats.distinct_values_by_param_index[param_index]
            if self.stats.depends_on_param(param_name) and len(distinct_values) == 2:
                val1 = list(
                    map(
                        lambda i: self.param_values[i][param_index]
                        == distinct_values[0],
                        range(len(self.param_values)),
                    )
                )
                val2 = np.invert(val1)
                val1_std = np.std(self.data[val1])
                val2_std = np.std(self.data[val2])
                std_by_param.append(np.mean([val1_std, val2_std]))
            else:
                std_by_param.append(np.inf)
        for arg_index in range(self.arg_count):
            distinct_values = self.stats.distinct_values_by_param_index[
                len(self.param_names) + arg_index
            ]
            if self.stats.depends_on_arg(arg_index) and len(distinct_values) == 2:
                val1 = list(
                    map(
                        lambda i: self.param_values[i][
                            len(self.param_names) + arg_index
                        ]
                        == distinct_values[0],
                        range(len(self.param_values)),
                    )
                )
                val2 = np.invert(val1)
                val1_std = np.std(self.data[val1])
                val2_std = np.std(self.data[val2])
                std_by_param.append(np.mean([val1_std, val2_std]))
            else:
                std_by_param.append(np.inf)
        split_param_index = np.argmin(std_by_param)
        split_std = std_by_param[split_param_index]
        if split_std == np.inf:
            return None
        return split_param_index

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
        split_param_index, child_by_param_value = self.split
        ret = list()
        for param_value, child in child_by_param_value.items():
            child_ret = child.get_data_for_paramfit(
                safe_functions_enabled=safe_functions_enabled
            )
            for key, param, val in child_ret:
                ret.append((key[:2] + (param_value,) + key[2:], param, val))
        return ret

    def get_data_for_paramfit_this(self, safe_functions_enabled=False):
        ret = list()
        for param_index, param_name in enumerate(self.param_names):
            if self.stats.depends_on_param(param_name):
                ret.append(
                    (
                        (self.name, self.attr),
                        param_name,
                        (self.by_param, param_index, safe_functions_enabled),
                    )
                )
        if self.arg_count:
            for arg_index in range(self.arg_count):
                if self.stats.depends_on_arg(arg_index):
                    ret.append(
                        (
                            (self.name, self.attr),
                            arg_index,
                            (
                                self.by_param,
                                len(self.param_names) + arg_index,
                                safe_functions_enabled,
                            ),
                        )
                    )

        return ret

    def set_data_from_paramfit(self, paramfit, prefix=tuple()):
        if self.split:
            self.set_data_from_paramfit_split(paramfit, prefix)
        else:
            self.set_data_from_paramfit_this(paramfit, prefix)

    def set_data_from_paramfit_split(self, paramfit, prefix):
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
        self.model_function = df.StaticFunction(self.median)
        if self.function_override is not None:
            function_str = self.function_override
            x = df.AnalyticFunction(
                self.median,
                function_str,
                self.param_names,
                self.arg_count,
                fit_by_param=fit_result,
            )
            x.fit(self.by_param)
            if x.fit_success:
                self.model_function = x
        elif os.getenv("DFATOOL_NO_PARAM"):
            pass
        elif len(fit_result.keys()):
            x = df.analytic.function_powerset(
                fit_result, self.param_names, self.arg_count
            )
            x.value = self.median
            x.fit(self.by_param)

            if x.fit_success:
                self.model_function = x
