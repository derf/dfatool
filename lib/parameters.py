import itertools
import logging
import numpy as np
import warnings
from collections import OrderedDict
from copy import deepcopy
from multiprocessing import Pool
from .utils import remove_index_from_tuple, is_numeric
from .utils import filter_aggregate_by_param, by_name_to_by_param

logger = logging.getLogger(__name__)


def distinct_param_values(by_name, state_or_tran):
    """
    Return the distinct values of each parameter in by_name[state_or_tran].

    E.g. if by_name[state_or_tran]['param'] contains the distinct entries (1, 1), (1, 2), (1, 3), (0, 3),
    this function returns [[1, 0], [1, 2, 3]].

    Note that this function deliberately also consider None
    (uninitialized parameter with unknown value) as a distinct value. Benchmarks
    and drivers must ensure that a parameter is only None when its value is
    not important yet, e.g. a packet length parameter must only be None when
    write() or similar has not been called yet. Other parameters should always
    be initialized when leaving UNINITIALIZED.
    """
    distinct_values = [
        OrderedDict() for i in range(len(by_name[state_or_tran]["param"][0]))
    ]
    for param_tuple in by_name[state_or_tran]["param"]:
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


def _std_by_param(by_param, all_param_values, state_or_tran, attribute, param_index):
    u"""
    Calculate standard deviations for a static model where all parameters but `param_index` are constant.

    :param by_param: measurements sorted by key/transition name and parameter values
    :param all_param_values: distinct values of each parameter in `state_or_tran`.
        E.g. for two parameters, the first being None, FOO, or BAR, and the second being 1, 2, 3, or 4, the argument is
        `[[None, 'FOO', 'BAR'], [1, 2, 3, 4]]`.
    :param state_or_tran: state or transition name (-> by_param[(state_or_tran, *)])
    :param attribute: model attribute, e.g. 'power' or 'duration'
           (-> by_param[(state_or_tran, *)][attribute])
    :param param_index: index of variable parameter
    :returns: (stddev matrix, mean stddev, LUT matrix)
        *stddev matrix* is an ((number of parameters)-1)-dimensional matrix giving the standard deviation of each individual parameter variation partition.
        E.g. for param_index == 2 and 4 parameters, stddev matrix[a][b][d] is the stddev of
        measurements with param0 == all_param_values[0][a],
        param1 == all_param_values[1][b], param2 variable, and
        param3 == all_param_values[3][d].
        *mean stddev* is the mean standard deviation of all measurements of `attribute`
        for `state_or_tran` where parameter `param_index` is dynamic and all other parameters are fixed.
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
        for k, v in by_param.items():
            if (
                k[0] == state_or_tran
                and (*k[1][:param_index], *k[1][param_index + 1 :]) == param_value
            ):
                param_partition.extend(v[attribute])
                std_list.append(np.std(v[attribute]))

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
            "{}/{} parameter #{} has no data partitions. stddev_matrix = {}".format(
                state_or_tran, attribute, param_index, stddev_matrix
            )
        )
        return stddev_matrix, 0.0

    return (
        stddev_matrix,
        np.nanmean(stddev_matrix),
        lut_matrix,
    )  # np.mean([np.std(partition) for partition in partitions])


def _corr_by_param(by_name, state_or_trans, attribute, param_index):
    """
    Return correlation coefficient (`np.corrcoef`) of `by_name[state_or_trans][attribute][:]` <-> `by_name[state_or_trans]['param'][:][param_index]`

    A correlation coefficient close to 1 indicates that the attribute likely depends on the value of the parameter denoted by `param_index`, if it is nearly 0, it likely does not depend on it.

    If any value of `param_index` is not numeric (i.e., can not be parsed as float), this function returns 0.

    :param by_name: measurements partitioned by state/transition name
    :param state_or_trans: state or transition name
    :param attribute: model attribute
    :param param_index: index of parameter in `by_name[state_or_trans]['param']`
    """
    if _all_params_are_numeric(by_name[state_or_trans], param_index):
        param_values = np.array(
            list((map(lambda x: x[param_index], by_name[state_or_trans]["param"])))
        )
        try:
            return np.corrcoef(by_name[state_or_trans][attribute], param_values)[0, 1]
        except FloatingPointError:
            # Typically happens when all parameter values are identical.
            # Building a correlation coefficient is pointless in this case
            # -> assume no correlation
            return 0.0
        except ValueError:
            logger.error(
                "ValueError in _corr_by_param(by_name, state_or_trans={}, attribute={}, param_index={})".format(
                    state_or_trans, attribute, param_index
                )
            )
            logger.error(
                "while executing np.corrcoef(by_name[{}][{}]={}, {}))".format(
                    state_or_trans,
                    attribute,
                    by_name[state_or_trans][attribute],
                    param_values,
                )
            )
            raise
    else:
        return 0.0


def _compute_param_statistics(
    by_name,
    by_param,
    parameter_names,
    arg_count,
    state_or_trans,
    attribute,
    distinct_values,
    distinct_values_by_param_index,
):
    """
    Compute standard deviation and correlation coefficient for various data partitions.

    It is strongly recommended to vary all parameter values evenly across partitions.
    For instance, given two parameters, providing only the combinations
    (1, 1), (5, 1), (7, 1,) (10, 1), (1, 2), (1, 6) will lead to bogus results.
    It is better to provide (1, 1), (5, 1), (1, 2), (5, 2), ... (i.e. a cross product of all individual parameter values)

    :param by_name: ground truth partitioned by state/transition name.
        by_name[state_or_trans][attribute] must be a list or 1-D numpy array.
        by_name[state_or_trans]['param'] must be a list of parameter values
        corresponding to the ground truth, e.g. [[1, 2, 3], ...] if the
        first ground truth element has the (lexically) first parameter set to 1,
        the second to 2 and the third to 3.
    :param by_param: ground truth partitioned by state/transition name and parameters.
        by_name[(state_or_trans, *)][attribute] must be a list or 1-D numpy array.
    :param parameter_names: list of parameter names, must have the same order as the parameter
        values in by_param (lexical sorting is recommended).
    :param arg_count: dict providing the number of functions args ("local parameters") for each function.
    :param state_or_trans: state or transition name, e.g. 'send' or 'TX'
    :param attribute: model attribute, e.g. 'power' or 'duration'

    :returns: a dict with the following content:
    std_static -- static parameter-unaware model error: stddev of by_name[state_or_trans][attribute]
    std_param_lut -- static parameter-aware model error: mean stddev of by_param[(state_or_trans, *)][attribute]
    std_by_param -- static parameter-aware model error ignoring a single parameter.
        dictionary with one key per parameter. The value is the mean stddev
        of measurements where all other parameters are fixed and the parameter
        in question is variable. E.g. std_by_param['X'] is the mean stddev of
        by_param[(state_or_trans, (X=*, Y=..., Z=...))][attribute].
    std_by_arg -- same, but ignoring a single function argument
        Only set if state_or_trans appears in arg_count, empty dict otherwise.
    corr_by_param -- correlation coefficient
    corr_by_arg -- same, but ignoring a single function argument
        Only set if state_or_trans appears in arg_count, empty dict otherwise.
    depends_on_param -- dict(parameter_name -> Bool). True if /attribute/ behaviour probably depends on /parameter_name/
    depends_on_arg -- list(bool). Same, but for function arguments, if any.
    """
    ret = {
        "std_static": np.std(by_name[state_or_trans][attribute]),
        "std_param_lut": np.mean(
            [
                np.std(by_param[x][attribute])
                for x in by_param.keys()
                if x[0] == state_or_trans
            ]
        ),
        "std_by_param": {},
        "std_by_param_values": {},
        "lut_by_param_values": {},
        "std_by_arg": [],
        "std_by_arg_values": [],
        "lut_by_arg_values": [],
        "corr_by_param": {},
        "corr_by_arg": [],
        "depends_on_param": {},
        "depends_on_arg": [],
    }

    np.seterr("raise")

    for param_idx, param in enumerate(parameter_names):
        std_matrix, mean_std, lut_matrix = _std_by_param(
            by_param,
            distinct_values_by_param_index,
            state_or_trans,
            attribute,
            param_idx,
        )
        ret["std_by_param"][param] = mean_std
        ret["std_by_param_values"][param] = std_matrix
        ret["lut_by_param_values"][param] = lut_matrix
        ret["corr_by_param"][param] = _corr_by_param(
            by_name, state_or_trans, attribute, param_idx
        )

        ret["depends_on_param"][param] = _depends_on_param(
            ret["corr_by_param"][param],
            ret["std_by_param"][param],
            ret["std_param_lut"],
        )

    if state_or_trans in arg_count:
        for arg_index in range(arg_count[state_or_trans]):
            std_matrix, mean_std, lut_matrix = _std_by_param(
                by_param,
                distinct_values_by_param_index,
                state_or_trans,
                attribute,
                len(parameter_names) + arg_index,
            )
            ret["std_by_arg"].append(mean_std)
            ret["std_by_arg_values"].append(std_matrix)
            ret["lut_by_arg_values"].append(lut_matrix)
            ret["corr_by_arg"].append(
                _corr_by_param(
                    by_name, state_or_trans, attribute, len(parameter_names) + arg_index
                )
            )

            if False:
                ret["depends_on_arg"].append(ret["corr_by_arg"][arg_index] > 0.1)
            elif ret["std_by_arg"][arg_index] == 0:
                # In general, std_param_lut < std_by_arg. So, if std_by_arg == 0, std_param_lut == 0 follows.
                # This means that the variation of arg does not affect the model quality -> no influence
                ret["depends_on_arg"].append(False)
            else:
                ret["depends_on_arg"].append(
                    ret["std_param_lut"] / ret["std_by_arg"][arg_index] < 0.5
                )

    return ret


def _compute_param_statistics_parallel(arg):
    return {"key": arg["key"], "result": _compute_param_statistics(*arg["args"])}


def _all_params_are_numeric(data, param_idx):
    """Check if all `data['param'][*][param_idx]` elements are numeric, as reported by `utils.is_numeric`."""
    param_values = list(map(lambda x: x[param_idx], data["param"]))
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


class ParamStats:
    """
    :param stats: `stats[state_or_tran][attribute]` = std_static, std_param_lut, ... (see `compute_param_statistics`)
    :param distinct_values: `distinct_values[state_or_tran][param]` = [distinct values in aggregate]
    :param distinct_values_by_param_index: `distinct_values[state_or_tran][i]` = [distinct values in aggregate]
    """

    def __init__(
        self,
        by_name,
        by_param,
        parameter_names,
        arg_count,
        use_corrcoef=False,
    ):
        """
        Compute standard deviation and correlation coefficient on parameterized data partitions.

        It is strongly recommended to vary all parameter values evenly.
        For instance, given two parameters, providing only the combinations
        (1, 1), (5, 1), (7, 1,) (10, 1), (1, 2), (1, 6) will lead to bogus results.
        It is better to provide (1, 1), (5, 1), (1, 2), (5, 2), ... (i.e. a cross product of all individual parameter values)

        arguments:
        by_name -- ground truth partitioned by state/transition name.
            by_name[state_or_trans][attribute] must be a list or 1-D numpy array.
            by_name[state_or_trans]['param'] must be a list of parameter values
            corresponding to the ground truth, e.g. [[1, 2, 3], ...] if the
            first ground truth element has the (lexically) first parameter set to 1,
            the second to 2 and the third to 3.
        by_param -- ground truth partitioned by state/transition name and parameters.
            by_name[(state_or_trans, *)][attribute] must be a list or 1-D numpy array.
        parameter_names -- list of parameter names, must have the same order as the parameter
            values in by_param (lexical sorting is recommended).
        arg_count -- dict providing the number of functions args ("local parameters") for each function.
        use_corrcoef -- use correlation coefficient instead of stddev heuristic for parameter detection
        """
        self.stats = dict()
        self.distinct_values = dict()
        self.distinct_values_by_param_index = dict()
        self.use_corrcoef = use_corrcoef
        self._parameter_names = parameter_names

        stats_queue = list()

        for state_or_tran in by_name.keys():
            self.stats[state_or_tran] = dict()
            self.distinct_values_by_param_index[state_or_tran] = distinct_param_values(
                by_name, state_or_tran
            )
            self.distinct_values[state_or_tran] = dict()
            for i, param in enumerate(parameter_names):
                self.distinct_values[state_or_tran][
                    param
                ] = self.distinct_values_by_param_index[state_or_tran][i]
            for attribute in by_name[state_or_tran]["attributes"]:
                stats_queue.append(
                    {
                        "key": [state_or_tran, attribute],
                        "args": [
                            by_name,
                            by_param,
                            parameter_names,
                            arg_count,
                            state_or_tran,
                            attribute,
                            self.distinct_values[state_or_tran],
                            self.distinct_values_by_param_index[state_or_tran],
                        ],
                    }
                )

        with Pool() as pool:
            stats_results = pool.map(_compute_param_statistics_parallel, stats_queue)

        for stats in stats_results:
            state_or_tran, attribute = stats["key"]
            self.stats[state_or_tran][attribute] = stats["result"]

    def can_be_fitted(self, state_or_tran=None) -> bool:
        """
        Return whether a sufficient amount of distinct numeric parameter values is available, allowing a parameter-aware model to be generated.

        :param state_or_tran: state or transition. If unset, returns whether any state or transition can be fitted.
        """
        if state_or_tran is None:
            keys = self.stats.keys()
        else:
            keys = [state_or_tran]

        for key in keys:
            for param in self._parameter_names:
                if (
                    len(
                        list(
                            filter(
                                lambda n: is_numeric(n),
                                self.distinct_values[key][param],
                            )
                        )
                    )
                    > 2
                ):
                    logger.debug(
                        "{} can be fitted for param {} on {}".format(
                            key,
                            param,
                            list(
                                filter(
                                    lambda n: is_numeric(n),
                                    self.distinct_values[key][param],
                                )
                            ),
                        )
                    )
                    return True
        return False

    def _generic_param_independence_ratio(self, state_or_trans, attribute):
        """
        Return the heuristic ratio of parameter independence for state_or_trans and attribute.

        This is not supported if the correlation coefficient is used.
        A value close to 1 means no influence, a value close to 0 means high probability of influence.
        """
        statistics = self.stats[state_or_trans][attribute]
        if self.use_corrcoef:
            # not supported
            raise ValueError
        if statistics["std_static"] == 0:
            return 0
        return statistics["std_param_lut"] / statistics["std_static"]

    def generic_param_dependence_ratio(self, state_or_trans, attribute):
        """
        Return the heuristic ratio of parameter dependence for state_or_trans and attribute.

        This is not supported if the correlation coefficient is used.
        A value close to 0 means no influence, a value close to 1 means high probability of influence.
        """
        return 1 - self._generic_param_independence_ratio(state_or_trans, attribute)

    def _param_independence_ratio(
        self, state_or_trans: str, attribute: str, param: str
    ) -> float:
        """
        Return the heuristic ratio of parameter independence for state_or_trans, attribute, and param.

        A value close to 1 means no influence, a value close to 0 means high probability of influence.
        """
        statistics = self.stats[state_or_trans][attribute]
        if self.use_corrcoef:
            return 1 - np.abs(statistics["corr_by_param"][param])
        if statistics["std_by_param"][param] == 0:
            if statistics["std_param_lut"] != 0:
                raise RuntimeError("wat")
            # In general, std_param_lut < std_by_param. So, if std_by_param == 0, std_param_lut == 0 follows.
            # This means that the variation of param does not affect the model quality -> no influence, return 1
            return 1.0

        return statistics["std_param_lut"] / statistics["std_by_param"][param]

    def param_dependence_ratio(
        self, state_or_trans: str, attribute: str, param: str
    ) -> float:
        """
        Return the heuristic ratio of parameter dependence for state_or_trans, attribute, and param.

        A value close to 0 means no influence, a value close to 1 means high probability of influence.

        :param state_or_trans: state or transition name
        :param attribute: model attribute
        :param param: parameter name

        :returns: parameter dependence (float between 0 == no influence and 1 == high probability of influence)
        """
        return 1 - self._param_independence_ratio(state_or_trans, attribute, param)

    def _arg_independence_ratio(self, state_or_trans, attribute, arg_index):
        statistics = self.stats[state_or_trans][attribute]
        if self.use_corrcoef:
            return 1 - np.abs(statistics["corr_by_arg"][arg_index])
        if statistics["std_by_arg"][arg_index] == 0:
            if statistics["std_param_lut"] != 0:
                raise RuntimeError("wat")
            # In general, std_param_lut < std_by_arg. So, if std_by_arg == 0, std_param_lut == 0 follows.
            # This means that the variation of arg does not affect the model quality -> no influence, return 1
            return 1
        return statistics["std_param_lut"] / statistics["std_by_arg"][arg_index]

    def arg_dependence_ratio(
        self, state_or_trans: str, attribute: str, arg_index: int
    ) -> float:
        return 1 - self._arg_independence_ratio(state_or_trans, attribute, arg_index)

    # This heuristic is very similar to the "function is not much better than
    # median" checks in get_fitted. So far, doing it here as well is mostly
    # a performance and not an algorithm quality decision.
    # --df, 2018-04-18
    def depends_on_param(self, state_or_trans, attribute, param):
        """Return whether attribute of state_or_trans depens on param."""
        return self.stats[state_or_trans][attribute]["depends_on_param"][param]

    # See notes on depends_on_param
    def depends_on_arg(self, state_or_trans, attribute, arg_index):
        """Return whether attribute of state_or_trans depens on arg_index."""
        return self.stats[state_or_trans][attribute]["depends_on_arg"][arg_index]
