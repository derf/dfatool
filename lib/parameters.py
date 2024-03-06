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
from .utils import remove_indexes_from_tuple, is_numeric
from .utils import filter_aggregate_by_param, partition_by_param
from .utils import param_to_ndarray
from .utils import soft_cast_int, soft_cast_float

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
    if not len(param_tuples):
        logger.warning("distinct_param_values called with param_tuples=[]")
        return list()
    distinct_values = [OrderedDict() for i in range(len(param_tuples[0]))]
    for param_tuple in param_tuples:
        for i in range(len(param_tuple)):
            distinct_values[i][param_tuple[i]] = True

    # Convert sets to lists
    distinct_values = list(map(lambda x: list(x.keys()), distinct_values))
    return distinct_values


def _depends_on_param(corr_param, std_param, std_lut, threshold=0.5):
    if std_param == 0:
        # In general, std_param_lut < std_by_param. So, if std_by_param == 0, std_param_lut == 0 follows.
        # This means that the variation of param does not affect the model quality -> no influence
        # assert std_lut == 0
        return False
    return std_lut / std_param < threshold


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
    return _mean_std_by_params(n_by_param, all_param_values, [param_index])


def _mean_std_by_params(n_by_param, all_param_values, param_indexes):
    """
    Calculate the mean standard deviation for a static model where all parameters but `param_indexes` are constant.

    :param n_by_param: measurements of a specific model attribute partitioned by parameter values.
        Example: `{(0, 2): [2], (0, 4): [4], (0, 6): [6]}`
    :param all_param_values: distinct values of each parameter.
        E.g. for two parameters, the first being None, FOO, or BAR, and the second being 1, 2, 3, or 4, the argument is
        `[[None, 'FOO', 'BAR'], [1, 2, 3, 4]]`.
    :param param_indexes: indexes of variable parameters
    :returns: mean stddev
        *mean stddev* is the mean standard deviation of all measurements where parameters `param_indexes` are dynamic and all other parameters are fixed.
        E.g., if parameters are a, b, c ∈ {1,2,3} and 'indexes' corresponds to b and c, then
        this function returns the mean of the standard deviations of (a=1, b=*, c=*),
        (a=2, b=*, c=*), and so on.
    """
    partition_by_tuple = dict()

    for k, v in n_by_param.items():
        tuple_key = remove_indexes_from_tuple(k, param_indexes)
        if not tuple_key in partition_by_tuple:
            partition_by_tuple[tuple_key] = list()
        partition_by_tuple[tuple_key].extend(v)

    if len(partition_by_tuple) == 0:
        return 0.0

    return np.mean([np.std(partition) for partition in partition_by_tuple.values()])


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
        param_values = np.array(
            list((map(lambda x: float(x[param_index]), param_values)))
        )
        try:
            return np.corrcoef(attribute_data, param_values)[0, 1]
        except FloatingPointError:
            # Typically happens when all parameter values are identical.
            # Building a correlation coefficient is pointless in this case
            # -> assume no correlation
            return 0.0
        except (TypeError, ValueError) as e:
            logger.error(f"{e} in _corr_by_param(param_index={param_index})")
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

    relevance_threshold = float(os.getenv("DFATOOL_PARAM_RELEVANCE_THRESHOLD", 0.5))

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
            relevance_threshold,
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
                    ret["std_param_lut"] / ret["std_by_arg"][arg_index]
                    < relevance_threshold
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
    if not len(param_values):
        logger.warning("codependent_param_dict called with param_values=[]")
        return dict()
    if bool(int(os.getenv("DFATOOL_ULS_SKIP_CODEPENDENT_CHECK", 0))):
        return dict()
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


class ParamType(dict):
    UNSET = 0
    USELESS = 1
    BOOLEAN = 2
    SCALAR = 3
    ENUM = 4

    def __init__(self, param_values, values_are_distinct=False):
        if values_are_distinct:
            distinct_values = param_values
        else:
            distinct_values = distinct_param_values(param_values)
        for param_index, param_values in enumerate(distinct_values):
            if None in param_values:
                none_adj = -1
            else:
                none_adj = 0
            value_count = len(param_values) + none_adj
            if value_count == 0:
                self[param_index] = self.UNSET
            elif value_count == 1:
                self[param_index] = self.USELESS
            elif value_count == 2:
                self[param_index] = self.BOOLEAN
            elif all(map(lambda n: n is None or is_numeric(n), param_values)):
                self[param_index] = self.SCALAR
            else:
                self[param_index] = self.ENUM


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
        logger.debug("Computing param stats in parallel")
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
        param_type=dict(),
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

        # dict: Parameter index -> Parameter type (UNSET, BOOLEAN, SCALAR, ...)
        self.param_type = param_type

        self.nonscalar_param_indexes = list(
            map(
                lambda kv: kv[0],
                filter(lambda kv: kv[1] != ParamType.SCALAR, self.param_type.items()),
            )
        )
        self.scalar_param_indexes = list(
            map(
                lambda kv: kv[0],
                filter(lambda kv: kv[1] == ParamType.SCALAR, self.param_type.items()),
            )
        )

        # Co-dependent parameters. If (param1_index, param2_index) in codependent_param, they are codependent.
        # In this case, only one of them must be used for parameter-dependent model attribute detection and modeling
        self.codependent_param_pair = codependent_param
        self.codependent_params = [list() for x in self.log_param_names]
        self.ignore_codependent_param = dict()

        # Static model used as lower bound of model accuracy
        if data is not None:
            self.mean = np.mean(data)
            self.median = np.median(data)
        else:
            self.mean = None
            self.median = None

        # LUT model used as upper bound of model accuracy
        self.by_param = None  # set via ParallelParamStats or get_by_param

        self.stats = None  # set via ParallelParamStats

        # param model override
        self.function_override = None

        # The best model we have. May be Static, Split, or Param (and later perhaps Substate)
        self.model_function = None

        self._check_codependent_param()

        # There must be at least 3 distinct data values (≠ None) if an analytic model
        # is to be fitted. For 2 (or fewer) values, decision trees are better.
        # Exceptions such as DFATOOL_RMT_SUBMODEL=fol (2 values sufficient)
        # can be handled via DFATOOL_ULS_MIN_DISTINCT_VALUES
        self.min_values_for_analytic_model = int(
            os.getenv("DFATOOL_ULS_MIN_DISTINCT_VALUES", "3")
        )

    def __repr__(self):
        mean = np.mean(self.data)
        return f"ModelAttribute<{self.name}, {self.attr}, mean={mean}>"

    def to_json(self, **kwargs):
        return {
            "paramNames": self.param_names,
            "argCount": self.arg_count,
            "modelFunction": self.model_function.to_json(**kwargs),
        }

    def to_dref(self, unit=None):
        ret = {"mean": (self.mean, unit), "median": (self.median, unit)}

        if issubclass(type(self.model_function), df.ModelFunction):
            ret["model/complexity"] = self.model_function.get_complexity_score()
        if self.by_param:
            ret["lut/complexity"] = len(self.by_param.keys()) + 1

        if type(self.model_function) in (
            df.SplitFunction,
            df.CARTFunction,
            df.XGBoostFunction,
            df.LMTFunction,
        ):
            ret["decision tree/nodes"] = self.model_function.get_number_of_nodes()
            ret["decision tree/leaves"] = self.model_function.get_number_of_leaves()
            ret["decision tree/inner nodes"] = (
                ret["decision tree/nodes"] - ret["decision tree/leaves"]
            )
            ret["decision tree/max depth"] = self.model_function.get_max_depth()
        elif type(self.model_function) in (df.StaticFunction, df.AnalyticFunction):
            ret["decision tree/nodes"] = 1
            ret["decision tree/leaves"] = 1
            ret["decision tree/inner nodes"] = 0
            ret["decision tree/max depth"] = 0

        return ret

    def to_dot(self):
        if type(self.model_function) in (
            df.SplitFunction,
            df.StaticFunction,
            df.AnalyticFunction,
            df.FOLFunction,
        ):
            import pydot

            graph = pydot.Dot("Regression Model Tree", graph_type="graph")
            self.model_function.to_dot(pydot, graph, self.param_names)
            return graph

        if type(self.model_function) == df.CARTFunction:
            import sklearn.tree

            return sklearn.tree.export_graphviz(
                self.model_function.regressor,
                out_file=None,
                feature_names=self.model_function.feature_names,
            )
        if type(self.model_function) == df.XGBoostFunction:
            import xgboost

            self.model_function.regressor.get_booster().feature_names = (
                self.model_function.feature_names
            )
            return [
                xgboost.to_graphviz(self.model_function.regressor, num_trees=i)
                for i in range(self.model_function.regressor.n_estimators)
            ]
        if type(self.model_function) == df.LMTFunction:
            return self.model_function.regressor.model_to_dot(
                feature_names=self.model_function.feature_names
            )
        return None

    def min(self):
        return np.min(self.data)

    def max(self):
        return np.max(self.data)

    def webconf_function_map(self):
        return self.model_function.webconf_function_map()

    @classmethod
    def from_json(cls, name, attr, data):
        param_names = data["paramNames"]
        arg_count = data["argCount"]

        self = cls(name, attr, None, None, param_names, arg_count)

        self.model_function = df.ModelFunction.from_json(data["modelFunction"])
        self.mean = self.model_function.value
        self.median = self.model_function.value

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
            # If all occurences of (param1, param2) are either (None, None) or (not None, not None), removing one of them is sensible.
            # Otherwise, one parameter may decide whether the other one has an effect or not (or what kind of effect it has). This is important for
            # decision-tree models, so do not remove parameters in that case.
            params_are_pairwise_none = all(
                map(
                    lambda pv: not (
                        (pv[param1_index] is None) ^ (pv[param2_index] is None)
                    ),
                    self.param_values,
                )
            )
            if (
                param1_numeric_count >= param2_numeric_count
                and params_are_pairwise_none
            ):
                self.ignore_codependent_param[param2_index] = True
                self.codependent_params[param1_index].append(param2_index)
                logger.debug(
                    f"{self.name} {self.attr}: parameters ({self.log_param_names[param1_index]}, {self.log_param_names[param2_index]}) are codependent. Ignoring {self.log_param_names[param2_index]}"
                )
            elif (
                param2_numeric_count >= param1_numeric_count
                and params_are_pairwise_none
            ):
                self.ignore_codependent_param[param1_index] = True
                self.codependent_params[param2_index].append(param1_index)
                logger.debug(
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

    def get_by_param(self):
        if self.by_param is None and self.param_values is not None:
            self.by_param = partition_by_param(self.data, self.param_values)
        return self.by_param

    def get_data_for_paramfit(self, safe_functions_enabled=False):
        ret = list()
        for param_index, param_name in enumerate(self.param_names):
            if (
                self.stats.depends_on_param(param_name)
                and not param_index in self.ignore_codependent_param
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
                    and not param_index in self.ignore_codependent_param
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

    def _by_param_for_index(self, param_index):
        if not self.codependent_params[param_index]:
            return self.by_param
        new_param_values = list()
        for param_tuple in self.param_values:
            new_param_tuple = param_tuple.copy()
            for i in self.codependent_params[param_index]:
                new_param_tuple[i] = None
            new_param_values.append(new_param_tuple)
        return partition_by_param(self.data, new_param_values)

    def depends_on_any_param(self):
        for param_index, param_name in enumerate(self.param_names):
            if (
                self.stats.depends_on_param(param_name)
                and not param_index in self.ignore_codependent_param
            ):
                return True
        return False

    def all_relevant_parameters_are_none_or_numeric(self):
        for param_index, param_name in enumerate(self.param_names):
            if (
                self.stats.depends_on_param(param_name)
                and not param_index in self.ignore_codependent_param
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

    def fit_override_function(self):
        function_str = self.function_override
        x = df.AnalyticFunction(
            self.median,
            function_str,
            self.param_names,
            self.arg_count,
            n_samples=self.data.shape[0],
            # fit_by_param=fit_result,
        )
        x.fit(self.by_param)
        if x.fit_success:
            self.model_function = x
        else:
            logger.warning(f"Fit of user-defined model function {function_str} failed.")

    def set_data_from_paramfit(self, paramfit, prefix=tuple()):
        fit_result = paramfit.get_result((self.name, self.attr) + prefix)
        if self.model_function is None:
            self.model_function = df.StaticFunction(
                self.median, n_samples=self.data.shape[0]
            )
        if os.getenv("DFATOOL_NO_PARAM"):
            pass
        elif len(fit_result.keys()):
            x = df.analytic.function_powerset(
                fit_result,
                self.param_names,
                self.arg_count,
                n_samples=self.data.shape[0],
            )
            x.value = self.median
            x.fit(self.by_param)

            if x.fit_success:
                self.model_function = x

    def build_cart(self):
        mf = df.CARTFunction(
            np.mean(self.data),
            n_samples=len(self.data),
            param_names=self.param_names,
            arg_count=self.arg_count,
        ).fit(
            self.param_values,
            self.data,
        )

        if mf.fit_success:
            self.model_function = mf
            return True
        else:
            logger.warning(f"CART generation for {self.name} {self.attr} faled")
            self.model_function = df.StaticFunction(
                np.mean(self.data), n_samples=len(self.data)
            )
            return False

    def build_decart(self):
        mf = df.CARTFunction(
            np.mean(self.data),
            n_samples=len(self.data),
            param_names=self.param_names,
            arg_count=self.arg_count,
            decart=True,
        ).fit(
            self.param_values,
            self.data,
            scalar_param_indexes=self.scalar_param_indexes,
        )

        if mf.fit_success:
            self.model_function = mf
            return True
        else:
            logger.warning(f"DECART generation for {self.name} {self.attr} faled")
            self.model_function = df.StaticFunction(
                np.mean(self.data), n_samples=len(self.data)
            )
            return False

    def build_fol(self):
        ignore_irrelevant = bool(
            int(os.getenv("DFATOOL_RMT_IGNORE_IRRELEVANT_PARAMS", "0"))
        )
        ignore_param_indexes = list()
        if ignore_irrelevant:
            for param_index, param in enumerate(self.param_names):
                if not self.stats.depends_on_param(param):
                    ignore_param_indexes.append(param_index)
        if not self.stats:
            logger.warning(
                "build_fol_model called with ModelAttribute.stats unavailable -- overfitting likely"
            )
        else:
            for param_index, _ in enumerate(self.param_names):
                if len(self.stats.distinct_values_by_param_index[param_index]) < 2:
                    ignore_param_indexes.append(param_index)
        x = df.FOLFunction(
            self.median,
            self.param_names,
            n_samples=self.data.shape[0],
            num_args=self.arg_count,
        )
        x.fit(self.param_values, self.data, ignore_param_indexes=ignore_param_indexes)
        if x.fit_success:
            self.model_function = x
            return True
        else:
            logger.warning(f"Fit of first-order linear model function failed.")
            self.model_function = df.StaticFunction(
                np.mean(self.data), n_samples=len(self.data)
            )
            return False

    def build_lmt(self):
        mf = df.LMTFunction(
            np.mean(self.data),
            n_samples=len(self.data),
            param_names=self.param_names,
            arg_count=self.arg_count,
        ).fit(self.param_values, self.data)

        if mf.fit_success:
            self.model_function = mf
            return True
        else:
            logger.warning(f"LMT generation for {self.name} {self.attr} faled")
            self.model_function = df.StaticFunction(
                np.mean(self.data), n_samples=len(self.data)
            )
            return False

    def build_symreg(self):
        ignore_irrelevant = bool(
            int(os.getenv("DFATOOL_RMT_IGNORE_IRRELEVANT_PARAMS", "0"))
        )
        ignore_param_indexes = list()
        if ignore_irrelevant:
            for param_index, param in enumerate(self.param_names):
                if not self.stats.depends_on_param(param):
                    ignore_param_indexes.append(param_index)
        x = df.SymbolicRegressionFunction(
            np.mean(self.data),
            n_samples=self.data.shape[0],
            param_names=self.param_names,
            arg_count=self.arg_count,
        ).fit(self.param_values, self.data, ignore_param_indexes=ignore_param_indexes)
        if x.fit_success:
            self.model_function = x
            return True
        else:
            logger.debug(
                f"Symbolic Regression model generation for {self.name} {self.attr} failed."
            )
            self.model_function = df.StaticFunction(
                np.mean(self.data), n_samples=len(self.data)
            )
            return False

    def build_lgbm(self):
        mf = df.LightGBMFunction(
            np.mean(self.data),
            n_samples=len(self.data),
            param_names=self.param_names,
            arg_count=self.arg_count,
        ).fit(self.param_values, self.data)

        if mf.fit_success:
            self.model_function = mf
            return True
        else:
            logger.warning(f"LightGBM generation for {self.name} {self.attr} faled")
            self.model_function = df.StaticFunction(
                np.mean(self.data), n_samples=len(self.data)
            )
            return False

    def build_xgb(self):
        mf = df.XGBoostFunction(
            np.mean(self.data),
            n_samples=len(self.data),
            param_names=self.param_names,
            arg_count=self.arg_count,
        ).fit(self.param_values, self.data)

        if mf.fit_success:
            self.model_function = mf
            return True
        else:
            logger.warning(f"XGB generation for {self.name} {self.attr} faled")
            self.model_function = df.StaticFunction(
                np.mean(self.data), n_samples=len(self.data)
            )
            return False

    def build_rmt(
        self,
        with_function_leaves=None,
        with_nonbinary_nodes=None,
        with_gplearn_symreg=None,
        ignore_irrelevant_parameters=None,
        loss_ignore_scalar=None,
        threshold=100,
    ):
        """
        Build a Decision Tree on `param` / `data` for kconfig models.

        :param parameters: parameter values for each measurement. [(data 1 param 1, data 1 param 2, ...), (data 2 param 1, data 2 param 2, ...), ...]
        :param data: Measurements. [data 1, data 2, data 3, ...]
        :param with_function_leaves: Use fitted function sets to generate function leaves for scalar parameters
        :param with_nonbinary_nodes: Allow non-binary nodes for enum and scalar parameters (i.e., nodes with more than two children)
        :param loss_ignore_scalar: Ignore scalar parameters when computing the loss for split candidates. Only sensible if with_function_leaves is enabled.
        :param threshold: Return a StaticFunction leaf node if std(data) < threshold. Default 100.

        :returns: SplitFunction or StaticFunction
        """

        if with_function_leaves is None:
            if os.getenv("DFATOOL_RMT_SUBMODEL", "uls") == "static":
                with_function_leaves = False
            else:
                with_function_leaves = True
        if with_nonbinary_nodes is None:
            with_nonbinary_nodes = bool(
                int(os.getenv("DFATOOL_RMT_NONBINARY_NODES", "1"))
            )
        if with_gplearn_symreg is None:
            with_gplearn_symreg = bool(int(os.getenv("DFATOOL_USE_SYMREG", "0")))
        if ignore_irrelevant_parameters is None:
            ignore_irrelevant_parameters = bool(
                int(os.getenv("DFATOOL_RMT_IGNORE_IRRELEVANT_PARAMS", "0"))
            )
        if loss_ignore_scalar is None:
            loss_ignore_scalar = bool(
                int(os.getenv("DFATOOL_RMT_LOSS_IGNORE_SCALAR", "0"))
            )

        if loss_ignore_scalar and not with_function_leaves:
            logger.warning(
                "build_rmt {self.name} {self.attr} called with loss_ignore_scalar=True, with_function_leaves=False. This does not make sense."
            )

        relevance_threshold = float(os.getenv("DFATOOL_PARAM_RELEVANCE_THRESHOLD", 0.5))

        logger.debug(
            f"build_rmt(threshold={threshold}, relevance_threshold={relevance_threshold})"
        )

        self.model_function = self._build_rmt(
            self.param_values,
            self.data,
            with_function_leaves=with_function_leaves,
            with_nonbinary_nodes=with_nonbinary_nodes,
            ignore_irrelevant_parameters=ignore_irrelevant_parameters,
            loss_ignore_scalar=loss_ignore_scalar,
            submodel=os.getenv("DFATOOL_RMT_SUBMODEL", "uls"),
            threshold=threshold,
            relevance_threshold=relevance_threshold,
        )

    def _build_rmt(
        self,
        parameters,
        data,
        with_function_leaves=False,
        with_nonbinary_nodes=True,
        ignore_irrelevant_parameters=True,
        loss_ignore_scalar=False,
        submodel="uls",
        threshold=100,
        relevance_threshold=0.5,
        level=0,
    ):
        """
        Build a Decision Tree on `param` / `data` for kconfig models.

        :param parameters: parameter values for each measurement. [(data 1 param 1, data 1 param 2, ...), (data 2 param 1, data 2 param 2, ...), ...]
        :param data: Measurements. [data 1, data 2, data 3, ...]
        :param with_function_leaves: Use fitted function sets to generate function leaves for scalar parameters
        :param with_nonbinary_nodes: Allow non-binary nodes for enum and scalar parameters (i.e., nodes with more than two children)
        :param loss_ignore_scalar: Ignore scalar parameters when computing the loss for split candidates. Only sensible if with_function_leaves is enabled.
        :param threshold: Return a StaticFunction leaf node if std(data) < threshold. Default 100.

        :returns: ModelFunction
        """

        nonarg_count = len(self.param_names)
        param_count = nonarg_count + self.arg_count
        # TODO eigentlich muss threshold hier auf Basis der aktuellen Messdatenpartition neu berechnet werden
        if param_count == 0 or np.std(data) <= threshold:
            return df.StaticFunction(np.mean(data), n_samples=len(data))
            # sf.value_error["std"] = np.std(data)

        loss = list()

        ffs_feasible = False
        if ignore_irrelevant_parameters:
            by_param = partition_by_param(data, parameters)
            distinct_values_by_param_index = distinct_param_values(parameters)
            std_lut = np.mean([np.std(v) for v in by_param.values()])
            irrelevant_params = list()

        if loss_ignore_scalar:
            ffs_eligible_params = list()
            ffs_unsuitable_params = list()
            for param_index in range(param_count):
                if param_index in self.ignore_codependent_param:
                    continue
                unique_values = list(set(map(lambda p: p[param_index], parameters)))
                if None in unique_values:
                    ffs_unsuitable_params.append(param_index)
                elif (
                    self.param_type[param_index] == ParamType.SCALAR
                    and len(unique_values) >= self.min_values_for_analytic_model
                ):
                    ffs_eligible_params.append(param_index)
                else:
                    ffs_unsuitable_params.append(param_index)

        for param_index in range(param_count):
            if (
                param_index >= nonarg_count
                and self.param_type[param_index] == ParamType.ENUM
            ):
                # do not split on non-numeric function arguments
                loss.append(np.inf)
                continue

            unique_values = list(set(map(lambda p: p[param_index], parameters)))

            if None in unique_values:
                # param is a choice and undefined in some configs. Do not split on it.
                loss.append(np.inf)
                continue

            if (
                with_function_leaves
                and self.param_type[param_index] == ParamType.SCALAR
                and len(unique_values) >= self.min_values_for_analytic_model
            ):
                # param can be modeled as a function. Do not split on it.
                loss.append(np.inf)
                ffs_feasible = True
                continue

            # if not with_nonbinary_nodes and sorted(unique_values) != [0, 1]:
            if not with_nonbinary_nodes and len(unique_values) > 2:
                # param cannot be handled with a binary split
                loss.append(np.inf)
                continue

            if ignore_irrelevant_parameters:
                std_by_param = _mean_std_by_params(
                    by_param,
                    distinct_values_by_param_index,
                    list(self.ignore_codependent_param.keys())
                    + irrelevant_params
                    + [param_index],
                )
                if not _depends_on_param(
                    None, std_by_param, std_lut, relevance_threshold
                ):
                    irrelevant_params.append(param_index)
                    loss.append(np.inf)
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
                assert len(child_indexes[-1]) > 0

            assert len(child_indexes) != 0

            if len(child_indexes) == 1:
                # this param only has a single value. there's no point in splitting.
                loss.append(np.inf)
                continue

            children = list()
            for child in child_indexes:
                child_data = list(map(lambda i: data[i], child))
                if loss_ignore_scalar and False:
                    child_param = list(map(lambda i: parameters[i], child))
                    child_data_by_scalar = partition_by_param(
                        child_data,
                        child_param,
                        ignore_parameters=list(self.ignore_codependent_param.keys())
                        + ffs_unsuitable_params,
                    )
                    logger.debug(f"got {len(child_data_by_scalar)} partitions")
                    for sub_data in child_data_by_scalar.values():
                        assert len(sub_data)
                        children.extend((np.array(sub_data) - np.mean(sub_data)) ** 2)
                else:
                    children.extend((np.array(child_data) - np.mean(child_data)) ** 2)

            assert not np.any(np.isnan(children))
            loss.append(np.sum(children))

        if np.all(np.isinf(loss)) or np.min(loss) >= np.sum(
            (np.array(data) - np.mean(data)) ** 2
        ):
            if ffs_feasible:
                # try generating a function. if it fails, model_function is a StaticFunction.
                ma = ModelAttribute(
                    self.name + "_",
                    self.attr,
                    data,
                    parameters,
                    self.param_names,
                    arg_count=self.arg_count,
                    param_type=self.param_type,
                    codependent_param=codependent_param_dict(parameters),
                )
                if submodel == "cart":
                    if ma.build_cart():
                        return ma.model_function
                elif submodel == "symreg":
                    if ma.build_symreg():
                        return ma.model_function
                else:
                    ParamStats.compute_for_attr(ma)
                    paramfit = ParamFit(parallel=False)
                    for key, param, args, kwargs in ma.get_data_for_paramfit():
                        paramfit.enqueue(key, param, args, kwargs)
                    paramfit.fit()
                    ma.set_data_from_paramfit(paramfit)
                    return ma.model_function
            return df.StaticFunction(np.mean(data), n_samples=len(data))

        split_feasible = True
        if loss_ignore_scalar:
            data_by_scalar = partition_by_param(
                data,
                parameters,
                ignore_parameters=list(self.ignore_codependent_param.keys())
                + ffs_unsuitable_params,
            )
            if np.all(
                np.array([np.std(partition) for partition in data_by_scalar.values()])
                <= threshold
            ):
                # Varying non-scalar params in partitions with fixed scalar params does not affect system behaviour
                # -> further non-scalar splits are _probably_ not sensible
                # (_probably_ because this implicitly assumes that there are multiple scalar configurations for each non-scalar configuration.
                split_feasible = False

        if ffs_feasible and not split_feasible:
            # There is a _probably_ above: the heuristic assumes that there are multiple scalar configurations for each non-scalar configuration.
            # If there is just one it may recommend to stop splitting too early.
            # Hence, we will try generating an FFS leaf node here, but continue splitting if it turns out that it is no good.
            ma = ModelAttribute(
                self.name + "_",
                self.attr,
                data,
                parameters,
                self.param_names,
                arg_count=self.arg_count,
                param_type=self.param_type,
                codependent_param=codependent_param_dict(parameters),
            )
            if submodel == "cart":
                if ma.build_cart():
                    return ma.model_function
            elif submodel == "symreg":
                if ma.build_symreg():
                    return ma.model_function
            else:
                ParamStats.compute_for_attr(ma)
                paramfit = ParamFit(parallel=False)
                for key, param, args, kwargs in ma.get_data_for_paramfit():
                    paramfit.enqueue(key, param, args, kwargs)
                paramfit.fit()
                ma.set_data_from_paramfit(paramfit)
                if type(ma.model_function) == df.AnalyticFunction:
                    return ma.model_function

        symbol_index = np.argmin(loss)
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
            assert len(child_data)
            child[value] = self._build_rmt(
                child_parameters,
                child_data,
                with_function_leaves=with_function_leaves,
                with_nonbinary_nodes=with_nonbinary_nodes,
                ignore_irrelevant_parameters=ignore_irrelevant_parameters,
                loss_ignore_scalar=loss_ignore_scalar,
                submodel=submodel,
                threshold=threshold,
                relevance_threshold=relevance_threshold,
                level=level + 1,
            )

        assert len(child.values()) >= 2

        return df.SplitFunction(
            np.mean(data),
            symbol_index,
            self.log_param_names[symbol_index],
            child,
            n_samples=len(data),
        )
