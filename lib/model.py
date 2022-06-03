#!/usr/bin/env python3

import logging
import numpy as np
import os
from .automata import PTA, ModelAttribute
from .functions import StaticFunction, SubstateFunction, SplitFunction
from .parameters import (
    ParamType,
    ParallelParamStats,
    ParamStats,
    codependent_param_dict,
    distinct_param_values,
)
from .paramfit import ParamFit
from .utils import is_numeric, soft_cast_int, by_name_to_by_param, regression_measures

logger = logging.getLogger(__name__)


def _num_args_from_by_name(by_name):
    num_args = dict()
    for key, value in by_name.items():
        if "args" in value:
            num_args[key] = len(value["args"][0])
    return num_args


class AnalyticModel:
    """
    Parameter-aware analytic energy/data size/... model.

    Supports both static and parameter-based model attributes, and automatic detection of parameter-dependence.

    These provide measurements aggregated by (function/state/...) name
    and (for by_param) parameter values. Layout:
    dictionary with one key per name ('send', 'TX', ...) or
    one key per name and parameter combination
    (('send', (1, 2)), ('send', (2, 3)), ('TX', (1, 2)), ('TX', (2, 3)), ...).

    Parameter values must be ordered corresponding to the lexically sorted parameter names.

    Each element is in turn a dict with the following elements:
    - param: list of parameter values in each measurement (-> list of lists)
    - attributes: list of keys that should be analyzed,
        e.g. ['power', 'duration']
    - for each attribute mentioned in 'attributes': A list with measurements.
      All list except for 'attributes' must have the same length.

    For example:
    parameters = ['foo_count', 'irrelevant']
    by_name = {
        'foo' : [1, 1, 2],
        'bar' : [5, 6, 7],
        'attributes' : ['foo', 'bar'],
        'param' : [[1, 0], [1, 0], [2, 0]]
    }

    methods:
    get_static -- return static (parameter-unaware) model.
    get_param_lut -- return parameter-aware look-up-table model. Cannot model parameter combinations not present in by_param.
    get_fitted -- return parameter-aware model using fitted functions for behaviour prediction.

    variables:
    names -- function/state/... names (i.e., the keys of by_name)
    parameters -- parameter names
    stats -- ParamStats object providing parameter-dependency statistics for each name and attribute
    assess -- calculate model quality
    """

    def __init__(
        self,
        by_name,
        parameters,
        arg_count=None,
        function_override=dict(),
        use_corrcoef=False,
        compute_stats=True,
        force_tree=False,
        max_std=None,
    ):
        """
        Create a new AnalyticModel and compute parameter statistics.

        :param by_name: measurements aggregated by (function/state/...) name.
            Layout: dictionary with one key per name ('send', 'TX', ...) or
            one key per name and parameter combination
            (('send', (1, 2)), ('send', (2, 3)), ('TX', (1, 2)), ('TX', (2, 3)), ...).

            Parameter values must be ordered corresponding to the lexically sorted parameter names.

            Each element is in turn a dict with the following elements:
            - param: list of parameter values in each measurement (-> list of lists)
            - attributes: list of keys that should be analyzed,
                e.g. ['power', 'duration']
            - for each attribute mentioned in 'attributes': A list with measurements.
            All list except for 'attributes' must have the same length.

            For example:
            parameters = ['foo_count', 'irrelevant']
            by_name = {
                'foo' : [1, 1, 2],
                'duration' : [5, 6, 7],
                'attributes' : ['foo', 'duration'],
                'param' : [[1, 0], [1, 0], [2, 0]]
                # foo_count-^  ^-irrelevant
            }
        :param parameters: List of parameter names
        :param function_override: dict of overrides for automatic parameter function generation.
            If (state or transition name, model attribute) is present in function_override,
            the corresponding text string is the function used for analytic (parameter-aware/fitted)
            modeling of this attribute. It is passed to AnalyticFunction, see
            there for the required format. Note that this happens regardless of
            parameter dependency detection: The provided analytic function will be assigned
            even if it seems like the model attribute is static / parameter-independent.
        :param use_corrcoef: use correlation coefficient instead of stddev comparison to detect whether a model attribute depends on a parameter
        """
        self.cache = dict()
        self.by_name = by_name  # no longer required?
        self.attr_by_name = dict()
        self.names = sorted(by_name.keys())
        self.parameters = sorted(parameters)
        self.function_override = function_override.copy()
        self.dtree_max_std = max_std
        self._use_corrcoef = use_corrcoef
        self._num_args = arg_count
        if self._num_args is None:
            self._num_args = _num_args_from_by_name(by_name)

        self.distinct_param_values_by_name = dict()
        self.param_type_by_name = dict()
        for name in self.names:
            self.distinct_param_values_by_name[name] = distinct_param_values(
                by_name[name]["param"]
            )
            self.param_type_by_name[name] = ParamType(
                self.distinct_param_values_by_name[name], values_are_distinct=True
            )

        self.fit_done = False

        if compute_stats:
            self._compute_stats(by_name)

        if force_tree:
            for name in self.names:
                for attr in self.by_name[name]["attributes"]:
                    if max_std and name in max_std and attr in max_std[name]:
                        threshold = max_std[name][attr]
                    elif compute_stats:
                        threshold = (self.attr_by_name[name][attr].stats.std_param_lut,)
                    else:
                        threshold = 0
                    logger.debug(f"build_dtree({name}, {attr}, threshold={threshold})")
                    self.build_dtree(
                        name,
                        attr,
                        threshold=threshold,
                    )
            self.fit_done = True

    def __repr__(self):
        names = ", ".join(self.by_name.keys())
        return f"AnalyticModel<names=[{names}]>"

    def _compute_stats(self, by_name):

        paramstats = ParallelParamStats()

        for name, data in by_name.items():
            self.attr_by_name[name] = dict()
            codependent_param = codependent_param_dict(data["param"])
            for attr in data["attributes"]:
                model_attr = ModelAttribute(
                    name,
                    attr,
                    data[attr],
                    data["param"],
                    self.parameters,
                    self._num_args.get(name, 0),
                    codependent_param=codependent_param,
                    param_type=self.param_type_by_name[name],
                )
                self.attr_by_name[name][attr] = model_attr
                paramstats.enqueue((name, attr), model_attr)
                if (name, attr) in self.function_override:
                    model_attr.function_override = self.function_override[(name, attr)]

        paramstats.compute()

    def attributes(self, name):
        return self.attr_by_name[name].keys()

    def param_index(self, param_name):
        if param_name in self.parameters:
            return self.parameters.index(param_name)
        return len(self.parameters) + int(param_name)

    def param_name(self, param_index):
        if param_index < len(self.parameters):
            return self.parameters[param_index]
        return str(param_index)

    def get_static(self, use_mean=False):
        """
        Get static model function: name, attribute -> model value.

        Uses the median of by_name for modeling, unless `use_mean` is set.
        """
        model = dict()
        for name, attr in self.attr_by_name.items():
            model[name] = dict()
            for k, v in attr.items():
                model[name][k] = v.get_static(use_mean=use_mean)

        def static_model_getter(name, key, **kwargs):
            return model[name][key]

        return static_model_getter

    def get_param_lut(self, use_mean=False, fallback=False):
        """
        Get parameter-look-up-table model function: name, attribute, parameter values -> model value.

        The function can only give model values for parameter combinations
        present in by_param. By default, it raises KeyError for other values.

        arguments:
        fallback -- Fall back to the (non-parameter-aware) static model when encountering unknown parameter values
        """
        static_model = dict()
        lut_model = dict()
        for name, attr in self.attr_by_name.items():
            static_model[name] = dict()
            lut_model[name] = dict()
            for k, v in attr.items():
                static_model[name][k] = v.get_static(use_mean=use_mean)
                lut_model[name][k] = dict()
                if not v.by_param:
                    raise RuntimeError(
                        f"ModelAttribute({name}, {k}).by_param is None. Did you run ParallelParamStats.compute?"
                    )
                for param, model_value in v.by_param.items():
                    lut_model[name][k][param] = v.get_lut(param, use_mean=use_mean)

        def lut_median_getter(name, key, param, arg=list(), **kwargs):
            if arg:
                if type(param) is tuple:
                    param = list(param)
                param.extend(map(soft_cast_int, arg))
            param = tuple(param)
            try:
                return lut_model[name][key][param]
            except KeyError:
                if fallback:
                    return static_model[name][key]
                raise

        return lut_median_getter

    def get_fitted(self, use_mean=False, safe_functions_enabled=False):
        """
        Get parameter-aware model function and model information function.

        Returns two functions:
        model_function(name, attribute, param=parameter values) -> model value.
        model_info(name, attribute) -> {'fit_result' : ..., 'function' : ... } or None
        """

        if not self.fit_done:

            paramfit = ParamFit()
            tree_allowed = bool(int(os.getenv("DFATOOL_DTREE_ENABLED", "1")))
            use_fol = bool(int(os.getenv("DFATOOL_FIT_FOL", "0")))
            tree_required = dict()

            for name in self.names:
                tree_required[name] = dict()
                for attr in self.attr_by_name[name].keys():
                    if self.attr_by_name[name][attr].function_override is not None:
                        self.attr_by_name[name][attr].fit_override_function()
                    elif use_fol:
                        self.attr_by_name[name][attr].build_fol_model()
                    elif self.attr_by_name[name][
                        attr
                    ].all_relevant_parameters_are_none_or_numeric():
                        for key, param, args, kwargs in self.attr_by_name[name][
                            attr
                        ].get_data_for_paramfit(
                            safe_functions_enabled=safe_functions_enabled
                        ):
                            paramfit.enqueue(key, param, args, kwargs)
                    elif tree_allowed:
                        tree_required[name][attr] = self.attr_by_name[name][
                            attr
                        ].depends_on_any_param()

            paramfit.fit()

            for name in self.names:
                for attr in self.attr_by_name[name].keys():
                    if tree_required[name].get(attr, False):
                        threshold = self.attr_by_name[name][attr].stats.std_param_lut
                        if (
                            self.dtree_max_std
                            and name in self.dtree_max_std
                            and attr in self.dtree_max_std[name]
                        ):
                            threshold = self.dtree_max_std[name][attr]
                        logger.debug(
                            f"build_dtree({name}, {attr}, threshold={threshold})"
                        )
                        self.build_dtree(name, attr, threshold=threshold)
                    else:
                        self.attr_by_name[name][attr].set_data_from_paramfit(paramfit)

            self.fit_done = True

        static_model = dict()
        for name, attr in self.attr_by_name.items():
            static_model[name] = dict()
            for k, v in attr.items():
                static_model[name][k] = v.get_static(use_mean=use_mean)

        def model_getter(name, key, **kwargs):
            model_function = self.attr_by_name[name][key].model_function
            model_info = self.attr_by_name[name][key].model_function

            # shortcut
            if type(model_info) is StaticFunction:
                return static_model[name][key]

            if "arg" in kwargs and "param" in kwargs:
                kwargs["param"].extend(map(soft_cast_int, kwargs["arg"]))

            if model_function.is_predictable(kwargs["param"]):
                return model_function.eval(kwargs["param"])

            return static_model[name][key]

        def info_getter(name, key):
            try:
                return self.attr_by_name[name][key].model_function
            except KeyError:
                return None

        return model_getter, info_getter

    def assess(self, model_function, ref=None, return_raw=False):
        """
        Calculate MAE, SMAPE, etc. of model_function for each by_name entry.

        state/transition/... name and parameter values are fed into model_function.
        The by_name entries of this AnalyticModel are used as ground truth and
        compared with the values predicted by model_function.

        For proper model assessments, the data used to generate model_function
        and the data fed into this AnalyticModel instance must be mutually
        exclusive (e.g. by performing cross validation). Otherwise,
        overfitting cannot be detected.
        """
        detailed_results = dict()
        raw_results = dict()
        if ref is None:
            ref = self.by_name
        for name, elem in sorted(ref.items()):
            detailed_results[name] = dict()
            raw_results[name] = {
                "paramValues": elem["param"],
                "paramNames": self.parameters,
                "attribute": dict(),
            }
            for attribute in elem["attributes"]:
                predicted_data = np.array(
                    list(
                        map(
                            lambda i: model_function(
                                name, attribute, param=elem["param"][i]
                            ),
                            range(len(elem[attribute])),
                        )
                    )
                )
                measures = regression_measures(predicted_data, elem[attribute])
                detailed_results[name][attribute] = measures
                if return_raw:
                    raw_results[name]["attribute"][attribute] = {
                        "groundTruth": list(elem[attribute]),
                        "modelOutput": list(predicted_data),
                    }

        if return_raw:
            return detailed_results, raw_results
        return detailed_results

    def build_dtree(self, name, attribute, threshold=100, **kwargs):

        if name not in self.attr_by_name:
            self.attr_by_name[name] = dict()

        if attribute not in self.attr_by_name[name]:
            self.attr_by_name[name][attribute] = ModelAttribute(
                name,
                attribute,
                self.by_name[name][attribute],
                self.by_name[name]["param"],
                self.parameters,
                param_type=ParamType(self.by_name[name]["param"]),
            )

        self.attr_by_name[name][attribute].build_dtree(
            self.by_name[name]["param"],
            self.by_name[name][attribute],
            threshold=threshold,
            **kwargs,
        )

    def to_dref(
        self, static_quality, lut_quality, model_quality, xv_models=None
    ) -> dict:
        ret = dict()
        for name in self.names:
            param_data = {
                "unset": 0,
                "useless": 0,
                "boolean": 0,
                "scalar": 0,
                "enum": 0,
            }
            for param_index in range(
                len(self.parameters) + self._num_args.get(name, 0)
            ):
                param_type = self.param_type_by_name[name][param_index]
                if param_type == ParamType.UNSET:
                    param_data["unset"] += 1
                elif param_type == ParamType.USELESS:
                    param_data["useless"] += 1
                elif param_type == ParamType.BOOLEAN:
                    param_data["boolean"] += 1
                elif param_type == ParamType.SCALAR:
                    param_data["scalar"] += 1
                elif param_type == ParamType.ENUM:
                    param_data["enum"] += 1
                else:
                    raise RuntimeError(f"Unknown param_type: {param_type}")
            ret[f"paramcount/{name}/useful"] = (
                param_data["boolean"] + param_data["scalar"] + param_data["enum"]
            )
            for k, v in param_data.items():
                ret[f"paramcount/{name}/{k}"] = v
            for attr_name, attr in self.attr_by_name[name].items():
                # attr.data must be the same for all attrs
                ret[f"data/{name}/num samples"] = len(attr.data)
                unit = None
                if "power" in attr.attr:
                    unit = r"\micro\watt"
                elif "energy" in attr.attr:
                    unit = r"\pico\joule"
                elif attr.attr == "duration":
                    unit = r"\micro\second"
                for k, v in attr.to_dref(unit).items():
                    ret[f"data/{name}/{attr_name}/{k}"] = v
                e_static = static_quality[name][attr_name]
                ret[f"error/static/{name}/{attr_name}/mae"] = (e_static["mae"], unit)
                ret[f"error/static/{name}/{attr_name}/mape"] = (
                    e_static["mape"],
                    r"\percent",
                )
                ret[f"error/static/{name}/{attr_name}/smape"] = (
                    e_static["smape"],
                    r"\percent",
                )
                try:
                    ret[f"error/static/{name}/{attr_name}/mape"] = (
                        e_static["mape"],
                        r"\percent",
                    )
                except KeyError:
                    logger.warning(f"{name} {attr_name} static model has no MAPE")

                if lut_quality is not None:
                    e_lut = lut_quality[name][attr_name]
                    ret[f"error/lut/{name}/{attr_name}/mae"] = (e_lut["mae"], unit)
                    ret[f"error/lut/{name}/{attr_name}/mape"] = (
                        e_lut["mape"],
                        r"\percent",
                    )
                    ret[f"error/lut/{name}/{attr_name}/smape"] = (
                        e_lut["smape"],
                        r"\percent",
                    )
                    try:
                        ret[f"error/lut/{name}/{attr_name}/mape"] = (
                            e_lut["mape"],
                            r"\percent",
                        )
                    except KeyError:
                        logger.warning(f"{name} {attr_name} LUT model has no MAPE")

                e_model = model_quality[name][attr_name]
                ret[f"error/model/{name}/{attr_name}/mae"] = (e_model["mae"], unit)
                ret[f"error/model/{name}/{attr_name}/mape"] = (
                    e_model["mape"],
                    r"\percent",
                )
                ret[f"error/model/{name}/{attr_name}/smape"] = (
                    e_model["smape"],
                    r"\percent",
                )
                try:
                    ret[f"error/model/{name}/{attr_name}/mape"] = (
                        e_model["mape"],
                        r"\percent",
                    )
                except KeyError:
                    logger.warning(f"{name} {attr_name} param model has no MAPE")

                if xv_models is not None:
                    keys = ("decision tree/nodes", "decision tree/max depth")
                    entry = dict()
                    for k in keys:
                        entry[k] = list()
                    for xv_model in xv_models:
                        dref = xv_model.attr_by_name[name][attr_name].to_dref()
                        for k in keys:
                            if k in dref:
                                entry[k].append(dref[k])
                    for k in keys:
                        if len(entry[k]):
                            ret[f"xv/{name}/{attr_name}/{k}"] = np.mean(entry[k])
        return ret

    def to_json(self, **kwargs) -> dict:
        """
        Return JSON encoding of this AnalyticModel.
        """
        ret = {
            "parameters": self.parameters,
            "name": dict([[name, dict()] for name in self.names]),
        }

        for name in self.names:
            for attr_name, attr in self.attr_by_name[name].items():
                ret["name"][name][attr_name] = attr.to_json(**kwargs)

        return ret

    def webconf_function_map(self) -> list:
        ret = list()
        for name in self.names:
            for attr_model in self.attr_by_name[name].values():
                ret.extend(attr_model.webconf_function_map())
        return ret

    def predict(self, trace, with_fitted=True, wth_lut=False):
        pass
        # TODO trace= ( (name, duration), (name, duration), ...)
        # -> Return predicted (duration, mean power, cumulative energy) for trace
        # Achtung: Teilweise schon in der PTA-Klasse implementiert. Am besten diese mitbenutzen.


class PTAModel(AnalyticModel):
    """
    Parameter-aware PTA-based energy model.

    Supports both static and parameter-based model attributes, and automatic detection of parameter-dependence.

    The model heavily relies on two internal data structures:
    PTAModel.by_name and PTAModel.by_param.

    These provide measurements aggregated by state/transition name
    and (for by_param) parameter values. Layout:
    dictionary with one key per state/transition ('send', 'TX', ...) or
    one key per state/transition and parameter combination
    (('send', (1, 2)), ('send', (2, 3)), ('TX', (1, 2)), ('TX', (2, 3)), ...).
    For by_param, parameter values are ordered corresponding to the lexically sorted parameter names.

    Each element is in turn a dict with the following elements:
    - isa: 'state' or 'transition'
    - power: list of mean power measurements in µW
    - duration: list of durations in µs
    - power_std: list of stddev of power per state/transition
    - energy: consumed energy (power*duration) in pJ
    - paramkeys: list of parameter names in each measurement (-> list of lists)
    - param: list of parameter values in each measurement (-> list of lists)
    - attributes: list of keys that should be analyzed,
        e.g. ['power', 'duration']
    additionally, only if isa == 'transition':
    - timeout: list of duration of previous state in µs
    - rel_energy_prev: transition energy relative to previous state mean power in pJ
    - rel_energy_next: transition energy relative to next state mean power in pJ
    - rel_power_prev: transition power relative to previous state mean power in µW
    - rel_power_next: transition power relative to next state mean power in µW
    """

    def __init__(
        self,
        by_name,
        parameters,
        arg_count,
        traces=[],
        ignore_trace_indexes=[],
        function_override={},
        use_corrcoef=False,
        pta=None,
        pelt=None,
        compute_stats=True,
        dtree_max_std=None,
        force_tree=False,
    ):
        """
        Prepare a new PTA energy model.

        Actual model generation is done on-demand by calling the respective functions.

        arguments:
        by_name -- state/transition measurements aggregated by name, as returned by pta_trace_to_aggregate.
        parameters -- list of parameter names, as returned by pta_trace_to_aggregate
        arg_count -- function arguments, as returned by pta_trace_to_aggregate
        traces -- list of preprocessed DFA traces, as returned by RawData.get_preprocessed_data()
        ignore_trace_indexes -- list of trace indexes. The corresponding traces will be ignored.
        function_override -- dict of overrides for automatic parameter function generation.
            If (state or transition name, model attribute) is present in function_override,
            the corresponding text string is the function used for analytic (parameter-aware/fitted)
            modeling of this attribute. It is passed to AnalyticFunction, see
            there for the required format. Note that this happens regardless of
            parameter dependency detection: The provided analytic function will be assigned
            even if it seems like the model attribute is static / parameter-independent.
        use_corrcoef -- use correlation coefficient instead of stddev comparison
            to detect whether a model attribute depends on a parameter
        pta -- hardware model as `PTA` object
        pelt -- perform sub-state detection via PELT and model sub-states as well. Requires traces to be set.
        """
        self.by_name = by_name
        self.attr_by_name = dict()
        self.by_param = by_name_to_by_param(by_name)

        self.names = sorted(by_name.keys())
        self.states = sorted(
            list(
                filter(lambda k: self.by_name[k]["isa"] == "state", self.by_name.keys())
            )
        )
        self.transitions = sorted(
            list(
                filter(
                    lambda k: self.by_name[k]["isa"] == "transition",
                    self.by_name.keys(),
                )
            )
        )
        self.states_and_transitions = self.states + self.transitions
        self.dtree_max_std = dtree_max_std

        self._parameter_names = sorted(parameters)
        self.parameters = sorted(parameters)
        self._num_args = arg_count
        self._use_corrcoef = use_corrcoef
        self.traces = traces
        self.function_override = function_override.copy()
        self.submodel_by_name = dict()
        self.substate_sequence_by_nc = dict()
        self.pta = pta
        self.ignore_trace_indexes = ignore_trace_indexes

        self.distinct_param_values_by_name = dict()
        self.param_type_by_name = dict()
        for name in self.names:
            self.distinct_param_values_by_name[name] = distinct_param_values(
                by_name[name]["param"]
            )
            self.param_type_by_name[name] = ParamType(
                self.distinct_param_values_by_name[name], values_are_distinct=True
            )

        self.fit_done = False

        if traces is not None and pelt is not None:
            from .pelt import PELT

            self.pelt = PELT(**pelt)
            # must run before _compute_stats so that _compute_stats produces a "substate_count" model
            self.find_substates()
        else:
            self.pelt = None

        self._aggregate_to_ndarray(self.by_name)

        if compute_stats:
            self._compute_stats(by_name)

        if force_tree:
            for name in self.names:
                for attr in self.by_name[name]["attributes"]:
                    if (
                        dtree_max_std
                        and name in dtree_max_std
                        and attr in dtree_max_std[name]
                    ):
                        threshold = dtree_max_std[name][attr]
                    elif compute_stats:
                        threshold = (self.attr_by_name[name][attr].stats.std_param_lut,)
                    else:
                        threshold = 0
                    logger.debug(f"build_dtree({name}, {attr}, threshold={threshold})")
                    self.build_dtree(
                        name,
                        attr,
                        threshold=threshold,
                    )
            self.fit_done = True

        if self.pelt is not None:
            # cluster_substates uses self.attr_by_name[*]["power"].param_values, which is set by _compute_stats
            # cluster_substates relies on fitted "substate_count" models, which are generated by get_fitted.
            self.get_fitted()
            # cluster_substates alters submodel_by_name, so we cannot use its keys() iterator.
            names_with_submodel = list(self.submodel_by_name.keys())
            for name in names_with_submodel:
                self.cluster_substates(name)

        np.seterr("raise")

    def __repr__(self):
        states = ", ".join(self.states)
        transitions = ", ".join(self.transitions)
        return f"PTAModel<states=[{states}], transitions=[{transitions}]>"

    def _aggregate_to_ndarray(self, aggregate):
        for elem in aggregate.values():
            for key in elem["attributes"]:
                elem[key] = np.array(elem[key])

    # This heuristic is very similar to the "function is not much better than
    # median" checks in get_fitted. So far, doing it here as well is mostly
    # a performance and not an algorithm quality decision.
    # --df, 2018-04-18
    def depends_on_param(self, state_or_trans, key, param):
        return self.stats.depends_on_param(state_or_trans, key, param)

    # See notes on depends_on_param
    def depends_on_arg(self, state_or_trans, key, param):
        return self.stats.depends_on_arg(state_or_trans, key, param)

    def _get_model_from_dict(self, model_dict, model_function):
        model = {}
        for name, elem in model_dict.items():
            model[name] = {}
            for key in elem["attributes"]:
                try:
                    model[name][key] = model_function(elem[key])
                except RuntimeWarning:
                    logger.warning("Got no data for {} {}".format(name, key))
                except FloatingPointError as fpe:
                    logger.warning("Got no data for {} {}: {}".format(name, key, fpe))
        return model

    def pelt_refine(self, by_param_key):
        logger.debug(f"PELT: {by_param_key} needs refinement")

        penalty_by_trace = list()
        changepoints_by_penalty_by_trace = list()
        num_changepoints_by_trace = list()
        changepoints_by_trace = list()

        pelt_results = self.pelt.get_penalty_and_changepoints(
            self.by_param[by_param_key]["power_traces"]
        )

        for penalty, changepoints_by_penalty in pelt_results:
            penalty_by_trace.append(penalty)
            changepoints_by_penalty_by_trace.append(changepoints_by_penalty)
            num_changepoints_by_trace.append(len(changepoints_by_penalty[penalty]))
            changepoints_by_trace.append(changepoints_by_penalty[penalty])

        if np.median(num_changepoints_by_trace) < 1:
            logger.debug(
                f"    we found no changepoints {num_changepoints_by_trace} with penalties {penalty_by_trace}"
            )
            substate_counts = [1 for i in self.by_param[by_param_key]["param"]]
            substate_data = [
                {
                    "duration": self.by_param[by_param_key]["duration"],
                    "power": self.by_param[by_param_key]["power"],
                    "power_std": self.by_param[by_param_key]["power_std"],
                }
            ]
            return 1, (substate_counts, substate_data)

        num_changepoints = np.argmax(np.bincount(num_changepoints_by_trace))

        logger.debug(
            f"    we found {num_changepoints} changepoints {num_changepoints_by_trace} with penalties {penalty_by_trace}"
        )
        return (
            num_changepoints + 1,
            self.pelt.calc_raw_states(
                self.by_param[by_param_key]["timestamps"],
                self.by_param[by_param_key]["power_traces"],
                changepoints_by_trace,
                num_changepoints,
            ),
        )

    def find_substates(self):
        """
        Finds substates via PELT and adds substate_count to by_name and by_param.
        """
        substates_by_param = dict()
        for k in self.by_param.keys():
            if (
                self.pelt.name_filter is None or k[0] == self.pelt.name_filter
            ) and self.pelt.needs_refinement(self.by_param[k]["power_traces"]):
                num_substates, (substate_counts, substate_data) = self.pelt_refine(k)
                # substate_data[substate index]["power"] = [mean power of substate in first iteration, ...]
                substates_by_param[k] = (num_substates, substate_counts, substate_data)
            else:
                substate_counts = [1 for i in self.by_param[k]["param"]]
                substates_by_param[k] = (1, substate_counts, None)

        # suitable for AEMR modeling
        sc_by_param = dict()
        for param_key, (_, substate_counts, _) in substates_by_param.items():
            # do not append "substate_count" to "attributes" here.
            # by_param[(foo, *)]["attributes"] is the same object as by_name[foo]["attributes"]
            self.by_param[param_key]["substate_count"] = substate_counts

        for state_name in self.names:
            param_offset = dict()
            state = self.by_name[state_name]
            state["attributes"].append("substate_count")
            state["substate_count"] = list()
            for i, param in enumerate(state["param"]):
                param = tuple(param)
                if param not in param_offset:
                    param_offset[param] = 0
                state["substate_count"].append(
                    self.by_param[(state_name, param)]["substate_count"][
                        param_offset[param]
                    ]
                )
                param_offset[param] += 1

        substate_counts_by_name = dict()
        for k, (num_substates, _, _) in substates_by_param.items():
            if k[0] not in substate_counts_by_name:
                substate_counts_by_name[k[0]] = set()
            substate_counts_by_name[k[0]].add(num_substates)

        for name in self.names:
            data = dict()
            substate_counts = list()
            for substate_count in substate_counts_by_name[name]:
                sub_data = list()
                for k, (num_substates, _, substate_data) in substates_by_param.items():
                    if (
                        k[0] == name
                        and substate_count > 1
                        and num_substates == substate_count
                    ):
                        sub_data.append((k[1], substate_data))
                if len(sub_data):
                    data[substate_count] = sub_data
                    substate_counts.append(substate_count)
            if len(data):
                self.mk_submodel(name, substate_counts, data)

    def cluster_substates(self, p_name):
        from sklearn.cluster import AgglomerativeClustering

        submodel = self.submodel_by_name[p_name]
        # Für nicht parameterabhängige Teilzustände:
        # - Dauer ± max(1%, 20µs) -> merge OK
        # - Leistung ± max(5%, 10 µW) -> merge OK
        # Besser in zwei Schritten oder besser gemeinsam? Das Problem ist, dass die distance_threshold nicht nach
        # Dimensionen unterscheidet.
        # Für parameterabhängige / allgemein: param_lut statt static nutzen.
        # values_to_cluster[i, 0] = duration für paramvektor 1 (fallback static duration)
        # values_to_cluster[i, 1] = duration für paramvektor 2 (fallback static duration)
        # etc. -> wenn die lut für alle Parameter ähnlich ist, wird gemerged. Das funktioniert auch bei geringfügigen
        # Schwankungen, die beim separaten Fitting zu unterschiedlichen Funktionen führen würden.
        p_attr = self.attr_by_name[p_name]["power"]
        p_params = list(set(map(tuple, p_attr.param_values)))
        sub_attr_by_function = dict()
        static = submodel.get_static()
        lut = submodel.get_param_lut(fallback=True)
        values_to_cluster = np.zeros((len(submodel.names), len(p_params)))
        for i, name in enumerate(submodel.names):
            for j, param in enumerate(p_params):
                values_to_cluster[i, j] = lut(name, "duration", param=param)

        clusters = list()

        d_cluster = AgglomerativeClustering(
            n_clusters=None,
            compute_full_tree=True,
            affinity="euclidean",
            linkage="ward",
            distance_threshold=50,
        )
        d_cluster.fit_predict(values_to_cluster)

        for d_cluster_i in range(d_cluster.n_clusters_):
            cl_substates = list()
            for i, name in enumerate(submodel.names):
                if d_cluster.labels_[i] == d_cluster_i:
                    cl_substates.append(name)
            if len(cl_substates) == 1:
                clusters.append(cl_substates)
                continue
            values_to_cluster = np.zeros((len(cl_substates), len(p_params)))
            for i, name in enumerate(cl_substates):
                for j, param in enumerate(p_params):
                    values_to_cluster[i, j] = lut(name, "power", param=param)
            p_cluster = AgglomerativeClustering(
                n_clusters=None,
                compute_full_tree=True,
                affinity="euclidean",
                linkage="ward",
                distance_threshold=500,
            )
            p_cluster.fit_predict(values_to_cluster)
            for p_cluster_i in range(p_cluster.n_clusters_):
                cluster = list()
                for i, name in enumerate(cl_substates):
                    if p_cluster.labels_[i] == p_cluster_i:
                        cluster.append(name)
                clusters.append(cluster)

        logger.debug(f"sub-state clusters = {clusters}")

        by_name = dict()
        new_subname_by_old = dict()
        for i, cluster in enumerate(clusters):
            sub_name = f"{p_name}.{i}"
            durations = list()
            powers = list()
            param_values = list()
            for substate in cluster:
                new_subname_by_old[substate] = sub_name
                durations.extend(submodel.attr_by_name[substate]["duration"].data)
                powers.extend(submodel.attr_by_name[substate]["power"].data)
                param_values.extend(
                    submodel.attr_by_name[substate]["power"].param_values
                )
            by_name[sub_name] = {
                "isa": "state",
                "param": param_values,
                "attributes": ["duration", "power"],
                "duration": durations,
                "power": powers,
            }
        self.submodel_by_name[p_name] = PTAModel(by_name, self.parameters, dict())
        sequence_by_count = dict()
        for name, count in self.substate_sequence_by_nc.keys():
            if name == p_name:
                sequence_by_count[int(count)] = list(
                    map(
                        lambda x: new_subname_by_old[x],
                        self.substate_sequence_by_nc[(name, count)],
                    )
                )

        self.attr_by_name[p_name]["power"].model_function = SubstateFunction(
            self.attr_by_name[p_name]["power"].get_static(),
            sequence_by_count,
            self.attr_by_name[p_name]["substate_count"].model_function,
            self.submodel_by_name[p_name],
        )

    # data[0] = [first sub-state, second sub-state, ...]
    # data[1] = [first sub-state, second sub-state, ...]
    # ...
    def mk_submodel(self, name, substate_counts, data):
        paramstats = ParallelParamStats()
        by_name = dict()
        sub_states = list()

        for substate_count in substate_counts:
            self.substate_sequence_by_nc[(name, substate_count)] = list()
            for substate_index in range(substate_count):
                sub_name = f"{name}.{substate_index+1}({substate_count})"
                self.substate_sequence_by_nc[(name, substate_count)].append(sub_name)
                durations = list()
                powers = list()
                param_values = list()
                for param, run in data[substate_count]:
                    # data units are s / W, models use µs / µW
                    durations.extend(np.array(run[substate_index]["duration"]) * 1e6)
                    powers.extend(np.array(run[substate_index]["power"]) * 1e6)
                    param_values.extend(
                        [list(param) for i in run[substate_index]["duration"]]
                    )

                by_name[sub_name] = {
                    "isa": "state",
                    "param": param_values,
                    "attributes": ["duration", "power"],
                    "duration": durations,
                    "power": powers,
                }

        self.submodel_by_name[name] = PTAModel(by_name, self.parameters, dict())

    def to_json(self, **kwargs):
        static_model = self.get_static()
        static_quality = self.assess(static_model)
        param_model, param_info = self.get_fitted()
        analytic_quality = self.assess(param_model)
        pta = self.pta
        if pta is None:
            pta = PTA(self.states, parameters=self._parameter_names)
            if self.traces:
                logger.warning(
                    "to_json: PTA is unavailable. Transitions may have incorrect or incomplete origin/destination states."
                )
            else:
                logger.warning(
                    "to_json: Neither PTA nor traces are available. Falling back to incorrectly mapping all transitions as UNINITIALIZED -> UNINITIALIZED."
                )
            for transition in self.transitions:
                for origin, destination in self.get_transition_states_from_traces(
                    transition
                ):
                    pta.add_transition(origin, destination, transition)
        pta.update(
            param_info, static_error=static_quality, function_error=analytic_quality
        )
        return pta.to_json(**kwargs)

    def to_dot(self) -> str:
        param_model, param_info = self.get_fitted()
        pta = self.pta
        if pta is None:
            pta = PTA(self.states, parameters=self._parameter_names)
            for transition in self.transitions:
                for origin, destination in self.get_transition_states_from_traces(
                    transition
                ):
                    pta.add_transition(origin, destination, transition)
        pta.update(param_info)
        return pta.to_dot()

    def get_transition_states_from_traces(self, transition_name):
        if self.traces is None:
            return [("UNINITIALIZED", "UNINITIALIZED")]
        pairs = set()
        for trace in self.traces:
            trace = trace["trace"]
            for i, tos in enumerate(trace):
                if (
                    i == 0
                    and tos["isa"] == "transition"
                    and tos["name"] == transition_name
                ):
                    pairs.add(("UNINITIALIZED", trace[i + 1]["name"]))
                elif (
                    i + 1 < len(trace)
                    and tos["isa"] == "transition"
                    and tos["name"] == transition_name
                ):
                    pairs.add((trace[i - 1]["name"], trace[i + 1]["name"]))
        return list(pairs)

    def assess(self, model_function, ref=None, return_raw=False):
        """
        Calculate MAE, SMAPE, etc. of model_function for each by_name entry.

        state/transition/... name and parameter values are fed into model_function.
        The by_name entries of this PTAModel are used as ground truth and
        compared with the values predicted by model_function.

        For proper model assessments, the data used to generate model_function
        and the data fed into this AnalyticModel instance must be mutually
        exclusive (e.g. by performing cross validation). Otherwise,
        overfitting cannot be detected.
        """
        if ref is None:
            ref = self.by_name
        if return_raw:
            detailed_results, raw_results = super().assess(
                model_function, ref=ref, return_raw=return_raw
            )
        else:
            detailed_results = super().assess(model_function, ref=ref)
        for name, elem in sorted(ref.items()):
            if elem["isa"] == "transition":
                predicted_data = np.array(
                    list(
                        map(
                            lambda i: model_function(
                                name, "power", param=elem["param"][i]
                            )
                            * model_function(name, "duration", param=elem["param"][i]),
                            range(len(elem["power"])),
                        )
                    )
                )
                measures = regression_measures(
                    predicted_data, elem["power"] * elem["duration"]
                )
                detailed_results[name]["energy_Pt"] = measures
                if return_raw:
                    raw_results[name]["attribute"]["energy_Pt"] = {
                        "groundTruth": list(elem["power"] * elem["duration"]),
                        "modelOutput": list(predicted_data),
                    }

        if return_raw:
            return detailed_results, raw_results
        return detailed_results

    def assess_states(
        self, model_function, model_attribute="power", distribution: dict = None
    ):
        """
        Calculate overall model error assuming equal distribution of states
        """
        # TODO calculate mean power draw for distribution and use it to
        # calculate relative error from MAE combination
        model_quality = self.assess(model_function)
        num_states = len(self.states)
        if distribution is None:
            distribution = dict(map(lambda x: [x, 1 / num_states], self.states))

        if not np.isclose(sum(distribution.values()), 1):
            raise ValueError(
                "distribution must be a probability distribution with sum 1"
            )

        # total_value = None
        # try:
        #     total_value = sum(map(lambda x: model_function(x, model_attribute) * distribution[x], self.states))
        # except KeyError:
        #     pass

        total_error = np.sqrt(
            sum(
                map(
                    lambda x: np.square(
                        model_quality[x][model_attribute]["mae"] * distribution[x]
                    ),
                    self.states,
                )
            )
        )
        return total_error

    def assess_on_traces(self, model_function):
        """
        Calculate MAE, SMAPE, etc. of model_function for each trace known to this PTAModel instance.

        :returns: dict of `duration_by_trace`, `energy_by_trace`, `timeout_by_trace`, `rel_energy_by_trace` and `state_energy_by_trace`.
            Each entry holds regression measures for the corresponding measure. Note that the determined model quality heavily depends on the
            traces: small-ish absolute errors in states which frequently occur may have more effect than large absolute errors in rarely occuring states
        """
        model_energy_list = []
        real_energy_list = []
        model_rel_energy_list = []
        model_state_energy_list = []
        model_duration_list = []
        real_duration_list = []
        model_timeout_list = []
        real_timeout_list = []

        for trace in self.traces:
            if trace["id"] not in self.ignore_trace_indexes:
                for rep_id in range(len(trace["trace"][0]["offline"])):
                    model_energy = 0.0
                    real_energy = 0.0
                    model_rel_energy = 0.0
                    model_state_energy = 0.0
                    model_duration = 0.0
                    real_duration = 0.0
                    model_timeout = 0.0
                    real_timeout = 0.0
                    for i, trace_part in enumerate(trace["trace"]):
                        name = trace_part["name"]
                        prev_name = trace["trace"][i - 1]["name"]
                        isa = trace_part["isa"]
                        if name != "UNINITIALIZED":
                            try:
                                param = trace_part["offline_aggregates"]["param"][
                                    rep_id
                                ]
                                prev_param = trace["trace"][i - 1][
                                    "offline_aggregates"
                                ]["param"][rep_id]
                                power = trace_part["offline"][rep_id]["uW_mean"]
                                duration = trace_part["offline"][rep_id]["us"]
                                prev_duration = trace["trace"][i - 1]["offline"][
                                    rep_id
                                ]["us"]
                                real_energy += power * duration
                                if isa == "state":
                                    model_energy += (
                                        model_function(name, "power", param=param)
                                        * duration
                                    )
                                else:
                                    model_energy += model_function(
                                        name, "energy", param=param
                                    )
                                    # If i == 1, the previous state was UNINITIALIZED, for which we do not have model data
                                    if i == 1:
                                        model_rel_energy += model_function(
                                            name, "energy", param=param
                                        )
                                    else:
                                        model_rel_energy += model_function(
                                            prev_name, "power", param=prev_param
                                        ) * (prev_duration + duration)
                                        model_state_energy += model_function(
                                            prev_name, "power", param=prev_param
                                        ) * (prev_duration + duration)
                                    model_rel_energy += model_function(
                                        name, "rel_energy_prev", param=param
                                    )
                                    real_duration += duration
                                    model_duration += model_function(
                                        name, "duration", param=param
                                    )
                                    if (
                                        "plan" in trace_part
                                        and trace_part["plan"]["level"] == "epilogue"
                                    ):
                                        real_timeout += trace_part["offline"][rep_id][
                                            "timeout"
                                        ]
                                        model_timeout += model_function(
                                            name, "timeout", param=param
                                        )
                            except KeyError:
                                # if states/transitions have been removed via --filter-param, this is harmless
                                pass
                    real_energy_list.append(real_energy)
                    model_energy_list.append(model_energy)
                    model_rel_energy_list.append(model_rel_energy)
                    model_state_energy_list.append(model_state_energy)
                    real_duration_list.append(real_duration)
                    model_duration_list.append(model_duration)
                    real_timeout_list.append(real_timeout)
                    model_timeout_list.append(model_timeout)

        return {
            "duration_by_trace": regression_measures(
                np.array(model_duration_list), np.array(real_duration_list)
            ),
            "energy_by_trace": regression_measures(
                np.array(model_energy_list), np.array(real_energy_list)
            ),
            "timeout_by_trace": regression_measures(
                np.array(model_timeout_list), np.array(real_timeout_list)
            ),
            "rel_energy_by_trace": regression_measures(
                np.array(model_rel_energy_list), np.array(real_energy_list)
            ),
            "state_energy_by_trace": regression_measures(
                np.array(model_state_energy_list), np.array(real_energy_list)
            ),
        }
