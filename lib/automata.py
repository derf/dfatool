"""Classes and helper functions for PTA and other automata."""

from .functions import AnalyticFunction, NormalizationFunction
from .utils import is_numeric
import itertools
import numpy as np
import json
import queue
import yaml


def _dict_to_list(input_dict: dict) -> list:
    return [input_dict[x] for x in sorted(input_dict.keys())]


class SimulationResult:
    """
    Duration, Energy, and state/parameter results from PTA.simulate on a single run.

    :param duration: run duration in s
    :param duration_mae: Mean Absolute Error of duration, assuming cycle-perfect delay/sleep calls
    :param duration_mape: Mean Absolute Percentage Error of duration, assuming cycle-perfect delay/sleep caals
    :param energy: run energy in J
    :param energy_mae: Mean Absolute Error of energy
    :param energy_mape: Mean Absolute Percentage Error of energy
    :param end_state: Final `State` of run
    :param parameters: Final parameters of run
    :param mean_power: mean power during run in W
    """

    def __init__(
        self,
        duration: float,
        energy: float,
        end_state,
        parameters,
        duration_mae: float = None,
        energy_mae: float = None,
    ):
        u"""
        Create a new SimulationResult.

        :param duration: run duration in µs
        :param duration_mae: Mean Absolute Error of duration in µs, default None
        :param energy: run energy in pJ
        :param energy_mae: Mean Absolute Error of energy in pJ, default None
        :param end_state: Final `State`  after simulation run
        :param parameters: Parameter values after simulation run
        """
        self.duration = duration * 1e-6
        if duration_mae is None or self.duration == 0:
            self.duration_mae = None
            self.duration_mape = None
        else:
            self.duration_mae = duration_mae * 1e-6
            self.duration_mape = self.duration_mae * 100 / self.duration
        self.energy = energy * 1e-12
        if energy_mae is None or self.energy == 0:
            self.energy_mae = None
            self.energy_mape = None
        else:
            self.energy_mae = energy_mae * 1e-12
            self.energy_mape = self.energy_mae * 100 / self.energy
        self.end_state = end_state
        self.parameters = parameters
        if self.duration > 0:
            self.mean_power = self.energy / self.duration
        else:
            self.mean_power = 0


class PTAAttribute:
    u"""
    A single PTA attribute (e.g. power, duration).

    A PTA attribute can be described by a static value and an analytic
    function (depending on parameters and function arguments).

    It is not specified how value_error and function_error are determined --
    at the moment, they do not use cross validation.

    :param value: static value, typically in µW/µs/pJ
    :param value_error: mean absolute error of value (optional)
    :param function: AnalyticFunction for parameter-aware prediction (optional)
    :param function_error: mean absolute error of function (optional)
    """

    def __init__(
        self,
        value: float = 0,
        function: AnalyticFunction = None,
        value_error=None,
        function_error=None,
    ):
        self.value = value
        self.function = function
        self.value_error = value_error
        self.function_error = function_error

    def __repr__(self):
        if self.function is not None:
            return "PTAATtribute<{:.0f}, {}>".format(
                self.value, self.function._model_str
            )
        return "PTAATtribute<{:.0f}, None>".format(self.value)

    def eval(self, param_dict=dict(), args=list()):
        """
        Return attribute for given `param_dict` and `args` value.

        Uses `function` if set and usable for the given `param_dict` and
        `value` otherwise.
        """
        param_list = _dict_to_list(param_dict)
        if self.function and self.function.is_predictable(param_list):
            return self.function.eval(param_list, args)
        return self.value

    def eval_mae(self, param_dict=dict(), args=list()):
        """
        Return attribute mean absolute error for given `param_dict` and `args` value.

        Uses `function_error` if `function` is set and usable for the given `param_dict` and `value_error` otherwise.
        """
        param_list = _dict_to_list(param_dict)
        if self.function and self.function.is_predictable(param_list):
            return self.function_error["mae"]
        return self.value_error["mae"]

    def to_json(self):
        ret = {
            "static": self.value,
            "static_error": self.value_error,
        }
        if self.function:
            ret["function"] = {
                "raw": self.function._model_str,
                "regression_args": list(self.function._regression_args),
            }
            ret["function_error"] = self.function_error
        return ret

    @classmethod
    def from_json(cls, json_input: dict, parameters: dict):
        ret = cls()
        if "static" in json_input:
            ret.value = json_input["static"]
        if "static_error" in json_input:
            ret.value_error = json_input["static_error"]
        if "function" in json_input:
            ret.function = AnalyticFunction(
                json_input["function"]["raw"],
                parameters,
                0,
                regression_args=json_input["function"]["regression_args"],
            )
        if "function_error" in json_input:
            ret.function_error = json_input["function_error"]
        return ret

    @classmethod
    def from_json_maybe(cls, json_wrapped: dict, attribute: str, parameters: dict):
        if type(json_wrapped) is dict and attribute in json_wrapped:
            return cls.from_json(json_wrapped[attribute], parameters)
        return cls()


def _json_function_to_analytic_function(base, attribute: str, parameters: list):
    if attribute in base and "function" in base[attribute]:
        base = base[attribute]["function"]
        return AnalyticFunction(
            base["raw"], parameters, 0, regression_args=base["regression_args"]
        )
    return None


class State:
    """A single PTA state."""

    def __init__(
        self,
        name: str,
        power: PTAAttribute = PTAAttribute(),
        power_function: AnalyticFunction = None,
    ):
        u"""
        Create a new PTA state.

        :param name: state name
        :param power: state power PTAAttribute in µW, default static 0 / parameterized None
        :param power_function: Legacy support
        """
        self.name = name
        self.power = power
        self.outgoing_transitions = {}

        if type(self.power) is float or type(self.power) is int:
            self.power = PTAAttribute(self.power)

        if power_function is not None:
            if type(power_function) is AnalyticFunction:
                self.power.function = power_function
            else:
                raise ValueError("power_function must be None or AnalyticFunction")

    def __repr__(self):
        return "State<{:s}, {}>".format(self.name, self.power)

    def add_outgoing_transition(self, new_transition: object):
        """Add a new outgoing transition."""
        self.outgoing_transitions[new_transition.name] = new_transition

    def get_energy(self, duration: float, param_dict: dict = {}) -> float:
        u"""
        Return energy spent in state in pJ.

        :param duration: duration in µs
        :param param_dict: current parameters
        :returns: energy spent in pJ
        """
        return self.power.eval(param_dict) * duration

    def set_random_energy_model(self, static_model=True):
        u"""Set a random static state power between 0 µW and 50 mW."""
        self.power.value = int(np.random.sample() * 50000)

    def get_transition(self, transition_name: str) -> object:
        """
        Return Transition object for outgoing transtion transition_name.

        :param transition_name: transition name
        :returns: `Transition` object
        """
        try:
            return self.outgoing_transitions[transition_name]
        except KeyError:
            raise ValueError(
                "State {} has no outgoing transition called {}".format(
                    self.name, transition_name
                )
            ) from None

    def has_interrupt_transitions(self) -> bool:
        """Return whether this state has any outgoing interrupt transitions."""
        for trans in self.outgoing_transitions.values():
            if trans.is_interrupt:
                return True
        return False

    def get_next_interrupt(self, parameters: dict) -> object:
        """
        Return the outgoing interrupt transition with the lowet timeout.

        Must only be called if has_interrupt_transitions returned true.

        :param parameters: current parameter values
        :returns: Transition object
        """
        interrupts = filter(
            lambda x: x.is_interrupt, self.outgoing_transitions.values()
        )
        interrupts = sorted(interrupts, key=lambda x: x.get_timeout(parameters))
        return interrupts[0]

    def dfs(self, depth: int, with_arguments=False, trace_filter=None, sleep=0):
        """
        Return a generator object for depth-first search over all outgoing transitions.

        :param depth: search depth
        :param with_arguments: perform dfs with function+argument transitions instead of just function transitions.
        :param trace_filter: list of lists. Each sub-list is a trace. Only traces matching one of the provided sub-lists are returned.
            E.g. trace_filter = [['init', 'foo'], ['init', 'bar']] will only return traces with init as first and foo or bar as second element.
            trace_filter = [['init', 'foo', '$'], ['init', 'bar', '$']] will only return the traces ['init', 'foo'] and ['init', 'bar'].
            Note that `trace_filter` takes precedence over `depth`: traces matching `trace_filter` are generated even if their length exceeds `depth`
        :param sleep: if set and non-zero: include sleep pseudo-states with <sleep> us duration
            For the [['init', 'foo', '$'], ['init', 'bar', '$']] example above, sleep=10 results in [(None, 10), 'init', (None, 10), 'foo'] and [(None, 10), 'init', (None, 10), 'bar']
        :returns: Generator object for depth-first search. Each access yields a list of (Transition, (arguments)) elements describing a single run through the PTA.
        """

        # TODO parametergewahrer Trace-Filter, z.B. "setHeaterDuration nur wenn bme680 power mode => FORCED und GAS_ENABLED"

        # A '$' entry in trace_filter indicates that the trace should (successfully) terminate here regardless of `depth`.
        if (
            trace_filter is not None
            and next(
                filter(lambda x: x == "$", map(lambda x: x[0], trace_filter)), None
            )
            is not None
        ):
            yield []
        # there may be other entries in trace_filter that still yield results.
        if depth == 0:
            for trans in self.outgoing_transitions.values():
                if (
                    trace_filter is not None
                    and len(
                        list(
                            filter(
                                lambda x: x == trans.name,
                                map(lambda x: x[0], trace_filter),
                            )
                        )
                    )
                    == 0
                ):
                    continue
                if with_arguments:
                    if trans.argument_combination == "cartesian":
                        for args in itertools.product(*trans.argument_values):
                            if sleep:
                                yield [(None, sleep), (trans, args)]
                            else:
                                yield [(trans, args)]
                    else:
                        for args in zip(*trans.argument_values):
                            if sleep:
                                yield [(None, sleep), (trans, args)]
                            else:
                                yield [(trans, args)]
                else:
                    if sleep:
                        yield [(None, sleep), (trans,)]
                    else:
                        yield [(trans,)]
        else:
            for trans in self.outgoing_transitions.values():
                if (
                    trace_filter is not None
                    and next(
                        filter(
                            lambda x: x == trans.name, map(lambda x: x[0], trace_filter)
                        ),
                        None,
                    )
                    is None
                ):
                    continue
                if trace_filter is not None:
                    new_trace_filter = map(
                        lambda x: x[1:],
                        filter(lambda x: x[0] == trans.name, trace_filter),
                    )
                    new_trace_filter = list(filter(len, new_trace_filter))
                    if len(new_trace_filter) == 0:
                        new_trace_filter = None
                else:
                    new_trace_filter = None
                for suffix in trans.destination.dfs(
                    depth - 1,
                    with_arguments=with_arguments,
                    trace_filter=new_trace_filter,
                    sleep=sleep,
                ):
                    if with_arguments:
                        if trans.argument_combination == "cartesian":
                            for args in itertools.product(*trans.argument_values):
                                if sleep:
                                    new_suffix = [(None, sleep), (trans, args)]
                                else:
                                    new_suffix = [(trans, args)]
                                new_suffix.extend(suffix)
                                yield new_suffix
                        else:
                            if len(trans.argument_values):
                                arg_values = zip(*trans.argument_values)
                            else:
                                arg_values = [tuple()]
                            for args in arg_values:
                                if sleep:
                                    new_suffix = [(None, sleep), (trans, args)]
                                else:
                                    new_suffix = [(trans, args)]
                                new_suffix.extend(suffix)
                                yield new_suffix
                    else:
                        if sleep:
                            new_suffix = [(None, sleep), (trans,)]
                        else:
                            new_suffix = [(trans,)]
                        new_suffix.extend(suffix)
                        yield new_suffix

    def to_json(self) -> dict:
        """Return JSON encoding of this state object."""
        ret = {"name": self.name, "power": self.power.to_json()}
        return ret


class Transition:
    u"""
    A single PTA transition with one origin and one destination state.

    :param name: transition name, corresponds to driver function name
    :param origin: origin `State`
    :param destination: destination `State`
    :param energy: static energy needed to execute this transition, in pJ
    :param energy_function: parameterized transition energy `AnalyticFunction`, returning pJ
    :param duration: transition duration, in µs
    :param duration_function: parameterized duration `AnalyticFunction`, returning µs
    :param timeout: transition timeout, in µs. Only set for interrupt transitions.
    :param timeout_function: parameterized transition timeout `AnalyticFunction`, in µs. Only set for interrupt transitions.
    :param is_interrupt: Is this an interrupt transition?
    :param arguments: list of function argument names
    :param argument_values: list of argument values used for benchmark generation. Each entry is a list of values for the corresponding argument
    :param argument_combination: During benchmark generation, should arguments be combined via `cartesian` or `zip`?
    :param param_update_function: Setter for parameters after a transition. Gets current parameter dict and function argument values as arguments, must return the new parameter dict
    :param arg_to_param_map: dict mapping argument index to the name of the parameter affected by its value
    :param set_param: dict mapping parameter name to their value (set as side-effect of executing the transition, not parameter-dependent)
    :param return_value_handlers: todo
    :param codegen: todo
    """

    def __init__(
        self,
        orig_state: State,
        dest_state: State,
        name: str,
        energy: PTAAttribute = PTAAttribute(),
        energy_function: AnalyticFunction = None,
        duration: PTAAttribute = PTAAttribute(),
        duration_function: AnalyticFunction = None,
        timeout: PTAAttribute = PTAAttribute(),
        timeout_function: AnalyticFunction = None,
        is_interrupt: bool = False,
        arguments: list = [],
        argument_values: list = [],
        argument_combination: str = "cartesian",  # or 'zip'
        param_update_function=None,
        arg_to_param_map: dict = None,
        set_param=None,
        return_value_handlers: list = [],
        codegen=dict(),
    ):
        """
        Create a new transition between two PTA states.

        :param orig_state: origin `State`
        :param dest_state: destination `State`
        :param name: transition name, typically the same as a driver/library function name
        """
        self.name = name
        self.origin = orig_state
        self.destination = dest_state
        self.energy = energy
        self.duration = duration
        self.timeout = timeout
        self.is_interrupt = is_interrupt
        self.arguments = arguments.copy()
        self.argument_values = argument_values.copy()
        self.argument_combination = argument_combination
        self.param_update_function = param_update_function
        self.arg_to_param_map = arg_to_param_map
        self.set_param = set_param
        self.return_value_handlers = return_value_handlers
        self.codegen = codegen

        if type(self.energy) is float or type(self.energy) is int:
            self.energy = PTAAttribute(self.energy)
        if energy_function is not None:
            if type(energy_function) is AnalyticFunction:
                self.energy.function = energy_function

        if type(self.duration) is float or type(self.duration) is int:
            self.duration = PTAAttribute(self.duration)
        if duration_function is not None:
            if type(duration_function) is AnalyticFunction:
                self.duration.function = duration_function

        if type(self.timeout) is float or type(self.timeout) is int:
            self.timeout = PTAAttribute(self.timeout)
        if timeout_function is not None:
            if type(timeout_function) is AnalyticFunction:
                self.timeout.function = timeout_function

        for handler in self.return_value_handlers:
            if "formula" in handler:
                handler["formula"] = NormalizationFunction(handler["formula"])

    def get_duration(self, param_dict: dict = {}, args: list = []) -> float:
        u"""
        Return transition duration in µs.

        :param param_dict: current parameter values
        :param args: function arguments

        :returns: transition duration in µs
        """
        return self.duration.eval(param_dict, args)

    def get_energy(self, param_dict: dict = {}, args: list = []) -> float:
        u"""
        Return transition energy cost in pJ.

        :param param_dict: current parameter values
        :param args: function arguments
        """
        return self.energy.eval(param_dict, args)

    def set_random_energy_model(self, static_model=True):
        self.energy.value = int(np.random.sample() * 50000)
        self.duration.value = int(np.random.sample() * 50000)
        if self.is_interrupt:
            self.timeout.value = int(np.random.sample() * 50000)

    def get_timeout(self, param_dict: dict = {}) -> float:
        u"""
        Return transition timeout in µs.

        Returns 0 if the transition does not have a timeout.

        :param param_dict: current parameter values
        :param args: function arguments
        """
        return self.timeout.eval(param_dict)

    def get_params_after_transition(self, param_dict: dict, args: list = []) -> dict:
        """
        Return the new parameter dict after taking this transition.

        parameter values may be affected by this transition's update function,
        it's argument-to-param map, and its set_param settings.

        Does not normalize parameter values.
        """
        if self.param_update_function:
            return self.param_update_function(param_dict, args)
        ret = param_dict.copy()
        # set_param is for default values, arg_to_param_map may contain optional overrides.
        # So arg_to_param_map must come last.
        if self.set_param:
            for k, v in self.set_param.items():
                ret[k] = v
        if self.arg_to_param_map:
            for k, v in self.arg_to_param_map.items():
                ret[v] = args[k]
        return ret

    def to_json(self) -> dict:
        """Return JSON encoding of this transition object."""
        ret = {
            "name": self.name,
            "origin": self.origin.name,
            "destination": self.destination.name,
            "is_interrupt": self.is_interrupt,
            "arguments": self.arguments,
            "argument_values": self.argument_values,
            "argument_combination": self.argument_combination,
            "arg_to_param_map": self.arg_to_param_map,
            "set_param": self.set_param,
            "duration": self.duration.to_json(),
            "energy": self.energy.to_json(),
            "timeout": self.timeout.to_json(),
        }
        return ret


def _json_get_static(base, attribute: str):
    if attribute in base:
        return base[attribute]["static"]
    return 0


class PTA:
    """
    A parameterized priced timed automaton.

    Suitable for simulation, model storage, and (soon) benchmark generation.

    :param state: dict mapping state name to `State` object
    :param accepting_states: list of accepting state names
    :param parameters: current parameters
    :param parameter_normalization:  dict mapping driver API parameter values to hardware values, e.g. a bitrate register value to an actual bitrate in kbit/s.
            Each parameter key has in turn a dict value. Supported entries:
            `enum`: Mapping of enum descriptors (eys) to parameter values. Note that the mapping is not required to correspond to the driver API.
            `formula`: NormalizationFunction mapping an argument or return value (passed as `param`) to a parameter value.
    :param codegen: TODO
    :param initial_param_values: TODO
    :param transitions: list of `Transition` objects
    """

    def __init__(
        self,
        state_names: list = [],
        accepting_states: list = None,
        parameters: list = [],
        initial_param_values: list = None,
        codegen: dict = {},
        parameter_normalization: dict = None,
    ):
        """
        Return a new PTA object.

        :param state_names: names of PTA states. Note that the PTA always contains
            an initial UNINITIALIZED state, regardless of the content of state_names.
        :param accepting_states: names of accepting states. By default, all states are accepting
        :param parameters: names of PTA parameters
        :param initial_param_values: initial value for each parameter
        :param instance: class used for generated C++ code
        :param header: header include path for C++ class definition
        :param parameter_normalization: dict mapping driver API parameter values to hardware values, e.g. a bitrate register value to an actual bitrate in kbit/s.
            Each parameter key has in turn a dict value. Supported entries:
            `enum`: maps enum descriptors (keys) to parameter values. Note that the mapping is not required to correspond to the driver API.
            `formula`: maps an argument or return value (passed as `param`) to a parameter value. Must be a string describing a valid python lambda function. NumPy is available as `np`.
        """
        self.state = dict(
            [[state_name, State(state_name)] for state_name in state_names]
        )
        self.accepting_states = accepting_states.copy() if accepting_states else None
        self.parameters = parameters.copy()
        self.parameter_normalization = parameter_normalization
        self.codegen = codegen
        if initial_param_values:
            self.initial_param_values = initial_param_values.copy()
        else:
            self.initial_param_values = [None for x in self.parameters]
        self.transitions = []

        if "UNINITIALIZED" not in state_names:
            self.state["UNINITIALIZED"] = State("UNINITIALIZED")

        if self.parameter_normalization:
            for normalization_spec in self.parameter_normalization.values():
                if "formula" in normalization_spec:
                    normalization_spec["formula"] = NormalizationFunction(
                        normalization_spec["formula"]
                    )

    def normalize_parameter(self, parameter_name: str, parameter_value) -> float:
        """
        Return normalized parameter.

        Normalization refers to anything specified in the model's `parameter_normalization` section,
        e.g. enum -> int translation or argument -> parameter value formulas.

        :param parameter_name: parameter name.
        :param parameter_value: parameter value.
        """
        if (
            parameter_value is not None
            and self.parameter_normalization is not None
            and parameter_name in self.parameter_normalization
        ):
            if (
                "enum" in self.parameter_normalization[parameter_name]
                and parameter_value
                in self.parameter_normalization[parameter_name]["enum"]
            ):
                return self.parameter_normalization[parameter_name]["enum"][
                    parameter_value
                ]
            if "formula" in self.parameter_normalization[parameter_name]:
                normalization_formula = self.parameter_normalization[parameter_name][
                    "formula"
                ]
                return normalization_formula.eval(parameter_value)
        return parameter_value

    def normalize_parameters(self, param_dict) -> dict:
        """
        Return normalized parameters.

        Normalization refers to anything specified in the model's `parameter_normalization` section,
        e.g. enum -> int translation or argument -> parameter value formulas.

        :param param_dict: non-normalized parameters.
        """
        if self.parameter_normalization is None:
            return param_dict.copy()
        normalized_param = param_dict.copy()
        for parameter, value in param_dict.items():
            normalized_param[parameter] = self.normalize_parameter(parameter, value)
        return normalized_param

    @classmethod
    def from_file(cls, model_file: str):
        """Return PTA loaded from the provided JSON or YAML file."""
        with open(model_file, "r") as f:
            if ".json" in model_file:
                return cls.from_json(json.load(f))
            else:
                return cls.from_yaml(yaml.safe_load(f))

    @classmethod
    def from_json(cls, json_input: dict):
        """
        Return a PTA created from the provided JSON data.

        Compatible with the to_json method.
        """
        if "transition" in json_input:
            return cls.from_legacy_json(json_input)

        kwargs = dict()
        for key in (
            "state_names",
            "parameters",
            "initial_param_values",
            "accepting_states",
        ):
            if key in json_input:
                kwargs[key] = json_input[key]
        pta = cls(**kwargs)
        for name, state in json_input["state"].items():
            pta.add_state(
                name, power=PTAAttribute.from_json_maybe(state, "power", pta.parameters)
            )
        for transition in json_input["transitions"]:
            kwargs = dict()
            for key in [
                "arguments",
                "argument_values",
                "argument_combination",
                "is_interrupt",
                "set_param",
            ]:
                if key in transition:
                    kwargs[key] = transition[key]
            # arg_to_param_map uses integer indices. This is not supported by JSON
            if "arg_to_param_map" in transition:
                kwargs["arg_to_param_map"] = dict()
                for arg_index, param_name in transition["arg_to_param_map"].items():
                    kwargs["arg_to_param_map"][int(arg_index)] = param_name
            origins = transition["origin"]
            if type(origins) != list:
                origins = [origins]
            for origin in origins:
                pta.add_transition(
                    origin,
                    transition["destination"],
                    transition["name"],
                    duration=PTAAttribute.from_json_maybe(
                        transition, "duration", pta.parameters
                    ),
                    energy=PTAAttribute.from_json_maybe(
                        transition, "energy", pta.parameters
                    ),
                    timeout=PTAAttribute.from_json_maybe(
                        transition, "timeout", pta.parameters
                    ),
                    **kwargs
                )

        return pta

    @classmethod
    def from_legacy_json(cls, json_input: dict):
        """
        Return a PTA created from the provided JSON data.

        Compatible with the legacy dfatool/perl format.
        """
        kwargs = {
            "parameters": list(),
            "initial_param_values": list(),
        }

        for param in sorted(json_input["parameter"].keys()):
            kwargs["parameters"].append(param)
            kwargs["initial_param_values"].append(
                json_input["parameter"][param]["default"]
            )

        pta = cls(**kwargs)

        for name, state in json_input["state"].items():
            pta.add_state(
                name, power=PTAAttribute(value=float(state["power"]["static"]))
            )

        for trans_name in sorted(json_input["transition"].keys()):
            transition = json_input["transition"][trans_name]
            destination = transition["destination"]
            arguments = list()
            argument_values = list()
            is_interrupt = False
            if transition["level"] == "epilogue":
                is_interrupt = True
            if type(destination) == list:
                destination = destination[0]
            for arg in transition["parameters"]:
                arguments.append(arg["name"])
                argument_values.append(arg["values"])
            for origin in transition["origins"]:
                pta.add_transition(
                    origin,
                    destination,
                    trans_name,
                    arguments=arguments,
                    argument_values=argument_values,
                    is_interrupt=is_interrupt,
                )

        return pta

    @classmethod
    def from_yaml(cls, yaml_input: dict):
        """Return a PTA created from the YAML DFA format (passed as dict)."""

        kwargs = dict()

        if "parameters" in yaml_input:
            kwargs["parameters"] = yaml_input["parameters"]

        if "initial_param_values" in yaml_input:
            kwargs["initial_param_values"] = yaml_input["initial_param_values"]

        if "states" in yaml_input:
            kwargs["state_names"] = yaml_input["states"]
        # else: set to UNINITIALIZED by class constructor

        if "codegen" in yaml_input:
            kwargs["codegen"] = yaml_input["codegen"]

        if "parameter_normalization" in yaml_input:
            kwargs["parameter_normalization"] = yaml_input["parameter_normalization"]

        pta = cls(**kwargs)

        if "state" in yaml_input:
            for state_name, state in yaml_input["state"].items():
                pta.add_state(
                    state_name,
                    power=PTAAttribute.from_json_maybe(state, "power", pta.parameters),
                )

        for trans_name in sorted(yaml_input["transition"].keys()):
            kwargs = dict()
            transition = yaml_input["transition"][trans_name]
            arguments = list()
            argument_values = list()
            arg_to_param_map = dict()
            if "arguments" in transition:
                for i, argument in enumerate(transition["arguments"]):
                    arguments.append(argument["name"])
                    argument_values.append(argument["values"])
                    if "parameter" in argument:
                        arg_to_param_map[i] = argument["parameter"]
            if "argument_combination" in transition:
                kwargs["argument_combination"] = transition["argument_combination"]
            if "set_param" in transition:
                kwargs["set_param"] = transition["set_param"]
            if "is_interrupt" in transition:
                kwargs["is_interrupt"] = transition["is_interrupt"]
            if "return_value" in transition:
                kwargs["return_value_handlers"] = transition["return_value"]
            if "codegen" in transition:
                kwargs["codegen"] = transition["codegen"]
            if "loop" in transition:
                for state_name in transition["loop"]:
                    pta.add_transition(
                        state_name,
                        state_name,
                        trans_name,
                        arguments=arguments,
                        argument_values=argument_values,
                        arg_to_param_map=arg_to_param_map,
                        **kwargs
                    )
            else:
                if "src" not in transition:
                    transition["src"] = ["UNINITIALIZED"]
                if "dst" not in transition:
                    transition["dst"] = "UNINITIALIZED"
                for origin in transition["src"]:
                    pta.add_transition(
                        origin,
                        transition["dst"],
                        trans_name,
                        arguments=arguments,
                        argument_values=argument_values,
                        arg_to_param_map=arg_to_param_map,
                        **kwargs
                    )

        return pta

    def to_json(self) -> dict:
        """
        Return JSON encoding of this PTA.

        Compatible with the from_json method.
        """
        ret = {
            "parameters": self.parameters,
            "initial_param_values": self.initial_param_values,
            "state": dict(
                [[state.name, state.to_json()] for state in self.state.values()]
            ),
            "transitions": [trans.to_json() for trans in self.transitions],
            "accepting_states": self.accepting_states,
        }
        return ret

    def add_state(self, state_name: str, **kwargs):
        """
        Add a new state.

        See the State() documentation for acceptable arguments.
        """
        if (
            "power_function" in kwargs
            and type(kwargs["power_function"]) != AnalyticFunction
            and kwargs["power_function"] is not None
        ):
            kwargs["power_function"] = AnalyticFunction(
                kwargs["power_function"], self.parameters, 0
            )
        self.state[state_name] = State(state_name, **kwargs)

    def add_transition(
        self, orig_state: str, dest_state: str, function_name: str, **kwargs
    ):
        """
        Add function_name as new transition from orig_state to dest_state.

        :param orig_state: origin state name. Must be known to PTA
        :param dest_state: destination state name. Must be known to PTA.
        :param function_name: function name
        :param kwargs: see Transition() documentation
        """
        orig_state = self.state[orig_state]
        dest_state = self.state[dest_state]
        for key in ("duration_function", "energy_function", "timeout_function"):
            if (
                key in kwargs
                and kwargs[key] is not None
                and type(kwargs[key]) != AnalyticFunction
            ):
                kwargs[key] = AnalyticFunction(kwargs[key], self.parameters, 0)

        new_transition = Transition(orig_state, dest_state, function_name, **kwargs)
        self.transitions.append(new_transition)
        orig_state.add_outgoing_transition(new_transition)

    def get_transition_id(self, transition: Transition) -> int:
        """Return PTA-specific ID of transition."""
        return self.transitions.index(transition)

    def get_state_names(self):
        """Return lexically sorted list of PTA state names."""
        return sorted(self.state.keys())

    def get_state_id(self, state: State) -> int:
        """Return PTA-specific ID of state."""
        return self.get_state_names().index(state.name)

    def get_unique_transitions(self):
        """
        Return list of PTA transitions without duplicates.

        I.e., each transition name only occurs once, even if it has several entries due to
        multiple origin states and/or overloading.
        """
        seen_transitions = set()
        ret_transitions = list()
        for transition in self.transitions:
            if transition.name not in seen_transitions:
                ret_transitions.append(transition)
                seen_transitions.add(transition.name)
        return ret_transitions

    def get_unique_transition_id(self, transition: Transition) -> int:
        """
        Return PTA-specific ID of transition in unique transition list.

        The followinng condition holds:
        `
        max_index = max(map(lambda t: pta.get_unique_transition_id(t), pta.get_unique_transitions()))
        max_index == len(pta.get_unique_transitions) - 1
        `
        """
        return self.get_unique_transitions().index(transition)

    def get_initial_param_dict(self):
        return dict(
            [
                [self.parameters[i], self.initial_param_values[i]]
                for i in range(len(self.parameters))
            ]
        )

    def set_random_energy_model(self, static_model=True):
        u"""
        Set random power/energy/duration/timeout for all states and transitions.

        Values in µW/pJ/µs are chosen from a uniform [0 .. 50000] distribution.
        Only sets the static model at the moment.
        """
        for state in self.state.values():
            state.set_random_energy_model(static_model)
        for transition in self.transitions:
            transition.set_random_energy_model(static_model)

    def get_most_expensive_state(self):
        max_state = None
        for state in self.state.values():
            if state.name != "UNINITIALIZED" and (
                max_state is None or state.power.value > max_state.power.value
            ):
                max_state = state
        return max_state

    def get_least_expensive_state(self):
        min_state = None
        for state in self.state.values():
            if state.name != "UNINITIALIZED" and (
                min_state is None or state.power.value < min_state.power.value
            ):
                min_state = state
        return min_state

    def min_duration_until_energy_overflow(
        self, energy_granularity=1e-12, max_energy_value=2 ** 32 - 1
    ):
        """
        Return minimum duration (in s) until energy counter overflow during online accounting.

        :param energy_granularity: granularity of energy counter variable in J, i.e., how many Joules does an increment of one in the energy counter represent. Default: 1e-12 J = 1 pJ
        :param max_energy_value: maximum raw value in energy variable. Default: 2^32 - 1
        """

        max_power_state = self.get_most_expensive_state()
        if max_power_state.has_interrupt_transitions():
            raise RuntimeWarning(
                "state with maximum power consumption has outgoing interrupt transitions, results will be inaccurate"
            )

        # convert from µW to W
        max_power = max_power_state.power.value * 1e-6

        min_duration = max_energy_value * energy_granularity / max_power
        return min_duration

    def max_duration_until_energy_overflow(
        self, energy_granularity=1e-12, max_energy_value=2 ** 32 - 1
    ):
        """
        Return maximum duration (in s) until energy counter overflow during online accounting.

        :param energy_granularity: granularity of energy counter variable in J, i.e., how many Joules does an increment of one in the energy counter represent. Default: 1e-12 J = 1 pJ
        :param max_energy_value: maximum raw value in energy variable. Default: 2^32 - 1
        """

        min_power_state = self.get_least_expensive_state()
        if min_power_state.has_interrupt_transitions():
            raise RuntimeWarning(
                "state with maximum power consumption has outgoing interrupt transitions, results will be inaccurate"
            )

        # convert from µW to W
        min_power = min_power_state.power.value * 1e-6

        max_duration = max_energy_value * energy_granularity / min_power
        return max_duration

    def shrink_argument_values(self):
        """
        Throw away all but two values for each numeric argument of each transition.

        This is meant to speed up an initial PTA-based benchmark by
        reducing the parameter space while still gaining insights in the
        effect (or lack thereof) or individual parameters on hardware behaviour.

        Parameters with non-numeric values (anything containing neither
        numbers nor enums) are left as-is, as they may be distinct
        toggles whose effect cannot be estimated when they are left out.
        """
        for transition in self.transitions:
            for i, argument in enumerate(transition.arguments):
                if len(transition.argument_values[i]) <= 2:
                    continue
                if transition.argument_combination == "zip":
                    continue
                values_are_numeric = True
                for value in transition.argument_values[i]:
                    if not is_numeric(
                        self.normalize_parameter(transition.arg_to_param_map[i], value)
                    ):
                        values_are_numeric = False
                if values_are_numeric and len(transition.argument_values[i]) > 2:
                    transition.argument_values[i] = [
                        transition.argument_values[i][0],
                        transition.argument_values[i][-1],
                    ]

    def _dfs_with_param(self, generator, param_dict):
        for trace in generator:
            param = param_dict.copy()
            ret = list()
            for elem in trace:
                transition, arguments = elem
                if transition is not None:
                    param = transition.get_params_after_transition(param, arguments)
                    ret.append(
                        (transition, arguments, self.normalize_parameters(param))
                    )
                else:
                    # parameters have already been normalized
                    ret.append((transition, arguments, param))
            yield ret

    def bfs(
        self,
        depth: int = 10,
        orig_state: str = "UNINITIALIZED",
        param_dict: dict = None,
        with_parameters: bool = False,
        transition_filter=None,
        state_filter=None,
    ):
        """
        Return a generator object for breadth-first search of traces starting at orig_state.

        Each trace consists of a list of
        tuples describing the corresponding transition and (if enabled)
        arguments and parameters. When both with_arguments and with_parameters
        are True, each transition is a (Transition object, argument list, parameter dict) tuple.

        Note that the parameter dict refers to parameter values _after_
        passing the corresponding transition. Although this may seem odd at
        first, it is useful when analyzing measurements: Properties of
        the state following this transition may be affected by the parameters
        set by the transition, so it is useful to have those readily available.

        A trace is (at the moment) a list of alternating states and transition, both starting and ending with a state.
        Does not yield the no-operation trace consisting only of `orig_state`. If `orig_state` has no outgoing transitions, the output is empty.

        :param orig_state: initial state for breadth-first search
        :param depth: search depth, default 10
        :param param_dict: initial parameter values
        :param with_arguments: perform dfs with argument values
        :param with_parameters: include parameters in trace?
        :param transition_filter: If set, only follow a transition if transition_filter(transition object) returns true. Default None.
        :param state_iflter: If set, only follow a state if state_filter(state_object) returns true. Default None.
        """
        state_queue = queue.Queue()
        state_queue.put((list(), self.state[orig_state]))

        while not state_queue.empty():
            trace, state = state_queue.get()
            if len(trace) > depth:
                return
            if state_filter is None or state_filter(state):
                for transition in state.outgoing_transitions.values():
                    if transition_filter is None or transition_filter(transition):
                        new_trace = trace.copy()
                        new_trace.append((transition,))
                        yield new_trace
                        state_queue.put((new_trace, transition.destination))

    def dfs(
        self,
        depth: int = 10,
        orig_state: str = "UNINITIALIZED",
        param_dict: dict = None,
        with_parameters: bool = False,
        **kwargs
    ):
        """
        Return a generator object for depth-first search starting at orig_state.

        :param depth: search depth, default 10
        :param orig_state: initial state for depth-first search
        :param param_dict: initial parameter values
        :param with_arguments: perform dfs with argument values
        :param with_parameters: include parameters in trace?
        :param trace_filter: list of lists. Each sub-list is a trace. Only traces matching one of the provided sub-lists are returned.
        :param sleep: sleep duration between states in us. If None or 0, no sleep pseudo-transitions will be included in the trace.

        Each trace consists of a list of
        tuples describing the corresponding transition and (if enabled)
        arguments and parameters. When both with_arguments and with_parameters
        are True, each transition is a (Transition object, argument list, parameter dict) tuple.

        Note that the parameter dict refers to parameter values _after_
        passing the corresponding transition. Although this may seem odd at
        first, it is useful when analyzing measurements: Properties of
        the state following this transition may be affected by the parameters
        set by the transition, so it is useful to have those readily available.
        """
        if with_parameters and not param_dict:
            param_dict = self.get_initial_param_dict()

        if with_parameters and "with_arguments" not in kwargs:
            raise ValueError("with_parameters = True requires with_arguments = True")

        if self.accepting_states:
            generator = filter(
                lambda x: x[-1][0].destination.name in self.accepting_states,
                self.state[orig_state].dfs(depth, **kwargs),
            )
        else:
            generator = self.state[orig_state].dfs(depth, **kwargs)

        if with_parameters:
            return self._dfs_with_param(generator, param_dict)
        else:
            return generator

    def simulate(
        self,
        trace: list,
        orig_state: str = "UNINITIALIZED",
        orig_param=None,
        accounting=None,
    ):
        u"""
        Simulate a single run through the PTA and return total energy, duration, final state, and resulting parameters.

        :param trace: list of (function name, arg1, arg2, ...) tuples representing the individual transitions,
            or list of (Transition, argument tuple, parameter) tuples originating from dfs.
            The tuple (None, duration) represents a sleep time between states in us
        :param orig_state: origin state, default UNINITIALIZED
        :param orig_param: initial parameters, default: `self.initial_param_values`
        :param accounting: EnergyAccounting object, default empty

        :returns: SimulationResult with duration in s, total energy in J, end state, and final parameters
        """
        total_duration = 0.0
        total_duration_mae = 0.0
        total_energy = 0.0
        total_energy_error = 0.0
        if type(orig_state) is State:
            state = orig_state
        else:
            state = self.state[orig_state]
        if orig_param:
            param_dict = orig_param.copy()
        else:
            param_dict = dict(
                [
                    [self.parameters[i], self.initial_param_values[i]]
                    for i in range(len(self.parameters))
                ]
            )
        for function in trace:
            if isinstance(function[0], Transition):
                function_name = function[0].name
                function_args = function[1]
            else:
                function_name = function[0]
                function_args = function[1:]
            if function_name is None or function_name == "_":
                duration = function_args[0]
                total_energy += state.get_energy(duration, param_dict)
                if state.power.value_error is not None:
                    total_energy_error += (
                        duration * state.power.eval_mae(param_dict, function_args)
                    ) ** 2
                total_duration += duration
                # assumption: sleep is near-exact and does not contribute to the duration error
                if accounting is not None:
                    accounting.sleep(duration)
            else:
                transition = state.get_transition(function_name)
                total_duration += transition.duration.eval(param_dict, function_args)
                if transition.duration.value_error is not None:
                    total_duration_mae += (
                        transition.duration.eval_mae(param_dict, function_args) ** 2
                    )
                total_energy += transition.get_energy(param_dict, function_args)
                if transition.energy.value_error is not None:
                    total_energy_error += (
                        transition.energy.eval_mae(param_dict, function_args) ** 2
                    )
                param_dict = transition.get_params_after_transition(
                    param_dict, function_args
                )
                state = transition.destination
                if accounting is not None:
                    accounting.pass_transition(transition)
                while state.has_interrupt_transitions():
                    transition = state.get_next_interrupt(param_dict)
                    duration = transition.get_timeout(param_dict)
                    total_duration += duration
                    total_energy += state.get_energy(duration, param_dict)
                    if accounting is not None:
                        accounting.sleep(duration)
                        accounting.pass_transition(transition)
                    param_dict = transition.get_params_after_transition(param_dict)
                    state = transition.destination

        return SimulationResult(
            total_duration,
            total_energy,
            state,
            param_dict,
            duration_mae=np.sqrt(total_duration_mae),
            energy_mae=np.sqrt(total_energy_error),
        )

    def update(self, static_model, param_model, static_error=None, analytic_error=None):
        for state in self.state.values():
            if state.name != "UNINITIALIZED":
                try:
                    state.power.value = static_model(state.name, "power")
                    if static_error is not None:
                        state.power.value_error = static_error[state.name]["power"]
                    if param_model(state.name, "power"):
                        state.power.function = param_model(state.name, "power")[
                            "function"
                        ]
                        if analytic_error is not None:
                            state.power.function_error = analytic_error[state.name][
                                "power"
                            ]
                except KeyError:
                    print(
                        "[W] skipping model update of state {} due to missing data".format(
                            state.name
                        )
                    )
                    pass
        for transition in self.transitions:
            try:
                transition.duration.value = static_model(transition.name, "duration")
                if param_model(transition.name, "duration"):
                    transition.duration.function = param_model(
                        transition.name, "duration"
                    )["function"]
                    if analytic_error is not None:
                        transition.duration.function_error = analytic_error[
                            transition.name
                        ]["duration"]
                transition.energy.value = static_model(transition.name, "energy")
                if param_model(transition.name, "energy"):
                    transition.energy.function = param_model(transition.name, "energy")[
                        "function"
                    ]
                    if analytic_error is not None:
                        transition.energy.function_error = analytic_error[
                            transition.name
                        ]["energy"]
                if transition.is_interrupt:
                    transition.timeout.value = static_model(transition.name, "timeout")
                    if param_model(transition.name, "timeout"):
                        transition.timeout.function = param_model(
                            transition.name, "timeout"
                        )["function"]
                        if analytic_error is not None:
                            transition.timeout.function_error = analytic_error[
                                transition.name
                            ]["timeout"]

                if static_error is not None:
                    transition.duration.value_error = static_error[transition.name][
                        "duration"
                    ]
                    transition.energy.value_error = static_error[transition.name][
                        "energy"
                    ]
                    transition.timeout.value_error = static_error[transition.name][
                        "timeout"
                    ]
            except KeyError:
                print(
                    "[W] skipping model update of transition {} due to missing data".format(
                        transition.name
                    )
                )
                pass
