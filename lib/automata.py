"""Classes and helper functions for PTA and other automata."""

from functions import AnalyticFunction, NormalizationFunction
from utils import is_numeric
import itertools
import numpy as np

def _dict_to_list(input_dict: dict) -> list:
    return [input_dict[x] for x in sorted(input_dict.keys())]

def _attribute_to_json(static_value: float, param_function: AnalyticFunction) -> dict:
    ret = {
        'static' : static_value
    }
    if param_function:
        ret['function'] = {
            'raw' : param_function._model_str,
            'regression_args' : list(param_function._regression_args)
        }
    return ret

class State:
    """A single PTA state."""

    def __init__(self, name: str, power: float = 0,
            power_function: AnalyticFunction = None):
        u"""
        Create a new PTA state.

        :param name: state name
        :param power: static state power in µW
        :param power_function: parameterized state power in µW
        """
        self.name = name
        self.power = power
        self.power_function = power_function
        self.outgoing_transitions = {}

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
        if self.power_function:
            return self.power_function.eval(_dict_to_list(param_dict)) * duration
        return self.power * duration

    def set_random_energy_model(self, static_model = True):
        """Set a random static energy value."""
        self.power = int(np.random.sample() * 50000)

    def get_transition(self, transition_name: str) -> object:
        """
        Return Transition object for outgoing transtion transition_name.

        :param transition_name: transition name
        :returns: `Transition` object
        """
        return self.outgoing_transitions[transition_name]

    def has_interrupt_transitions(self) -> bool:
        """Check whether this state has any outgoing interrupt transitions."""
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
        interrupts = filter(lambda x: x.is_interrupt, self.outgoing_transitions.values())
        interrupts = sorted(interrupts, key = lambda x: x.get_timeout(parameters))
        return interrupts[0]

    def dfs(self, depth: int, with_arguments: bool = False, trace_filter = None, sleep: int = 0):
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

        # A '$' entry in trace_filter indicates that the trace should (successfully) terminate here regardless of `depth`.
        if trace_filter is not None and next(filter(lambda x: x == '$', map(lambda x: x[0], trace_filter)), None) is not None:
            yield []
        # there may be other entries in trace_filter that still yield results.
        if depth == 0:
            for trans in self.outgoing_transitions.values():
                if trace_filter is not None and len(list(filter(lambda x: x == trans.name, map(lambda x: x[0], trace_filter)))) == 0:
                    continue
                if with_arguments:
                    if trans.argument_combination == 'cartesian':
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
                if trace_filter is not None and next(filter(lambda x: x == trans.name, map(lambda x: x[0], trace_filter)), None) is None:
                    continue
                if trace_filter is not None:
                    new_trace_filter = map(lambda x: x[1:], filter(lambda x: x[0] == trans.name, trace_filter))
                    new_trace_filter = list(filter(len, new_trace_filter))
                    if len(new_trace_filter) == 0:
                        new_trace_filter = None
                else:
                    new_trace_filter = None
                for suffix in trans.destination.dfs(depth - 1, with_arguments = with_arguments, trace_filter = new_trace_filter, sleep = sleep):
                    if with_arguments:
                        if trans.argument_combination == 'cartesian':
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
        ret = {
            'name' : self.name,
            'power' : _attribute_to_json(self.power, self.power_function)
        }
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

    def __init__(self, orig_state: State, dest_state: State, name: str,
            energy: float = 0, energy_function: AnalyticFunction = None,
            duration: float = 0, duration_function: AnalyticFunction = None,
            timeout: float = 0, timeout_function: AnalyticFunction = None,
            is_interrupt: bool = False,
            arguments: list = [],
            argument_values: list = [],
            argument_combination: str = 'cartesian', # or 'zip'
            param_update_function = None,
            arg_to_param_map: dict = None,
            set_param = None,
            return_value_handlers: list = [],
            codegen = dict()):
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
        self.energy_function = energy_function
        self.duration = duration
        self.duration_function = duration_function
        self.timeout = timeout
        self.timeout_function = timeout_function
        self.is_interrupt = is_interrupt
        self.arguments = arguments.copy()
        self.argument_values = argument_values.copy()
        self.argument_combination = argument_combination
        self.param_update_function = param_update_function
        self.arg_to_param_map = arg_to_param_map
        self.set_param = set_param
        self.return_value_handlers = return_value_handlers
        self.codegen = codegen

        for handler in self.return_value_handlers:
            if 'formula' in handler:
                handler['formula'] = NormalizationFunction(handler['formula'])


    def get_duration(self, param_dict: dict = {}, args: list = []) -> float:
        u"""
        Return transition duration in µs.

        :param param_dict: current parameter values
        :param args: function arguments

        :returns: transition duration in µs
        """
        if self.duration_function:
            return self.duration_function.eval(_dict_to_list(param_dict), args)
        return self.duration

    def get_energy(self, param_dict: dict = {}, args: list = []) -> float:
        u"""
        Return transition energy cost in pJ.

        arguments:
        param_dict -- current parameter values
        args -- function arguments
        """
        if self.energy_function:
            return self.energy_function.eval(_dict_to_list(param_dict), args)
        return self.energy

    def set_random_energy_model(self, static_model = True):
        self.energy = int(np.random.sample() * 50000)

    def get_timeout(self, param_dict: dict = {}) -> float:
        u"""
        Return transition timeout in µs.

        Returns 0 if the transition does not have a timeout.

        arguments:
        param_dict -- current parameter values
        args -- function arguments
        """
        if self.timeout_function:
            return self.timeout_function.eval(_dict_to_list(param_dict))
        return self.timeout

    def get_params_after_transition(self, param_dict: dict, args: list = []) -> dict:
        """
        Return the new parameter dict after taking this transition.

        parameter values may be affected by this transition's update function,
        it's argument-to-param map, and its set_param settings.
        """
        if self.param_update_function:
            return self.param_update_function(param_dict, args)
        ret = param_dict.copy()
        if self.arg_to_param_map:
            for k, v in self.arg_to_param_map.items():
                ret[v] = args[k]
        if self.set_param:
            for k, v in self.set_param.items():
                ret[k] = v
        return ret

    def to_json(self) -> dict:
        """Return JSON encoding of this transition object."""
        ret = {
            'name' : self.name,
            'origin' : self.origin.name,
            'destination' : self.destination.name,
            'is_interrupt' : self.is_interrupt,
            'arguments' : self.arguments,
            'argument_values' : self.argument_values,
            'argument_combination' : self.argument_combination,
            'arg_to_param_map' : self.arg_to_param_map,
            'set_param' : self.set_param,
            'duration' : _attribute_to_json(self.duration, self.duration_function),
            'energy' : _attribute_to_json(self.energy, self.energy_function),
            'timeout' : _attribute_to_json(self.timeout, self.timeout_function),
        }
        return ret

def _json_function_to_analytic_function(base, attribute: str, parameters: list):
    if attribute in base and 'function' in base[attribute]:
        base = base[attribute]['function']
        return AnalyticFunction(base['raw'], parameters, 0, regression_args = base['regression_args'])
    return None

def _json_get_static(base, attribute: str):
    if attribute in base:
        return base[attribute]['static']
    return 0

class PTA:
    """
    A parameterized priced timed automaton. All states are accepting.

    Suitable for simulation, model storage, and (soon) benchmark generation.

    :param state: dict mapping state name to `State` object
    :param accepting_states: list of accepting state names
    :param parameters: current parameters
    :param parameter_normalization: TODO
    :param codegen: TODO
    :param initial_param_values: TODO
    :param transitions: list of `Transition` objects
    """

    def __init__(self, state_names: list = [],
            accepting_states: list = None,
            parameters: list = [], initial_param_values: list = None,
            codegen: dict = {}, parameter_normalization: dict = None):
        """
        Return a new PTA object.

        arguments:
        state_names -- names of PTA states. Note that the PTA always contains
            an initial UNINITIALIZED state, regardless of the content of state_names.
        accepting_states -- names of accepting states. By default, all states are accepting
        parameters -- names of PTA parameters
        initial_param_values -- initial value for each parameter
        instance -- class used for generated C++ code
        header -- header include path for C++ class definition
        parameter_normalization -- dict mapping driver API parameter values to hardware values, e.g. a bitrate register value to an actual bitrate in kbit/s.
            Each parameter key has in turn a dict value. Supported entries:
            `enum`: Mapping of enum descriptors (keys) to parameter values. Note that the mapping is not required to correspond to the driver API.
        """
        self.state = dict([[state_name, State(state_name)] for state_name in state_names])
        self.accepting_states = accepting_states.copy() if accepting_states else None
        self.parameters = parameters.copy()
        self.parameter_normalization = parameter_normalization
        self.codegen = codegen
        if initial_param_values:
            self.initial_param_values = initial_param_values.copy()
        else:
            self.initial_param_values = [None for x in self.parameters]
        self.transitions = []

        if not 'UNINITIALIZED' in state_names:
            self.state['UNINITIALIZED'] = State('UNINITIALIZED')

        if self.parameter_normalization:
            for normalization_spec in self.parameter_normalization.values():
                if 'formula' in normalization_spec:
                    normalization_spec['formula'] = NormalizationFunction(normalization_spec['formula'])

    def normalize_parameter(self, parameter_name, parameter_value):
        if parameter_value is not None and self.parameter_normalization is not None and parameter_name in self.parameter_normalization:
            if 'enum' in self.parameter_normalization[parameter_name] and parameter_value in self.parameter_normalization[parameter_name]['enum']:
                return self.parameter_normalization[parameter_name]['enum'][parameter_value]
            if 'formula' in self.parameter_normalization[parameter_name]:
                normalization_formula = self.parameter_normalization[parameter_name]['formula']
                return normalization_formula.eval(parameter_value)
        return parameter_value

    def normalize_parameters(self, param_dict):
        if self.parameter_normalization is None:
            return param_dict.copy()
        normalized_param = param_dict.copy()
        for parameter, value in param_dict.items():
            normalized_param[parameter] = self.normalize_parameter(parameter, value)
        return normalized_param

    @classmethod
    def from_json(cls, json_input: dict):
        """
        Return a PTA created from the provided JSON data.

        Compatible with the to_json method.
        """
        if 'transition' in json_input:
            return cls.from_legacy_json(json_input)

        kwargs = dict()
        for key in ('state_names', 'parameters', 'initial_param_values', 'accepting_states'):
            if key in json_input:
                kwargs[key] = json_input[key]
        pta = cls(**kwargs)
        for name, state in json_input['state'].items():
            power_function = _json_function_to_analytic_function(state, 'power', pta.parameters)
            pta.add_state(name, power = _json_get_static(state, 'power'), power_function = power_function)
        for transition in json_input['transitions']:
            duration_function = _json_function_to_analytic_function(transition, 'duration', pta.parameters)
            energy_function = _json_function_to_analytic_function(transition, 'energy', pta.parameters)
            timeout_function = _json_function_to_analytic_function(transition, 'timeout', pta.parameters)
            kwargs = dict()
            for key in ['arg_to_param_map', 'argument_values', 'argument_combination']:
                if key in transition:
                    kwargs[key] = transition[key]
            origins = transition['origin']
            if type(origins) != list:
                origins = [origins]
            for origin in origins:
                pta.add_transition(origin, transition['destination'],
                    transition['name'],
                    duration = _json_get_static(transition, 'duration'),
                    duration_function = duration_function,
                    energy = _json_get_static(transition, 'energy'),
                    energy_function = energy_function,
                    timeout = _json_get_static(transition, 'timeout'),
                    timeout_function = timeout_function,
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
            'parameters' : list(),
            'initial_param_values': list(),
        }

        for param in sorted(json_input['parameter'].keys()):
            kwargs['parameters'].append(param)
            kwargs['initial_param_values'].append(json_input['parameter'][param]['default'])

        pta = cls(**kwargs)

        for name, state in json_input['state'].items():
            pta.add_state(name, power = float(state['power']['static']))

        for trans_name in sorted(json_input['transition'].keys()):
            transition = json_input['transition'][trans_name]
            destination = transition['destination']
            arguments = list()
            argument_values = list()
            is_interrupt = False
            if transition['level'] == 'epilogue':
                is_interrupt = True
            if type(destination) == list:
                destination = destination[0]
            for arg in transition['parameters']:
                arguments.append(arg['name'])
                argument_values.append(arg['values'])
            for origin in transition['origins']:
                pta.add_transition(origin, destination, trans_name,
                    arguments = arguments, argument_values = argument_values,
                    is_interrupt = is_interrupt)

        return pta

    @classmethod
    def from_yaml(cls, yaml_input: dict):
        """Return a PTA created from the YAML DFA format (passed as dict)."""

        kwargs = dict()

        if 'parameters' in yaml_input:
            kwargs['parameters'] = yaml_input['parameters']

        if 'initial_param_values' in yaml_input:
            kwargs['initial_param_values'] = yaml_input['initial_param_values']

        if 'states' in yaml_input:
            kwargs['state_names'] = yaml_input['states']
        # else: set to UNINITIALIZED by class constructor

        if 'codegen' in yaml_input:
            kwargs['codegen'] = yaml_input['codegen']

        if 'parameter_normalization' in yaml_input:
            kwargs['parameter_normalization'] = yaml_input['parameter_normalization']

        pta = cls(**kwargs)

        for trans_name in sorted(yaml_input['transition'].keys()):
            kwargs = dict()
            transition = yaml_input['transition'][trans_name]
            arguments = list()
            argument_values = list()
            arg_to_param_map = dict()
            if 'arguments' in transition:
                for i, argument in enumerate(transition['arguments']):
                    arguments.append(argument['name'])
                    argument_values.append(argument['values'])
                    if 'parameter' in argument:
                        arg_to_param_map[i] = argument['parameter']
            if 'argument_combination' in transition:
                kwargs['argument_combination'] = transition['argument_combination']
            if 'set_param' in transition:
                kwargs['set_param'] = transition['set_param']
            if 'is_interrupt' in transition:
                kwargs['is_interrupt'] = transition['is_interrupt']
            if 'return_value' in transition:
                kwargs['return_value_handlers'] = transition['return_value']
            if 'loop' in transition:
                for state_name in transition['loop']:
                    pta.add_transition(state_name, state_name, trans_name,
                        arguments = arguments, argument_values = argument_values,
                        arg_to_param_map = arg_to_param_map,
                        **kwargs)
            else:
                if not 'src' in transition:
                    transition['src'] = ['UNINITIALIZED']
                if not 'dst' in transition:
                    transition['dst'] = 'UNINITIALIZED'
                for origin in transition['src']:
                    pta.add_transition(origin, transition['dst'], trans_name,
                        arguments = arguments, argument_values = argument_values,
                        arg_to_param_map = arg_to_param_map,
                        **kwargs)

        return pta

    def to_json(self) -> dict:
        """
        Return JSON encoding of this PTA.

        Compatible with the from_json method.
        """
        ret = {
            'parameters' : self.parameters,
            'initial_param_values' : self.initial_param_values,
            'state' : dict([[state.name, state.to_json()] for state in self.state.values()]),
            'transitions' : [trans.to_json() for trans in self.transitions],
            'accepting_states' : self.accepting_states,
        }
        return ret

    def add_state(self, state_name: str, **kwargs):
        """
        Add a new state.

        See the State() documentation for acceptable arguments.
        """
        if 'power_function' in kwargs and type(kwargs['power_function']) != AnalyticFunction and kwargs['power_function'] != None:
            kwargs['power_function'] = AnalyticFunction(kwargs['power_function'],
                self.parameters, 0)
        self.state[state_name] = State(state_name, **kwargs)

    def add_transition(self, orig_state: str, dest_state: str, function_name: str, **kwargs):
        """
        Add function_name as new transition from orig_state to dest_state.

        arguments:
        orig_state -- origin state name. Must be known to PTA
        dest_state -- destination state name. Must be known to PTA.
        function_name -- function name
        kwargs -- see Transition() documentation
        """
        orig_state = self.state[orig_state]
        dest_state = self.state[dest_state]
        for key in ('duration_function', 'energy_function', 'timeout_function'):
            if key in kwargs and kwargs[key] != None and type(kwargs[key]) != AnalyticFunction:
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
        return dict([[self.parameters[i], self.initial_param_values[i]] for i in range(len(self.parameters))])

    def set_random_energy_model(self, static_model = True):
        for state in self.state.values():
            state.set_random_energy_model(static_model)
        for transition in self.transitions:
            transition.set_random_energy_model(static_model)

    def shrink_argument_values(self):
        """
        Throw away all but two values for each numeric argument of each transition.

        This is meant to speed up an initial PTA-based benchmark by
        reducing the parameter space while still gaining insights in the
        effect (or nop) or individual parameters on hardware behaviour.

        Parameters with non-numeric values (anything containing neither
        numbers nor enums) are left as-is, as they may be distinct
        toggles whose effect cannot be estimated when they are left out.
        """
        for transition in self.transitions:
            for i, argument in enumerate(transition.arguments):
                if len(transition.argument_values[i]) <= 2:
                    continue
                if transition.argument_combination == 'zip':
                    continue
                values_are_numeric = True
                for value in transition.argument_values[i]:
                    if not is_numeric(self.normalize_parameter(transition.arg_to_param_map[i], value)):
                        values_are_numeric = False
                if values_are_numeric and len(transition.argument_values[i]) > 2:
                    transition.argument_values[i] = [transition.argument_values[i][0], transition.argument_values[i][-1]]

    def _dfs_with_param(self, generator, param_dict):
        for trace in generator:
            param = param_dict.copy()
            ret = list()
            for elem in trace:
                transition, arguments = elem
                if transition is not None:
                    param = transition.get_params_after_transition(param, arguments)
                    ret.append((transition, arguments, self.normalize_parameters(param)))
                else:
                    # parameters have already been normalized
                    ret.append((transition, arguments, param))
            yield ret

    def dfs(self, depth: int = 10, orig_state: str = 'UNINITIALIZED', param_dict: dict = None, with_parameters: bool = False, **kwargs):
        """
        Return a generator object for depth-first search starting at orig_state.

        :param depth: search depth
        :param orig_state: initial state for depth-first search
        :param param_dict: initial parameter values
        :param with_arguments: perform dfs with argument values
        :param with_parameters: include parameters in trace?
        :param trace_filter: list of lists. Each sub-list is a trace. Only traces matching one of the provided sub-lists are returned.
        :param sleep: sleep duration between states in us. If None or 0, no sleep pseudo-transitions will be included in the trace.

        The returned generator emits traces. Each trace consts of a list of
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

        if with_parameters and not 'with_arguments' in kwargs:
            raise ValueError("with_parameters = True requires with_arguments = True")

        if self.accepting_states:
            generator = filter(lambda x: x[-1][0].destination.name in self.accepting_states,
                self.state[orig_state].dfs(depth, **kwargs))
        else:
            generator = self.state[orig_state].dfs(depth, **kwargs)

        if with_parameters:
            return self._dfs_with_param(generator, param_dict)
        else:
            return generator

    def simulate(self, trace: list, orig_state: str = 'UNINITIALIZED', accounting = None):
        u"""
        Simulate a single run through the PTA and return total energy, duration, final state, and resulting parameters.

        :param trace: list of (function name, arg1, arg2, ...) tuples representing the individual transitions,
            or list of (Transition, argument tuple, parameter) tuples originating from dfs.
            The tuple (None, duration) represents a sleep time between states in us
        :param orig_state: origin state, default UNINITIALIZED

        :returns (total energy in pJ, total duration in µs, end state, end parameters)
        """
        total_duration = 0.
        total_energy = 0.
        state = self.state[orig_state]
        param_dict = dict([[self.parameters[i], self.initial_param_values[i]] for i in range(len(self.parameters))])
        for function in trace:
            if isinstance(function[0], Transition):
                function_name = function[0].name
                function_args = function[1]
            else:
                function_name = function[0]
                function_args = function[1 : ]
            if function_name is None:
                duration = function_args[0]
                total_energy += state.get_energy(duration, param_dict)
                total_duration += duration
                if accounting is not None:
                    accounting.sleep(duration)
            else:
                transition = state.get_transition(function_name)
                total_duration += transition.get_duration(param_dict, function_args)
                total_energy += transition.get_energy(param_dict, function_args)
                param_dict = transition.get_params_after_transition(param_dict, function_args)
                state = transition.destination
                if accounting is not None:
                    accounting.pass_transition(transition)
                while (state.has_interrupt_transitions()):
                    transition = state.get_next_interrupt(param_dict)
                    duration = transition.get_timeout(param_dict)
                    total_duration += duration
                    total_energy += state.get_energy(duration, param_dict)
                    if accounting is not None:
                        accounting.sleep(duration)
                        accounting.pass_transition(transition)
                    param_dict = transition.get_params_after_transition(param_dict)
                    state = transition.destination

        return total_energy, total_duration, state, param_dict

    def update(self, static_model, param_model):
        for state in self.state.values():
            if state.name != 'UNINITIALIZED':
                state.power = static_model(state.name, 'power')
                if param_model(state.name, 'power'):
                    state.power_function = param_model(state.name, 'power')['function']
        for transition in self.transitions:
            transition.duration = static_model(transition.name, 'duration')
            if param_model(transition.name, 'duration'):
                transition.duration_function = param_model(transition.name, 'duration')['function']
            transition.energy = static_model(transition.name, 'energy')
            if param_model(transition.name, 'energy'):
                transition.energy_function = param_model(transition.name, 'energy')['function']
            if transition.is_interrupt:
                transition.timeout = static_model(transition.name, 'timeout')
                if param_model(transition.name, 'timeout'):
                    transition.timeout_function = param_model(transition.name, 'timeout')['function']
