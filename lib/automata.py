from functions import AnalyticFunction

def _parse_function(input_function):
    if type('input_function') == 'str':
        raise NotImplemented
    if type('input_function') == 'function':
        return 'raise ValueError', input_function
    raise ValueError('Function description must be provided as string or function')

def _dict_to_list(input_dict):
    return [input_dict[x] for x in sorted(input_dict.keys())]

def _attribute_to_json(static_value, param_function):
    ret = {
        'static' : static_value
    }
    if param_function:
        ret['function'] = {
            'raw' : param_function._model_str,
            'regression_args' : list(param_function._regression_args)
        }
    return ret

class Transition:
    def __init__(self, orig_state, dest_state, name,
            energy = 0, energy_function = None,
            duration = 0, duration_function = None,
            timeout = 0, timeout_function = None,
            is_interrupt = False,
            arguments = [], param_update_function = None,
            arg_to_param_map = None, set_param = None):
        self.name = name
        self.origin = orig_state.name
        self.destination = dest_state.name
        self.energy = energy
        self.energy_function = energy_function
        self.duration = duration
        self.duration_function = duration_function
        self.timeout = timeout
        self.timeout_function = timeout_function
        self.is_interrupt = is_interrupt
        self.arguments = arguments.copy()
        self.param_update_function = param_update_function
        self.arg_to_param_map = arg_to_param_map
        self.set_param = set_param

    def get_duration(self, param_dict = {}, args = []):
        if self.duration_function:
            return self.duration_function.eval(_dict_to_list(param_dict), args)
        return self.duration

    def get_energy(self, param_dict = {}, args = []):
        if self.energy_function:
            return self.energy_function.eval(_dict_to_list(param_dict), args)
        return self.energy

    def get_timeout(self, param_dict = {}):
        if self.timeout_function:
            return self.timeout_function.eval(_dict_to_list(param_dict))
        return self.timeout

    def get_params_after_transition(self, param_dict, args = []):
        if self.param_update_function:
            return self.param_update_function(param_dict, args)
        ret = param_dict.copy()
        if self.arg_to_param_map:
            for k, v in self.arg_to_param_map.items():
                ret[k] = args[v]
        if self.set_param:
            for k, v in self.set_param.items():
                ret[k] = v
        return ret

    def to_json(self):
        ret = {
            'name' : self.name,
            'origin' : self.origin,
            'destination' : self.destination,
            'is_interrupt' : self.is_interrupt,
            'arguments' : self.arguments,
            'arg_to_param_map' : self.arg_to_param_map,
            'set_param' : self.set_param,
            'duration' : _attribute_to_json(self.duration, self.duration_function),
            'energy' : _attribute_to_json(self.energy, self.energy_function),
            'timeout' : _attribute_to_json(self.timeout, self.timeout_function),
        }
        return ret

class State:
    def __init__(self, name, power = 0, power_function = None):
        self.name = name
        self.power = power
        self.power_function = power_function
        self.outgoing_transitions = {}

    def add_outgoing_transition(self, new_transition):
        self.outgoing_transitions[new_transition.name] = new_transition

    def get_energy(self, duration, param_dict = {}):
        if self.power_function:
            return self.power_function.eval(_dict_to_list(param_dict)) * duration
        return self.power * duration

    def get_transition(self, transition_name):
        return self.outgoing_transitions[transition_name]

    def has_interrupt_transitions(self):
        for trans in self.outgoing_transitions.values():
            if trans.is_interrupt:
                return True
        return False

    def get_next_interrupt(self, parameters):
        interrupts = filter(lambda x: x.is_interrupt, self.outgoing_transitions.values())
        interrupts = sorted(interrupts, key = lambda x: x.get_timeout(parameters))
        return interrupts[0]

    def dfs(self, depth):
        if depth == 0:
            for trans in self.outgoing_transitions.values():
                yield [trans.name]
        else:
            for trans in self.outgoing_transitions.values():
                for suffix in trans.destination.dfs(depth - 1):
                    new_suffix = [trans.name]
                    new_suffix.extend(suffix)
                    yield new_suffix

    def to_json(self):
        ret = {
            'name' : self.name,
            'power' : _attribute_to_json(self.power, self.power_function)
        }
        return ret

def _json_function_to_analytic_function(base, attribute, parameters):
    if attribute in base and 'function' in base[attribute]:
        base = base[attribute]['function']
        return AnalyticFunction(base['raw'], parameters, 0, regression_args = base['regression_args'])
    return None

def _json_get_static(base, attribute):
    if attribute in base:
        return base[attribute]['static']
    return 0

class PTA:
    def __init__(self, state_names = [], parameters = [], initial_param_values = None):
        self.states = dict([[state_name, State(state_name)] for state_name in state_names])
        self.parameters = parameters.copy()
        if initial_param_values:
            self.initial_param_values = initial_param_values.copy()
        else:
            self.initial_param_values = [None for x in self.parameters]
        self.transitions = []

        if not 'UNINITIALIZED' in state_names:
            self.states['UNINITIALIZED'] = State('UNINITIALIZED')

    @classmethod
    def from_json(cls, json_input):
        kwargs = {}
        for key in ('state_names', 'parameters', 'initial_param_values'):
            if key in json_input:
                kwargs[key] = json_input[key]
        pta = cls(**kwargs)
        for name, state in json_input['states'].items():
            power_function = _json_function_to_analytic_function(state, 'power', pta.parameters)
            pta.add_state(name, power = _json_get_static(state, 'power'), power_function = power_function)
        for transition in json_input['transitions']:
            duration_function = _json_function_to_analytic_function(transition, 'duration', pta.parameters)
            energy_function = _json_function_to_analytic_function(transition, 'energy', pta.parameters)
            timeout_function = _json_function_to_analytic_function(transition, 'timeout', pta.parameters)
            arg_to_param_map = None
            if 'arg_to_param_map' in transition:
                arg_to_param_map = transition['arg_to_param_map']
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
                    arg_to_param_map = arg_to_param_map
                )

        return pta

    def to_json(self):
        ret = {
            'parameters' : self.parameters,
            'initial_param_values' : self.initial_param_values,
            'states' : dict([[state.name, state.to_json()] for state in self.states.values()]),
            'transitions' : [trans.to_json() for trans in self.transitions]
        }
        return ret

    def add_state(self, state_name, **kwargs):
        if 'power_function' in kwargs and type(kwargs['power_function']) != AnalyticFunction:
            kwargs['power_function'] = AnalyticFunction(kwargs['power_function'],
                self.parameters, 0)
        self.states[state_name] = State(state_name, **kwargs)

    def add_transition(self, orig_state, dest_state, function_name, **kwargs):
        orig_state = self.states[orig_state]
        dest_state = self.states[dest_state]
        for key in ('duration_function', 'energy_function', 'timeout_function'):
            if key in kwargs and type(kwargs[key]) != AnalyticFunction:
                kwargs[key] = AnalyticFunction(kwargs[key], self.parameters, 0)

        new_transition = Transition(orig_state, dest_state, function_name, **kwargs)
        self.transitions.append(new_transition)
        orig_state.add_outgoing_transition(new_transition)

    def dfs(self, depth = 10, orig_state = 'UNINITIALIZED'):
        return self.states[orig_state].dfs(depth)

    def simulate(self, trace, orig_state = 'UNINITIALIZED'):
        total_duration = 0.
        total_energy = 0.
        state = self.states[orig_state]
        param_dict = dict([[self.parameters[i], self.initial_param_values[i]] for i in range(len(self.parameters))])
        for function in trace:
            function_name = function[0]
            function_args = function[1 : ]
            if function_name == 'sleep':
                duration = function_args[0]
                total_energy += state.get_energy(duration, param_dict)
                total_duration += duration
            else:
                transition = state.get_transition(function_name)
                total_duration += transition.get_duration(param_dict, function_args)
                total_energy += transition.get_energy(param_dict, function_args)
                param_dict = transition.get_params_after_transition(param_dict, function_args)
                state = transition.destination
                while (state.has_interrupt_transitions()):
                    transition = state.get_next_interrupt(param_dict)
                    duration = transition.get_timeout(param_dict)
                    total_duration += duration
                    total_energy += state.get_energy(duration, param_dict)
                    param_dict = transition.get_params_after_transition(param_dict)
                    state = transition.destination

        return total_energy, total_duration, state, param_dict

    def update(self, static_model, param_model):
        for state in self.states.values():
            if state.name != 'UNINITIALIZED':
                state.power = static_model(state.name, 'power')
                if param_model(state.name, 'power'):
                    state.power_function = param_model(state.name, 'power')['function']
                print(state.name, state.power, state.power_function.__dict__)