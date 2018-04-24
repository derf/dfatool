from dfatool import AnalyticFunction

def _parse_function(input_function):
    if type('input_function') == 'str':
        raise NotImplemented
    if type('input_function') == 'function':
        return 'raise ValueError', input_function
    raise ValueError('Function description must be provided as string or function')

def _dict_to_list(input_dict):
    return [input_dict[x] for x in sorted(input_dict.keys())]

class Transition:
    def __init__(self, orig_state, dest_state, name,
            energy = 0, energy_function = None,
            duration = 0, duration_function = None,
            timeout = 0, timeout_function = None,
            is_interrupt = False,
            arguments = [], param_update_function = None,
            arg_to_param_map = None):
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
        self.param_update_function = param_update_function
        self.arg_to_param_map = arg_to_param_map

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
        if self.arg_to_param_map:
            ret = param_dict.copy()
            for k, v in self.arg_to_param_map.items():
                ret[k] = args[v]
            return ret
        return param_dict

class State:
    def __init__(self, name, power = 0, power_function = None):
        self.name = name
        self.power = power
        self.power_function = power_function
        self.outgoing_transitions = {}

    """@classmethod
    def from_json(cls, serialized_state):
        if 'power' in serialized_state:
            cls.power = serialized_state['power']['static']
            if 'function' in serialized_state:
                cls.power_function = """

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

    def add_state(self, state_name, **kwargs):
        if 'power_function' in kwargs:
            kwargs['power_function'] = AnalyticFunction(kwargs['power_function'],
                self.parameters, 0)
        self.states[state_name] = State(state_name, **kwargs)

    def add_transition(self, orig_state, dest_state, function_name, **kwargs):
        orig_state = self.states[orig_state]
        dest_state = self.states[dest_state]
        for key in ('duration_function', 'energy_function', 'timeout_function'):
            if key in kwargs:
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
