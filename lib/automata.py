class Transition:
    def __init__(self, orig_state, dest_state, name,
            energy = 0, energy_function = None,
            duration = 0, duration_function = None,
            timeout = 0, timeout_function = None,
            is_interrupt = False,
            arguments = [], param_update_function = None):
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

    def get_duration(self, parameters = [], args = []):
        if self.duration_function:
            return self.duration_function(parameters, args)
        return self.duration

    def get_energy(self, parameters = [], args = []):
        if self.energy_function:
            return self.energy_function(parameters, args)
        return self.energy

    def get_timeout(self, parameters = []):
        if self.timeout_function:
            return self.timeout_function(parameters)
        return self.timeout

    def get_params_after_transition(self, parameters, args = []):
        if self.param_update_function:
            return self.param_update_function(parameters, args)
        return parameters

class State:
    def __init__(self, name, power = 0, power_function = None):
        self.name = name
        self.power = power
        self.power_function = power_function
        self.outgoing_transitions = {}

    def add_outgoing_transition(self, new_transition):
        self.outgoing_transitions[new_transition.name] = new_transition

    def get_energy(self, duration, parameters = []):
        if self.power_function:
            return self.power_function(parameters) * duration
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
        self.states[state_name] = State(state_name, **kwargs)

    def add_transition(self, orig_state, dest_state, function_name, **kwargs):
        orig_state = self.states[orig_state]
        dest_state = self.states[dest_state]
        new_transition = Transition(orig_state, dest_state, function_name, **kwargs)
        self.transitions.append(new_transition)
        orig_state.add_outgoing_transition(new_transition)

    def dfs(self, depth = 10, orig_state = 'UNINITIALIZED'):
        return self.states[orig_state].dfs(depth)

    def simulate(self, trace, orig_state = 'UNINITIALIZED'):
        total_duration = 0.
        total_energy = 0.
        state = self.states[orig_state]
        parameters = self.initial_param_values
        for function in trace:
            function_name = function[0]
            function_args = function[1 : ]
            if function_name == 'sleep':
                duration = function_args[0]
                total_energy += state.get_energy(duration, parameters)
                total_duration += duration
            else:
                transition = state.get_transition(function_name)
                total_duration += transition.get_duration(parameters, function_args)
                total_energy += transition.get_energy(parameters, function_args)
                parameters = transition.get_params_after_transition(parameters, function_args)
                state = transition.destination
                while (state.has_interrupt_transitions()):
                    transition = state.get_next_interrupt(parameters)
                    duration = transition.get_timeout(parameters)
                    total_duration += duration
                    total_energy += state.get_energy(duration, parameters)
                    parameters = transition.get_params_after_transition(parameters)
                    state = transition.destination

        return total_energy, total_duration, state, parameters
