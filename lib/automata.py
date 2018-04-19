class Transition:
    def __init__(self, orig_state, dest_state, name, arguments, arg_param_map):
        self.name = name
        self.origin = orig_state
        self.destination = dest_state
        self.arguments = list(arguments)

class State:
    def __init__(self, name):
        self.name = name
        self.outgoing_transitions = []

    def add_outgoing_transition(self, new_transition):
        self.outgoing_transitions.append(new_transition)

    def dfs(self, depth):
        if depth == 0:
            return [[trans.name] for trans in self.outgoing_transitions]

        ret = []
        for trans in self.outgoing_transitions:
            for suffix in trans.destination.dfs(depth - 1):
                new_suffix = [trans.name]
                new_suffix.extend(suffix)
                ret.append(new_suffix)
        return ret

class PTA:
    def __init__(self, state_names, parameters):
        self.states = dict([[state_name, State(state_name)] for state_name in state_names])
        self.parameters = list(parameters)
        self.transitions = []

        if not 'UNINITIALIZED' in state_names:
            self.states['UNINITIALIZED'] = State('UNINITIALIZED')

    def add_transition(self, orig_state, dest_state, function_name, arguments, arg_param_map):
        orig_state = self.states[orig_state]
        dest_state = self.states[dest_state]
        new_transition = Transition(orig_state, dest_state, function_name, arguments, arg_param_map)
        self.transitions.append(new_transition)
        orig_state.add_outgoing_transition(new_transition)

    def dfs(self, depth = 10, orig_state = 'UNINITIALIZED'):
        return self.states[orig_state].dfs(depth)
