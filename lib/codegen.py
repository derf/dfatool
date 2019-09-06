"""Code generators for multipass dummy drivers for online model evaluation."""

from automata import PTA

header_template = """
#ifndef DFATOOL_{name}_H
#define DFATOOL_{name}_H

{includes}

class {name}
{{
private:
{name}(const {name} &copy);
{private_variables}
{private_functions}

public:
{enums}
{public_variables}
{public_functions}
}};

extern {name} {name_lower};

#endif
"""

implementation_template = """
#include "driver/dummy.h"

{functions}

{name} {name_lower};
"""

array_template = """
{type} const {name}[{length}] = {{{elements}}};
"""

class ClassFunction:
    def __init__(self, class_name, return_type, name, arguments, body):
        """
        Create a new C++ class method wrapper.

        :param class_name: Class name
        :param return_type: function return type
        :param name: function name
        :param arguments: list of arguments (must contain type and name)
        :param body: function body (str)
        """
        self.class_name = class_name
        self.return_type = return_type
        self.name = name
        self.arguments = arguments
        self.body = body

    def get_definition(self):
        return '{} {}({});'.format(self.return_type, self.name, ', '.join(self.arguments))

    def get_implementation(self):
        if self.body is None:
            return ''
        return '{} {}::{}({}) {{\n{}}}\n'.format(self.return_type, self.class_name, self.name, ', '.join(self.arguments), self.body)

class AccountingMethod:
    def __init__(self, class_name: str, pta: PTA, ):
        self.class_name = class_name
        self.pta = pta
        self.include_paths = list()
        self.private_variables = list()
        self.public_variables = list()
        self.private_functions = list()
        self.public_functions = list()
    
    def pre_transition_hook(self, transition):
        return ''

    def init_code(self):
        return ''

    def get_includes(self):
        return map(lambda x: '#include "{}"'.format(x), self.include_paths)

class StaticStateOnlyAccountingImmediateCalculation(AccountingMethod):
    def __init__(self, class_name: str, pta: PTA, ts_type = 'unsigned int', power_type = 'unsigned int', energy_type = 'unsigned long'):
        super().__init__(class_name, pta)
        self.ts_type = ts_type
        self.include_paths.append('driver/uptime.h')
        self.private_variables.append('unsigned char lastState;')
        self.private_variables.append('{} lastStateChange;'.format(ts_type))
        self.private_variables.append('{} totalEnergy;'.format(energy_type))
        self.private_variables.append(array_template.format(
            type = power_type,
            name = 'state_power',
            length = len(pta.state),
            elements = ', '.join(map(lambda state_name: '{:.0f}'.format(pta.state[state_name].power), pta.get_state_names()))
        ))

        get_energy_function = """return totalEnergy;"""
        self.public_functions.append(ClassFunction(class_name, energy_type, 'getEnergy', list(), get_energy_function))

    def pre_transition_hook(self, transition):
        return """
        unsigned int now = uptime.get_us();
        totalEnergy += (now - lastStateChange) * state_power[lastState];
        lastStateChange = now;
        lastState = {};
        """.format(self.pta.get_state_id(transition.destination))

    def init_code(self):
        return """
        totalEnergy = 0;
        lastStateChange = 0;
        lastState = 0;
        """.format(num_states = len(self.pta.state))

class StaticStateOnlyAccounting(AccountingMethod):
    def __init__(self, class_name: str, pta: PTA, ts_type = 'unsigned int', power_type = 'unsigned int', energy_type = 'unsigned long'):
        super().__init__(class_name, pta)
        self.ts_type = ts_type
        self.include_paths.append('driver/uptime.h')
        self.private_variables.append('unsigned char lastState;')
        self.private_variables.append('{} lastStateChange;'.format(ts_type))
        self.private_variables.append(array_template.format(
            type = power_type,
            name = 'state_power',
            length = len(pta.state),
            elements = ', '.join(map(lambda state_name: '{:.0f}'.format(pta.state[state_name].power), pta.get_state_names()))
        ))
        self.private_variables.append('{} timeInState[{}];'.format(ts_type, len(pta.state)))

        get_energy_function = """
        {energy_type} total_energy = 0;
        for (int i = 0; i < {num_states}; i++) {{
            total_energy += timeInState[i] * state_power[i];
        }}
        return total_energy;
        """.format(energy_type = energy_type, num_states = len(pta.state))
        self.public_functions.append(ClassFunction(class_name, energy_type, 'getEnergy', list(), get_energy_function))

    def pre_transition_hook(self, transition):
        return """
        unsigned int now = uptime.get_us();
        timeInState[lastState] += now - lastStateChange;
        lastStateChange = now;
        lastState = {};
        """.format(self.pta.get_state_id(transition.destination))

    def init_code(self):
        return """
        for (unsigned char i = 0; i < {num_states}; i++) {{
            timeInState[i] = 0;
        }}
        lastState = 0;
        lastStateChange = 0;
        """.format(num_states = len(self.pta.state))


class MultipassDriver:
    """Generate C++ header and no-op implementation for a multipass driver based on a DFA model."""

    def __init__(self, name, pta, class_info, enum = dict(), accounting = AccountingMethod):
        self.impl = ''
        self.header = ''
        self.name = name
        self.pta = pta
        self.class_info = class_info
        self.enum = enum

        includes = list()
        private_functions = list()
        public_functions = list()
        private_variables = list()
        public_variables = list()

        public_functions.append(ClassFunction(self.name, '', self.name, list(), accounting.init_code()))
        seen_transitions = set()

        for transition in self.pta.transitions:
            if transition.name in seen_transitions:
                continue
            seen_transitions.add(transition.name)

            # XXX right now we only verify whether both functions have the
            # same number of arguments. This breaks in many overloading cases.
            function_info = self.class_info.function[transition.name]
            for function_candidate in self.class_info.functions:
                if function_candidate.name == transition.name and len(function_candidate.argument_types) == len(transition.arguments):
                    function_info = function_candidate

            function_arguments = list()

            for i in range(len(transition.arguments)):
                function_arguments.append('{} {}'.format(function_info.argument_types[i], transition.arguments[i]))

            function_definition = '{} {}({})'.format(function_info.return_type, transition.name, ', '.join(function_arguments))
            function_head = '{} {}::{}({})'.format(function_info.return_type, self.name, transition.name, ', '.join(function_arguments))

            function_body = accounting.pre_transition_hook(transition)

            if function_info.return_type != 'void':
                function_body += 'return 0;\n'

            public_functions.append(ClassFunction(self.name, function_info.return_type, transition.name, function_arguments, function_body))

        enums = list()
        for enum_name in self.enum.keys():
            enums.append('enum {} {{ {} }};'.format(enum_name, ', '.join(self.enum[enum_name])))

        if accounting:
            includes.extend(accounting.get_includes())
            private_functions.extend(accounting.private_functions)
            public_functions.extend(accounting.public_functions)
            private_variables.extend(accounting.private_variables)
            public_variables.extend(accounting.public_variables)

        self.header = header_template.format(
            name = self.name, name_lower = self.name.lower(),
            includes = '\n'.join(includes),
            private_variables = '\n'.join(private_variables),
            public_variables = '\n'.join(public_variables),
            public_functions = '\n'.join(map(lambda x: x.get_definition(), public_functions)),
            private_functions = '',
            enums = '\n'.join(enums))
        self.impl = implementation_template.format(name = self.name, name_lower = self.name.lower(), functions = '\n\n'.join(map(lambda x: x.get_implementation(), public_functions)))
