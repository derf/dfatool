"""Code generators for multipass dummy drivers for online model evaluation."""

header_template = """
#ifndef DFATOOL_{name}_H
#define DFATOOL_{name}_H

class {name}
{{
private:
{name}(const {name} &copy);

public:
{enums}
{functions}
}};

extern {name} {name_lower};

#endif
"""

implementation_template = """
#include "driver/dummy.h"

{functions}

{name} {name_lower};
"""

class MultipassDriver:
    """Generate C++ header and no-op implementation for a multipass driver based on a DFA model."""

    def __init__(self, name, pta, class_info, enum = dict()):
        self.impl = ''
        self.header = ''
        self.name = name
        self.pta = pta
        self.class_info = class_info
        self.enum = enum

        function_definitions = list()
        function_bodies = list()

        function_definitions.append('{}() {{}}'.format(self.name))
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

            function_body = str()

            if function_info.return_type != 'void':
                function_body = 'return 0;\n'

            function_definitions.append(function_definition + ';')
            function_bodies.append('{} {{\n{}}}'.format(function_head, function_body))

        enums = list()
        for enum_name in self.enum.keys():
            enums.append('enum {} {{ {} }};'.format(enum_name, ', '.join(self.enum[enum_name])))

        self.header = header_template.format(name = self.name, name_lower = self.name.lower(), functions = '\n'.join(function_definitions), enums = '\n'.join(enums))
        self.impl = implementation_template.format(name = self.name, name_lower = self.name.lower(), functions = '\n\n'.join(function_bodies))
