"""Code generators for multipass dummy drivers for online model evaluation."""

from automata import PTA, Transition
from modular_arithmetic import simulate_int_type

header_template = """
#ifndef DFATOOL_{name}_H
#define DFATOOL_{name}_H

#include "stdint.h"

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


def get_accountingmethod(method):
    """Return AccountingMethod class for method."""
    if method == 'static_state_immediate':
        return StaticStateOnlyAccountingImmediateCalculation
    if method == 'static_state':
        return StaticStateOnlyAccounting
    if method == 'static_statetransition_immediate':
        return StaticAccountingImmediateCalculation
    if method == 'static_statetransition':
        return StaticAccounting
    raise ValueError('Unknown accounting method: {}'.format(method))


def get_simulated_accountingmethod(method):
    """Return SimulatedAccountingMethod class for method."""
    if method == 'static_state_immediate':
        return SimulatedStaticStateOnlyAccountingImmediateCalculation
    if method == 'static_statetransition_immediate':
        return SimulatedStaticAccountingImmediateCalculation
    if method == 'static_state':
        return SimulatedStaticStateOnlyAccounting
    if method == 'static_statetransition':
        return SimulatedStaticAccounting
    raise ValueError('Unknown accounting method: {}'.format(method))


class SimulatedAccountingMethod:
    """
    Simulates overflows and timing inaccuracies in online energy accounting on embedded devices.

    Inaccuracies are based on:
    * timer resolution (e.g. a 10kHz timer cannot reliably measure sub-100us timings)
    * timer counter size (e.g. a 16-bit timer at 1MHz will overflow after 65us)
    * variable size for accounting of durations, power and energy values
    """

    def __init__(self, pta: PTA, timer_freq_hz, timer_type, ts_type, power_type, energy_type, ts_granularity=1e-6, power_granularity=1e-6, energy_granularity=1e-12):
        """
        Simulate Online Accounting for a given PTA.

        :param pta: PTA object
        :param timer_freq_hz: Frequency of timer used for state time measurement, in Hz
        :param timer_type: Size of timer counter register, as C standard type (uint8_t / uint16_t / uint32_t / uint64_t)
        :param ts_type: Size of timestamp variables, as C standard type
        :param power_type: Size of power variables, as C standard type
        :param energy_type: Size of energy variables, as C standard type
        """
        self.pta = pta
        self.timer_freq_hz = timer_freq_hz
        self.timer_class = simulate_int_type(timer_type)
        self.ts_class = simulate_int_type(ts_type)
        self.power_class = simulate_int_type(power_type)
        self.energy_class = simulate_int_type(energy_type)
        self.current_state = pta.state['UNINITIALIZED']

        self.ts_granularity = ts_granularity
        self.power_granularity = power_granularity
        self.energy_granularity = energy_granularity

        """Energy in pJ."""
        self.energy = self.energy_class(0)

    def _energy_from_power_and_time(self, power, time):
        """
        Return energy (=power * time), accounting for configured granularity.

        Does not use Module types and therefore does not consider overflows or data-type limitations"""
        if self.energy_granularity == self.power_granularity * self.ts_granularity:
            return power * time
        return int(power * self.power_granularity * time * self.ts_granularity / self.energy_granularity)

    def _sleep_duration(self, duration_us):
        u"""
        Return the sleep duration a timer with the configured timer frequency would measure, according to the configured granularity.

        I.e., for a 35us sleep with a 50kHz timer (-> one tick per 20us) and 1us time resolution, the OS would likely measure one tick == 20us.
        This is based on the assumption that the timer is reset at each transition, so the duration of states may be under-, but not over-estimated
        """
        us_per_tick = 1000000 / self.timer_freq_hz
        ticks = self.timer_class(int(duration_us // us_per_tick))
        time_units_per_tick = 1 / (self.timer_freq_hz * self.ts_granularity)
        return int(ticks.val * time_units_per_tick)

    def sleep(self, duration_us):
        pass

    def pass_transition(self, transition: Transition):
        """Updates current state to `transition.destination`."""
        self.current_state = transition.destination

    def get_energy(self):
        """Return total energy in pJ."""
        return self.energy.val * self.energy_granularity * 1e12


class SimulatedStaticStateOnlyAccountingImmediateCalculation(SimulatedAccountingMethod):
    """
    Simulated state-only energy accounting with immediate calculation.

    Does not use functions or LUTs, only static (median) state power.
    Transitions are assumed to be immediate and have negligible energy overhead.

    Keeps track of the current state and the time it is active. On each
    transition, current state power and duration is used to update the
    total energy spent.
    """

    def __init__(self, pta: PTA, *args, **kwargs):
        super().__init__(pta, *args, **kwargs)

    def sleep(self, duration_us):
        time = self._sleep_duration(duration_us)
        power = int(self.current_state.power.value)
        energy = self._energy_from_power_and_time(time, power)
        self.energy += energy


class SimulatedStaticAccountingImmediateCalculation(SimulatedAccountingMethod):
    """
    Simulated energy accounting with states and transitions, immediate calculation.

    Does not use functions or LUTs, only static (median) state power and transition energ.

    Keeps track of the current state and the time it is active. On each
    transition, current state power and duration is used to calculate the
    energy spent in the state, which is used in conjunction with the
    transition's energy cost to update the total energy spent.
    """

    def __init__(self, pta: PTA, *args, **kwargs):
        super().__init__(pta, *args, **kwargs)

    def sleep(self, duration_us):
        time = self._sleep_duration(duration_us)
        print('sleep duration is {}'.format(time))
        power = int(self.current_state.power.value)
        print('power is {}'.format(power))
        energy = self._energy_from_power_and_time(time, power)
        print('energy is {}'.format(energy))
        self.energy += energy

    def pass_transition(self, transition: Transition):
        self.energy += int(transition.energy.value)
        super().pass_transition(transition)


class SimulatedStaticAccounting(SimulatedAccountingMethod):
    """
    Simulated energy accounting with states and transitions, deferred energy calculation.

    Does not use functions or LUTs, only static (median) state power and transition energ.

    Keeps track of the time spent in each state and the number of calls for
    each transition. This data is update whenever passing a transition and used
    to calculate total energy spent on-demand: E = sum(P_q * t_q) + sum(E_t * n_t).
    """

    def __init__(self, pta: PTA, *args, **kwargs):
        super().__init__(pta, *args, **kwargs)
        self.time_in_state = dict()
        for state_name in pta.state.keys():
            self.time_in_state[state_name] = self.ts_class(0)
        self.transition_count = list()
        for transition in pta.transitions:
            self.transition_count.append(simulate_int_type('uint16_t')(0))

    def sleep(self, duration_us):
        self.time_in_state[self.current_state.name] += self._sleep_duration(duration_us)

    def pass_transition(self, transition: Transition):
        self.transition_count[self.pta.transitions.index(transition)] += 1
        super().pass_transition(transition)

    def get_energy(self):
        pta = self.pta
        energy = self.energy_class(0)
        for state in pta.state.values():
            energy += self._energy_from_power_and_time(self.time_in_state[state.name], int(state.power.value))
        for i, transition in enumerate(pta.transitions):
            energy += self.transition_count[i] * int(transition.energy.value)
        return energy.val


class SimulatedStaticStateOnlyAccounting(SimulatedAccountingMethod):
    """
    Simulated energy accounting with states and transitions, deferred energy calculation.

    Does not use functions or LUTs, only static (median) state power and transition energ.

    Keeps track of the time spent in each state and the number of calls for
    each transition. This data is update whenever passing a transition and used
    to calculate total energy spent on-demand: E = sum(P_q * t_q) + sum(E_t * n_t).
    """

    def __init__(self, pta: PTA, *args, **kwargs):
        super().__init__(pta, *args, **kwargs)
        self.time_in_state = dict()
        for state_name in pta.state.keys():
            self.time_in_state[state_name] = self.ts_class(0)

    def sleep(self, duration_us):
        self.time_in_state[self.current_state.name] += self._sleep_duration(duration_us)

    def get_energy(self):
        pta = self.pta
        energy = self.energy_class(0)
        for state in pta.state.values():
            energy += self._energy_from_power_and_time(self.time_in_state[state.name], int(state.power.value))
        return energy.val


class AccountingMethod:
    def __init__(self, class_name: str, pta: PTA):
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
    def __init__(self, class_name: str, pta: PTA, ts_type='unsigned int', power_type='unsigned int', energy_type='unsigned long'):
        super().__init__(class_name, pta)
        self.ts_type = ts_type
        self.include_paths.append('driver/uptime.h')
        self.private_variables.append('unsigned char lastState;')
        self.private_variables.append('{} lastStateChange;'.format(ts_type))
        self.private_variables.append('{} totalEnergy;'.format(energy_type))
        self.private_variables.append(array_template.format(
            type=power_type,
            name='state_power',
            length=len(pta.state),
            elements=', '.join(map(lambda state_name: '{:.0f}'.format(pta.state[state_name].power), pta.get_state_names()))
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
        """.format(num_states=len(self.pta.state))


class StaticStateOnlyAccounting(AccountingMethod):
    def __init__(self, class_name: str, pta: PTA, ts_type='unsigned int', power_type='unsigned int', energy_type='unsigned long'):
        super().__init__(class_name, pta)
        self.ts_type = ts_type
        self.include_paths.append('driver/uptime.h')
        self.private_variables.append('unsigned char lastState;')
        self.private_variables.append('{} lastStateChange;'.format(ts_type))
        self.private_variables.append(array_template.format(
            type=power_type,
            name='state_power',
            length=len(pta.state),
            elements=', '.join(map(lambda state_name: '{:.0f}'.format(pta.state[state_name].power), pta.get_state_names()))
        ))
        self.private_variables.append('{} timeInState[{}];'.format(ts_type, len(pta.state)))

        get_energy_function = """
        {energy_type} total_energy = 0;
        for (int i = 0; i < {num_states}; i++) {{
            total_energy += timeInState[i] * state_power[i];
        }}
        return total_energy;
        """.format(energy_type=energy_type, num_states=len(pta.state))
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
        """.format(num_states=len(self.pta.state))


class StaticAccounting(AccountingMethod):
    def __init__(self, class_name: str, pta: PTA, ts_type='unsigned int', power_type='unsigned int', energy_type='unsigned long'):
        super().__init__(class_name, pta)
        self.ts_type = ts_type
        self.include_paths.append('driver/uptime.h')
        self.private_variables.append('unsigned char lastState;')
        self.private_variables.append('{} lastStateChange;'.format(ts_type))
        self.private_variables.append(array_template.format(
            type=power_type,
            name='state_power',
            length=len(pta.state),
            elements=', '.join(map(lambda state_name: '{:.0f}'.format(pta.state[state_name].power), pta.get_state_names()))
        ))
        self.private_variables.append(array_template.format(
            type=energy_type,
            name='transition_energy',
            length=len(pta.get_unique_transitions()),
            elements=', '.join(map(lambda transition: '{:.0f}'.format(transition.energy), pta.get_unique_transitions()))
        ))
        self.private_variables.append('{} timeInState[{}];'.format(ts_type, len(pta.state)))
        self.private_variables.append('{} transitionCount[{}];'.format('unsigned int', len(pta.get_unique_transitions())))

        get_energy_function = """
        {energy_type} total_energy = 0;
        for (unsigned char i = 0; i < {num_states}; i++) {{
            total_energy += timeInState[i] * state_power[i];
        }}
        for (unsigned char i = 0; i < {num_transitions}; i++) {{
            total_energy += transitionCount[i] * transition_energy[i];
        }}
        return total_energy;
        """.format(energy_type=energy_type, num_states=len(pta.state), num_transitions=len(pta.get_unique_transitions()))
        self.public_functions.append(ClassFunction(class_name, energy_type, 'getEnergy', list(), get_energy_function))

    def pre_transition_hook(self, transition):
        return """
        unsigned int now = uptime.get_us();
        timeInState[lastState] += now - lastStateChange;
        transitionCount[{}]++;
        lastStateChange = now;
        lastState = {};
        """.format(self.pta.get_unique_transition_id(transition), self.pta.get_state_id(transition.destination))

    def init_code(self):
        return """
        for (unsigned char i = 0; i < {num_states}; i++) {{
            timeInState[i] = 0;
        }}
        for (unsigned char i = 0; i < {num_transitions}; i++) {{
            transitionCount[i] = 0;
        }}
        lastState = 0;
        lastStateChange = 0;
        """.format(num_states=len(self.pta.state), num_transitions=len(self.pta.get_unique_transitions()))


class StaticAccountingImmediateCalculation(AccountingMethod):
    def __init__(self, class_name: str, pta: PTA, ts_type='unsigned int', power_type='unsigned int', energy_type='unsigned long'):
        super().__init__(class_name, pta)
        self.ts_type = ts_type
        self.include_paths.append('driver/uptime.h')
        self.private_variables.append('unsigned char lastState;')
        self.private_variables.append('{} lastStateChange;'.format(ts_type))
        self.private_variables.append('{} totalEnergy;'.format(energy_type))
        self.private_variables.append(array_template.format(
            type=power_type,
            name='state_power',
            length=len(pta.state),
            elements=', '.join(map(lambda state_name: '{:.0f}'.format(pta.state[state_name].power), pta.get_state_names()))
        ))

        get_energy_function = """
        return totalEnergy;
        """.format(energy_type=energy_type, num_states=len(pta.state), num_transitions=len(pta.get_unique_transitions()))
        self.public_functions.append(ClassFunction(class_name, energy_type, 'getEnergy', list(), get_energy_function))

    def pre_transition_hook(self, transition):
        return """
        unsigned int now = uptime.get_us();
        totalEnergy += (now - lastStateChange) * state_power[lastState];
        totalEnergy += {};
        lastStateChange = now;
        lastState = {};
        """.format(transition.energy, self.pta.get_state_id(transition.destination))

    def init_code(self):
        return """
        lastState = 0;
        lastStateChange = 0;
        """.format(num_states=len(self.pta.state), num_transitions=len(self.pta.get_unique_transitions()))


class MultipassDriver:
    """Generate C++ header and no-op implementation for a multipass driver based on a DFA model."""

    def __init__(self, name, pta, class_info, enum=dict(), accounting=AccountingMethod):
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

        for transition in self.pta.get_unique_transitions():

            if transition.name == 'getEnergy':
                continue

            # XXX right now we only verify whether both functions have the
            # same number of arguments. This breaks in many overloading cases.
            function_info = self.class_info.function[transition.name]
            for function_candidate in self.class_info.functions:
                if function_candidate.name == transition.name and len(function_candidate.argument_types) == len(transition.arguments):
                    function_info = function_candidate

            function_arguments = list()

            for i in range(len(transition.arguments)):
                function_arguments.append('{} {}'.format(function_info.argument_types[i], transition.arguments[i]))

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
            name=self.name, name_lower=self.name.lower(),
            includes='\n'.join(includes),
            private_variables='\n'.join(private_variables),
            public_variables='\n'.join(public_variables),
            public_functions='\n'.join(map(lambda x: x.get_definition(), public_functions)),
            private_functions='',
            enums='\n'.join(enums))
        self.impl = implementation_template.format(name=self.name, name_lower=self.name.lower(), functions='\n\n'.join(map(lambda x: x.get_implementation(), public_functions)))
