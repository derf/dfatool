"""
Harnesses for various types of benchmark logs.

tbd
"""
import subprocess
import re
from pubcode import Code128

class TransitionHarness:
    """
    TODO

    :param done: True if the specified amount of iterations have been logged.
    :param synced: True if `parser_cb` has synchronized with UART output, i.e., the benchmark has successfully started.
    :param traces: List of annotated PTA traces from benchmark execution. This list is updated during UART logging and should only be read back when `done` is True.
        Uses the standard dfatool trace format: `traces` is a list of `{'id': ..., 'trace': ...}` dictionaries, each of which represents a single PTA trace (AKA
        run). Each `trace` is in turn a list of state or transition dictionaries with the
        following attributes:
        * `isa`: 'state' or 'transition'
        * `name`: state or transition name
        * `parameter`: currently valid parameter values. If normalization is used, they are already normalized. Each parameter value is either a primitive
          int/float/str value (-> constant for each iteration) or a list of
          primitive values (-> set by the return value of the current run, not necessarily constan)
        * `args`: function arguments, if isa == 'transition'
    """
    def __init__(self, gpio_pin = None, gpio_mode = 'around', pta = None, log_return_values = False, repeat = 0, post_transition_delay_us = 0):
        """
        Create a new TransitionHarness

        :param gpio_pin: multipass GPIO Pin used for transition synchronization with an external measurement device, e.g. `GPIO::p1_0`. Optional.
            The GPIO output is high iff a transition is executing
        :param pta: PTA object. Needed to map UART output IDs to states and transitions
        :param log_return_values: Log return values of transition function calls?
        :param repeat: How many times to run the benchmark until setting `one`, default 0.
            When 0, `done` is never set.
        :param post_transition_delay_us: If set, inject `arch.delay_us` after each transition, before logging the transition as completed (and releasing
            `gpio_pin`). This artificially increases transition duration by the specified time and is useful if an external measurement device's resolution is
            lower than the expected minimum transition duration.
        """
        self.gpio_pin = gpio_pin
        self.gpio_mode = gpio_mode
        self.pta = pta
        self.log_return_values = log_return_values
        self.repeat = repeat
        self.post_transition_delay_us = post_transition_delay_us
        self.reset()

    def copy(self):
        new_object = __class__(gpio_pin = self.gpio_pin, gpio_mode = self.gpio_mode, pta = self.pta, log_return_values = self.log_return_values, repeat = self.repeat, post_transition_delay_us = self.post_transition_delay_us)
        new_object.traces = self.traces.copy()
        new_object.trace_id = self.trace_id
        return new_object

    def undo(self, undo_from):
        """
        Undo all benchmark runs starting with index `undo_from`.

        :param undo_from: index of measurements to be undone. Measurementh with a higher index (i.e., which happened later) will also be undone.

        Removes all logged results (nondeterministic parameter values and return values)
        of the current benchmark iteration. Resets `done` and `synced`,
        """
        for trace in self.traces:
            for state_or_transition in trace['trace']:
                if 'return_values' in state_or_transition:
                    state_or_transition['return_values'] = state_or_transition['return_values'][:undo_from]
                for param_name in state_or_transition['parameter'].keys():
                    if type(state_or_transition['parameter'][param_name]) is list:
                        state_or_transition['parameter'][param_name] = state_or_transition['parameter'][param_name][:undo_from]

    def reset(self):
        """
        Reset harness for a new benchmark.

        Truncates `traces`, `trace_id`, `done`, and `synced`.
        """
        self.traces = []
        self.trace_id = 0
        self.repetitions = 0
        self.abort = False
        self.done = False
        self.synced = False

    def restart(self):
        """
        Reset harness for a new execution of the current benchmark.

        Resets `done` and `synced`.
        """
        self.repetitions = 0
        self.abort = False
        self.done = False
        self.synced = False

    def global_code(self):
        """Return global (pre-`main()`) C++ code needed for tracing."""
        ret = ''
        if self.gpio_pin != None:
            ret += '#define PTALOG_GPIO {}\n'.format(self.gpio_pin)
            if self.gpio_mode == 'before':
                ret += '#define PTALOG_GPIO_BEFORE\n'
            elif self.gpio_mode == 'bar':
                ret += '#define PTALOG_GPIO_BAR\n'
        if self.log_return_values:
            ret += '#define PTALOG_WITH_RETURNVALUES\n'
            ret += 'uint16_t transition_return_value;\n'
        ret += '#include "object/ptalog.h"\n'
        if self.gpio_pin != None:
            ret += 'PTALog ptalog({});\n'.format(self.gpio_pin)
        else:
            ret += 'PTALog ptalog;\n'
        return ret

    def start_benchmark(self, benchmark_id = 0):
        """Return C++ code to signal benchmark start to harness."""
        return 'ptalog.startBenchmark({:d});\n'.format(benchmark_id)

    def start_trace(self):
        """Prepare a new trace/run in the internal `.traces` structure."""
        self.traces.append({
            'id' : self.trace_id,
            'trace' : list(),
        })
        self.trace_id += 1

    def append_state(self, state_name, param):
        """
        Append a state to the current run in the internal `.traces` structure.

        :param state_name: state name
        :param param: parameter dict
        """
        self.traces[-1]['trace'].append({
            'name': state_name,
            'isa': 'state',
            'parameter': param,
        })

    def append_transition(self, transition_name, param, args = []):
        """
        Append a transition to the current run in the internal `.traces` structure.

        :param transition_name: transition name
        :param param: parameter dict
        :param args: function arguments (optional)
        """
        self.traces[-1]['trace'].append({
            'name': transition_name,
            'isa': 'transition',
            'parameter': param,
            'args' : args,
        })

    def start_run(self):
        """Return C++ code used to start a new run/trace."""
        return 'ptalog.reset();\n'

    def _pass_transition_call(self, transition_id):
        if self.gpio_mode == 'bar':
            barcode_bits = Code128('T{}'.format(transition_id), charset='B').modules
            if len(barcode_bits) % 8 != 0:
                barcode_bits.extend([1] * (8 - (len(barcode_bits) % 8)))
            barcode_bytes = [255 - int("".join(map(str, reversed(barcode_bits[i:i+8]))), 2) for i in range(0, len(barcode_bits), 8)]
            inline_array = "".join(map(lambda s: '\\x{:02x}'.format(s), barcode_bytes))
            return 'ptalog.startTransition("{}", {});\n'.format(inline_array, len(barcode_bytes))
        else:
            return 'ptalog.startTransition();\n'

    def pass_transition(self, transition_id, transition_code, transition: object = None):
        """
        Return C++ code used to pass a transition, including the corresponding function call.

        Tracks which transition has been executed and optionally its return value. May also inject a delay, if
        `post_transition_delay_us` is set.
        """
        ret = 'ptalog.passTransition({:d});\n'.format(transition_id)
        ret += self._pass_transition_call(transition_id)
        if self.log_return_values and transition and len(transition.return_value_handlers):
            ret += 'transition_return_value = {}\n'.format(transition_code)
            ret += 'ptalog.logReturn(transition_return_value);\n'
        else:
            ret += '{}\n'.format(transition_code)
        if self.post_transition_delay_us:
            ret += 'arch.delay_us({});\n'.format(self.post_transition_delay_us)
        ret += 'ptalog.stopTransition();\n'
        return ret

    def stop_run(self, num_traces = 0):
        return 'ptalog.dump({:d});\n'.format(num_traces)

    def stop_benchmark(self):
        return 'ptalog.stopBenchmark();\n'

    def _append_nondeterministic_parameter_value(self, log_data_target, parameter_name, parameter_value):
        if log_data_target['parameter'][parameter_name] is None:
            log_data_target['parameter'][parameter_name] = list()
        log_data_target['parameter'][parameter_name].append(parameter_value)

    def parser_cb(self, line):
        #print('[HARNESS] got line {}'.format(line))
        if re.match(r'\[PTA\] benchmark stop', line):
            self.repetitions += 1
            self.synced = False
            if self.repeat > 0 and self.repetitions == self.repeat:
                self.done = True
                print('[HARNESS] done')
                return
        if re.match(r'\[PTA\] benchmark start, id=(\S+)', line):
            self.synced = True
            print('[HARNESS] synced, {}/{}'.format(self.repetitions + 1, self.repeat))
        if self.synced:
            res = re.match(r'\[PTA\] trace=(\S+) count=(\S+)', line)
            if res:
                self.trace_id = int(res.group(1))
                self.trace_length = int(res.group(2))
                self.current_transition_in_trace = 0
            if self.log_return_values:
                res = re.match(r'\[PTA\] transition=(\S+) return=(\S+)', line)
            else:
                res = re.match(r'\[PTA\] transition=(\S+)', line)
            if res:
                transition_id = int(res.group(1))
                # self.traces contains transitions and states, UART output only contains transitions -> use index * 2
                try:
                    log_data_target = self.traces[self.trace_id]['trace'][self.current_transition_in_trace * 2]
                except IndexError:
                    transition_name = None
                    if self.pta:
                        transition_name = self.pta.transitions[transition_id].name
                    print('[HARNESS] benchmark id={:d} trace={:d}: transition #{:d} (ID {:d}, name {}) is out of bounds'.format(0, self.trace_id, self.current_transition_in_trace, transition_id, transition_name))
                    print('          Offending line: {}'.format(line))
                    return
                if log_data_target['isa'] != 'transition':
                    self.abort = True
                    raise RuntimeError('Log mismatch: Expected transition, got {:s}'.format(log_data_target['isa']))
                if self.pta:
                    transition = self.pta.transitions[transition_id]
                    if transition.name != log_data_target['name']:
                        self.abort = True
                        raise RuntimeError('Log mismatch: Expected transition {:s}, got transition {:s} -- may have been caused by preceding malformed UART output'.format(log_data_target['name'], transition.name))
                    if self.log_return_values and len(transition.return_value_handlers):
                        for handler in transition.return_value_handlers:
                            if 'parameter' in handler:
                                parameter_value = return_value = int(res.group(2))

                                if 'return_values' not in log_data_target:
                                    log_data_target['return_values'] = list()
                                log_data_target['return_values'].append(return_value)

                                if 'formula' in handler:
                                    parameter_value = handler['formula'].eval(return_value)

                                self._append_nondeterministic_parameter_value(log_data_target, handler['parameter'], parameter_value)
                                for following_log_data_target in self.traces[self.trace_id]['trace'][(self.current_transition_in_trace * 2 + 1) :]:
                                    self._append_nondeterministic_parameter_value(following_log_data_target, handler['parameter'], parameter_value)
                                if 'apply_from' in handler and any(map(lambda x: x['name'] == handler['apply_from'], self.traces[self.trace_id]['trace'][: (self.current_transition_in_trace * 2 + 1)])):
                                    for preceding_log_data_target in reversed(self.traces[self.trace_id]['trace'][: (self.current_transition_in_trace * 2)]):
                                        self._append_nondeterministic_parameter_value(preceding_log_data_target, handler['parameter'], parameter_value)
                                        if preceding_log_data_target['name'] == handler['apply_from']:
                                            break
                self.current_transition_in_trace += 1

class OnboardTimerHarness(TransitionHarness):
    """TODO

    Additional parameters / changes from TransitionHarness:

    :param traces: Each trace element (`.traces[*]['trace'][*]`) additionally contains
        the dict `offline_aggregates` with the member `duration`. It contains a list of durations (in us) of the corresponding state/transition for each
        benchmark iteration.
        I.e. `.traces[*]['trace'][*]['offline_aggregates']['duration'] = [..., ...]`
    """
    def __init__(self, counter_limits, **kwargs):
        super().__init__(**kwargs)
        self.trace_length = 0
        self.one_cycle_in_us, self.one_overflow_in_us, self.counter_max_overflow = counter_limits

    def copy(self):
        new_harness = __class__((self.one_cycle_in_us, self.one_overflow_in_us, self.counter_max_overflow), gpio_pin = self.gpio_pin, gpio_mode = self.gpio_mode, pta = self.pta, log_return_values = self.log_return_values, repeat = self.repeat)
        new_harness.traces = self.traces.copy()
        new_harness.trace_id = self.trace_id
        return new_harness

    def undo(self, undo_from):
        """
        Undo all benchmark runs starting with index `undo_from`.

        :param undo_from: index of measurements to be undone. Measurementh with a higher index (i.e., which happened later) will also be undone.

        Removes all logged results (durations, nondeterministic parameter values, return values)
        of the current benchmark iteration. Resets `done` and `synced`,
        """
        super().undo(undo_from)
        for trace in self.traces:
            for state_or_transition in trace['trace']:
                if 'offline_aggregates' in state_or_transition:
                    state_or_transition['offline_aggregates']['duration'] = state_or_transition['offline_aggregates']['duration'][:undo_from]

    def global_code(self):
        ret = '#include "driver/counter.h"\n'
        ret += '#define PTALOG_TIMING\n'
        ret += super().global_code()
        return ret

    def start_benchmark(self, benchmark_id = 0):
        ret = 'counter.start();\n'
        ret += 'counter.stop();\n'
        ret += 'ptalog.passNop(counter);\n'
        ret += super().start_benchmark(benchmark_id)
        return ret

    def pass_transition(self, transition_id, transition_code, transition: object = None):
        ret = 'ptalog.passTransition({:d});\n'.format(transition_id)
        ret += self._pass_transition_call(transition_id)
        ret += 'counter.start();\n'
        if self.log_return_values and transition and len(transition.return_value_handlers):
            ret += 'transition_return_value = {}\n'.format(transition_code)
        else:
            ret += '{}\n'.format(transition_code)
        ret += 'counter.stop();\n'
        if self.log_return_values and transition and len(transition.return_value_handlers):
            ret += 'ptalog.logReturn(transition_return_value);\n'
        ret += 'ptalog.stopTransition(counter);\n'
        return ret

    def _append_nondeterministic_parameter_value(self, log_data_target, parameter_name, parameter_value):
        if log_data_target['parameter'][parameter_name] is None:
            log_data_target['parameter'][parameter_name] = list()
        log_data_target['parameter'][parameter_name].append(parameter_value)

    def parser_cb(self, line):
        # print('[HARNESS] got line {}'.format(line))
        res = re.match(r'\[PTA\] nop=(\S+)/(\S+)', line)
        if res:
            self.nop_cycles = int(res.group(1))
            if int(res.group(2)):
                raise RuntimeError('Counter overflow ({:d}/{:d}) during NOP test, wtf?!'.format(res.group(1), res.group(2)))
        if re.match(r'\[PTA\] benchmark stop', line):
            self.repetitions += 1
            self.synced = False
            if self.repeat > 0 and self.repetitions == self.repeat:
                self.done = True
                print('[HARNESS] done')
                return
        # May be repeated, e.g. if the device is reset shortly after start by
        # EnergyTrace.
        if re.match(r'\[PTA\] benchmark start, id=(\S+)', line):
            self.synced = True
            print('[HARNESS] synced, {}/{}'.format(self.repetitions + 1, self.repeat))
        if self.synced:
            res = re.match(r'\[PTA\] trace=(\S+) count=(\S+)', line)
            if res:
                self.trace_id = int(res.group(1))
                self.trace_length = int(res.group(2))
                self.current_transition_in_trace = 0
            if self.log_return_values:
                res = re.match(r'\[PTA\] transition=(\S+) cycles=(\S+)/(\S+) return=(\S+)', line)
            else:
                res = re.match(r'\[PTA\] transition=(\S+) cycles=(\S+)/(\S+)', line)
            if res:
                transition_id = int(res.group(1))
                cycles = int(res.group(2))
                overflow = int(res.group(3))
                if overflow >= self.counter_max_overflow:
                    self.abort = True
                    raise RuntimeError('Counter overflow ({:d}/{:d}) in benchmark id={:d} trace={:d}: transition #{:d} (ID {:d})'.format(cycles, overflow, 0, self.trace_id, self.current_transition_in_trace, transition_id))
                duration_us = cycles * self.one_cycle_in_us + overflow * self.one_overflow_in_us - self.nop_cycles * self.one_cycle_in_us
                if duration_us < 0:
                    duration_us = 0
                # self.traces contains transitions and states, UART output only contains transitions -> use index * 2
                try:
                    log_data_target = self.traces[self.trace_id]['trace'][self.current_transition_in_trace * 2]
                except IndexError:
                    transition_name = None
                    if self.pta:
                        transition_name = self.pta.transitions[transition_id].name
                    print('[HARNESS] benchmark id={:d} trace={:d}: transition #{:d} (ID {:d}, name {}) is out of bounds'.format(0, self.trace_id, self.current_transition_in_trace, transition_id, transition_name))
                    print('          Offending line: {}'.format(line))
                    return
                if log_data_target['isa'] != 'transition':
                    self.abort = True
                    raise RuntimeError('Log mismatch in benchmark id={:d} trace={:d}: transition #{:d} (ID {:d}): Expected transition, got {:s}'.format(0,
                        self.trace_id, self.current_transition_in_trace, transition_id, log_data_target['isa']))
                if self.pta:
                    transition = self.pta.transitions[transition_id]
                    if transition.name != log_data_target['name']:
                        self.abort = True
                        raise RuntimeError('Log mismatch in benchmark id={:d} trace={:d}: transition #{:d} (ID {:d}): Expected transition {:s}, got transition {:s} -- may have been caused by preceding maformed UART output'.format(0, self.trace_id, self.current_transition_in_trace, transition_id, log_data_target['name'], transition.name, line))
                    if self.log_return_values and len(transition.return_value_handlers):
                        for handler in transition.return_value_handlers:
                            if 'parameter' in handler:
                                parameter_value = return_value = int(res.group(4))

                                if 'return_values' not in log_data_target:
                                    log_data_target['return_values'] = list()
                                log_data_target['return_values'].append(return_value)

                                if 'formula' in handler:
                                    parameter_value = handler['formula'].eval(return_value)

                                self._append_nondeterministic_parameter_value(log_data_target, handler['parameter'], parameter_value)
                                for following_log_data_target in self.traces[self.trace_id]['trace'][(self.current_transition_in_trace * 2 + 1) :]:
                                    self._append_nondeterministic_parameter_value(following_log_data_target, handler['parameter'], parameter_value)
                                if 'apply_from' in handler and any(map(lambda x: x['name'] == handler['apply_from'], self.traces[self.trace_id]['trace'][: (self.current_transition_in_trace * 2 + 1)])):
                                    for preceding_log_data_target in reversed(self.traces[self.trace_id]['trace'][: (self.current_transition_in_trace * 2)]):
                                        self._append_nondeterministic_parameter_value(preceding_log_data_target, handler['parameter'], parameter_value)
                                        if preceding_log_data_target['name'] == handler['apply_from']:
                                            break
                if 'offline_aggregates' not in log_data_target:
                    log_data_target['offline_aggregates'] = {
                        'duration' : list()
                    }
                log_data_target['offline_aggregates']['duration'].append(duration_us)
                self.current_transition_in_trace += 1
