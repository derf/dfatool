"""
Harnesses for various types of benchmark logs.

tbd
"""
import subprocess
import re

# TODO prepare benchmark log JSON with parameters etc.
# Should be independent of PTA class, as benchmarks may also be
# generated otherwise and it should also work with AnalyticModel (which does
# not have states)
class TransitionHarness:
    def __init__(self, gpio_pin = None, pta = None):
        self.gpio_pin = gpio_pin
        self.pta = pta
        self.reset()

    def reset(self):
        self.traces = []
        self.trace_id = 1
        self.synced = False

    def global_code(self):
        ret = ''
        if self.gpio_pin != None:
            ret += '#define PTALOG_GPIO {}\n'.format(self.gpio_pin)
        ret += '#include "object/ptalog.h"\n'
        if self.gpio_pin != None:
            ret += 'PTALog ptalog({});\n'.format(self.gpio_pin)
        else:
            ret += 'PTALog ptalog;\n'
        return ret

    def start_benchmark(self, benchmark_id = 0):
        return 'ptalog.startBenchmark({:d});\n'.format(benchmark_id)

    def start_trace(self):
        self.traces.append({
            'id' : self.trace_id,
            'trace' : list(),
        })
        self.trace_id += 1

    def append_state(self, state_name, param):
        self.traces[-1]['trace'].append({
            'name': state_name,
            'isa': 'state',
            'parameter': param,
        })

    def append_transition(self, transition_name, param, args = []):
        self.traces[-1]['trace'].append({
            'name': transition_name,
            'isa': 'transition',
            'parameter': param,
            'args' : args,
        })

    def start_run(self):
        return 'ptalog.reset();\n'

    def pass_transition(self, transition_id, transition_code, transition: object = None):
        ret = 'ptalog.passTransition({:d});\n'.format(transition_id)
        ret += 'ptalog.startTransition();\n'
        ret += '{}\n'.format(transition_code)
        ret += 'ptalog.stopTransition();\n'
        return ret

    def stop_run(self, num_traces = 0):
        return 'ptalog.dump({:d});\n'.format(num_traces)

    def stop_benchmark(self):
        return ''

    def parser_cb(self, line):
        pass

    def parse_log(self, lines):
        sync = False
        for line in lines:
            print(line)
            res = re.fullmatch(r'\[PTA\] (.*=.*)', line)
            if re.fullmatch(r'\[PTA\] benchmark start, id=(.*)', line):
                print('> got sync')
                sync = True
            elif not sync:
                continue
            elif re.fullmatch(r'\[PTA\] trace, count=(.*)', line):
                print('> got transition')
                pass
            elif res:
                print(dict(map(lambda x: x.split('='), res.group(1).split())))
                pass

class OnboardTimerHarness(TransitionHarness):
    def __init__(self, counter_limits, **kwargs):
        super().__init__(**kwargs)
        self.trace_id = 0
        self.trace_length = 0
        self.one_cycle_in_us, self.one_overflow_in_us, self.counter_max_overflow = counter_limits

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
        ret += 'ptalog.startTransition();\n'
        ret += 'counter.start();\n'
        ret += '{}\n'.format(transition_code)
        ret += 'counter.stop();\n'
        ret += 'ptalog.stopTransition(counter);\n'
        return ret

    def parser_cb(self, line):
        #print('[HARNESS] got line {}'.format(line))
        if re.match(r'\[PTA\] benchmark start, id=(.*)', line):
            self.synced = True
            print('[HARNESS] synced')
        if self.synced:
            res = re.match(r'\[PTA\] trace=(.*) count=(.*)', line)
            if res:
                self.trace_id = int(res.group(1))
                self.trace_length = int(res.group(2))
                self.current_transition_in_trace = 0
                #print('[HARNESS] trace {:d} contains {:d} transitions. Expecting {:d} transitions.'.format(self.trace_id, self.trace_length, len(self.traces[self.trace_id]['trace']) // 2))
            res = re.match(r'\[PTA\] transition=(.*) cycles=(.*)/(.*)', line)
            if res:
                transition_id = int(res.group(1))
                # TODO Handle Overflows (requires knowledge of arch-specific max cycle value)
                cycles = int(res.group(2))
                overflow = int(res.group(3))
                if overflow >= self.counter_max_overflow:
                    raise RuntimeError('Counter overflow ({:d}/{:d}) in benchmark id={:d} trace={:d}: transition #{:d} (ID {:d})'.format(cycles, overflow, 0, self.trace_id, self.current_transition_in_trace, transition_id))
                duration_us = cycles * self.one_cycle_in_us + overflow * self.one_overflow_in_us
                # self.traces contains transitions and states, UART output only contains trnasitions -> use index * 2
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
                    raise RuntimeError('Log mismatch: Expected transition, got {:s}'.format(log_data_target['isa']))
                if self.pta:
                    transition = self.pta.transitions[transition_id]
                    if transition.name != log_data_target['name']:
                        raise RuntimeError('Log mismatch: Expected transition {:s}, got transition {:s}'.format(log_data_target['name'], transition.name))
                #print('[HARNESS] Logging data for transition {}'.format(log_data_target['name']))
                if 'offline_aggregates' not in log_data_target:
                    log_data_target['offline_aggregates'] = {
                        'duration' : list()
                    }
                log_data_target['offline_aggregates']['duration'].append(duration_us)
                self.current_transition_in_trace += 1