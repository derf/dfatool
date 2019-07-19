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
    def __init__(self, gpio_pin = None):
        self.gpio_pin = gpio_pin
        self.traces = []
        self.trace_id = 1
        pass

    def start_benchmark(self):
        pass

    def stop_benchmark(self):
        pass

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

    def start_benchmark(self):
        return 'ptalog.startBenchmark(0);\n'

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

    def pass_transition(self, transition_id, transition_code, transition: object = None, parameter: dict = dict()):
        ret = 'ptalog.passTransition({:d});\n'.format(transition_id)
        ret += 'ptalog.startTransition();\n'
        ret += '{}\n'.format(transition_code)
        ret += 'ptalog.stopTransition();\n'
        return ret

    def stop_run(self, trace_id = 0):
        return 'ptalog.dump({:d});\n'.format(trace_id)

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
    def __init__(self, gpio_pin = None):
        super().__init__(gpio_pin = gpio_pin)

    def global_code(self):
        ret = '#include "driver/counter.h"\n'
        ret += '#define PTALOG_TIMING\n'
        ret += super().global_code()
        return ret

    def start_benchmark(self):
        ret = 'counter.start();\n'
        ret += 'counter.stop();\n'
        ret += 'ptalog.passNop(counter);\n'
        ret += super().start_benchmark()
        return ret

    def pass_transition(self, transition_id, transition_code, transition: object = None, parameter: dict = dict()):
        ret = 'ptalog.passTransition({:d});\n'.format(transition_id)
        ret += 'ptalog.startTransition();\n'
        ret += 'counter.start();\n'
        ret += '{}\n'.format(transition_code)
        ret += 'counter.stop();\n'
        ret += 'ptalog.stopTransition(counter);\n'
        return ret