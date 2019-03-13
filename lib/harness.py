"""
Harnesses for various types of benchmark logs.

tbd
"""

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
            'trace' : [{
                'name' : 'UNINITIALIZED',
                'isa' : 'state',
                'parameter' : dict(),
                'offline_aggregates' : list(),
            }]
        })
        self.trace_id += 1

    #def append_state(self):

    #def append_transition(self, ):

    def start_run(self):
        self.start_trace()
        return 'ptalog.reset();\n'

    def pass_transition(self, transition_id, transition_code, parameter = dict()):
        ret = 'ptalog.passTransition({:d});\n'.format(transition_id)
        ret += 'ptalog.startTransition();\n'
        ret += '{}\n'.format(transition_code)
        ret += 'ptalog.stopTransition();\n'
        return ret

    def stop_run(self):
        return 'ptalog.dump();\n'

    def stop_benchmark(self):
        return ''

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

    def pass_transition(self, transition_id, transition_code, parameter = dict()):
        ret = 'ptalog.passTransition({:d});\n'.format(transition_id)
        ret += 'ptalog.startTransition();\n'
        ret += 'counter.start();\n'
        ret += '{}\n'.format(transition_code)
        ret += 'counter.stop();\n'
        ret += 'ptalog.stopTransition(counter);\n'
        return ret