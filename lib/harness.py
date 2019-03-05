"""
Harnesses for various types of benchmark logs.

tbd
"""

class OnboardTimerHarness:
    def __init__(self, gpio_pin = None):
        self.gpio_pin = gpio_pin
        pass

    def global_code(self):
        ret = '#include "driver/counter.h"\n'
        ret += '#define PTALOG_TIMING\n'
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

    def start_run(self):
        return 'ptalog.reset();\n'

    def pass_transition(self, transition_id, transition_code):
        ret = 'ptalog.passTransition({:d});\n'.format(transition_id)
        ret += 'ptalog.startTransition();\n'
        ret += 'counter.start();\n'
        ret += '{}\n'.format(transition_code)
        ret += 'counter.stop();\n'
        ret += 'ptalog.stopTransition(counter);\n'
        return ret

    def stop_run(self):
        return 'ptalog.dump();\n'

    def stop_benchmark(self):
        return ''

class TransitionHarness:
    def __init__(self, gpio_pin = None):
        self.gpio_pin = gpio_pin
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

    def start_run(self):
        return 'ptalog.reset();\n'

    def pass_transition(self, transition_id, transition_code):
        ret = 'ptalog.passTransition({:d});\n'.format(transition_id)
        ret += 'ptalog.startTransition();\n'
        ret += '{}\n'.format(transition_code)
        ret += 'ptalog.stopTransition();\n'
        return ret

    def stop_run(self):
        return 'ptalog.dump();\n'

    def stop_benchmark(self):
        return ''
