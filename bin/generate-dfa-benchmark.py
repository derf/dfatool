#!/usr/bin/env python3
"""
Generate a driver/library benchmark based on DFA/PTA traces.

Usage:
PYTHONPATH=lib bin/generate-dfa-benchmark.py [options] <pta/dfa definition> [output.cc]

generate-dfa-benchmarks reads in a DFA definition and generates runs
(i.e., all words accepted by the DFA up to a configurable length). Each symbol
corresponds to a function call. If arguments are specified in the DFA
definition, each symbol corresponds to a function call with a specific set of
arguments (so all argument combinations are present in the generated runs).

Options:
--depth=<depth> (default: 3)
    Maximum number of function calls per run

--instance=<name>
    Override the name of the class instance used for benchmarking

--sleep=<ms> (default: 0)
    How long to sleep between function calls.
"""

import getopt
import json
import re
import runner
import sys
import time
import io
import yaml
from automata import PTA
from harness import OnboardTimerHarness

opt = {}

def trace_matches_filter(trace: list, trace_filter: list) -> bool:
    for allowed_trace in trace_filter:
        if len(trace) < len(allowed_trace):
            continue
        different_element_count = len(list(filter(None, map(lambda x,y: x[0].name != y, trace, allowed_trace))))
        if different_element_count == 0:
            return True
    return False


if __name__ == '__main__':

    try:
        optspec = (
            'arch= '
            'app= '
            'depth= '
            'instance= '
            'run= '
            'sleep= '
            'timer-pin= '
            'trace-filter= '
        )
        raw_opts, args = getopt.getopt(sys.argv[1:], "", optspec.split(' '))

        for option, parameter in raw_opts:
            optname = re.sub(r'^--', '', option)
            opt[optname] = parameter

        if 'depth' in opt:
            opt['depth'] = int(opt['depth'])
        else:
            opt['depth'] = 3

        if 'sleep' in opt:
            opt['sleep'] = int(opt['sleep'])

        if 'trace-filter' in opt:
            trace_filter = []
            for trace in opt['trace-filter'].split():
                trace_filter.append(trace.split(','))
            opt['trace-filter'] = trace_filter

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    modelfile = args[0]

    with open(modelfile, 'r') as f:
        if '.json' in modelfile:
            pta = PTA.from_json(json.load(f))
        else:
            pta = PTA.from_yaml(yaml.safe_load(f))

    if 'timer-pin' in opt:
        timer_pin = opt['timer-pin']
    else:
        timer_pin = 'GPIO::p1_0'

    harness = OnboardTimerHarness(timer_pin)

    outbuf = io.StringIO()

    outbuf.write('#include "arch.h"\n')
    if 'includes' in pta.codegen:
        for include in pta.codegen['includes']:
            outbuf.write('#include "{}"\n'.format(include))
    outbuf.write(harness.global_code())

    outbuf.write('int main(void)\n')
    outbuf.write('{\n')
    for driver in ('arch', 'gpio', 'kout'):
        outbuf.write('{}.setup();\n'.format(driver))
    if 'setup' in pta.codegen:
        for call in pta.codegen['setup']:
            outbuf.write(call)

    outbuf.write('while (1) {\n')
    outbuf.write(harness.start_benchmark())

    class_prefix = ''
    if 'instance' in opt:
        class_prefix = '{}.'.format(opt['instance'])
    elif 'instance' in pta.codegen:
        class_prefix = '{}.'.format(pta.codegen['instance'])

    num_transitions = 0
    num_traces = 0
    for run in pta.dfs(opt['depth'], with_arguments = True, with_parameters = True):
        if 'trace-filter' in opt and not trace_matches_filter(run, opt['trace-filter']):
            continue
        outbuf.write(harness.start_run())
        harness.start_trace()
        param = pta.get_initial_param_dict()
        for transition, arguments, parameter in run:
            num_transitions += 1
            harness.append_state(transition.origin.name, param)
            harness.append_transition(transition.name, param, arguments)
            param = transition.get_params_after_transition(param, arguments)
            outbuf.write('// {} -> {}\n'.format(transition.origin.name, transition.destination.name))
            if transition.is_interrupt:
                outbuf.write('// wait for {} interrupt\n'.format(transition.name))
                transition_code = '// TODO add startTransition / stopTransition calls to interrupt routine'
            else:
                transition_code = '{}{}({});'.format(class_prefix, transition.name, ', '.join(map(str, arguments)))
            outbuf.write(harness.pass_transition(pta.get_transition_id(transition), transition_code, transition = transition, parameter = parameter))

            param = parameter

            if 'sleep' in opt:
                outbuf.write('arch.delay_ms({:d});\n'.format(opt['sleep']))

        outbuf.write(harness.stop_run(num_traces))
        outbuf.write('\n')
        num_traces += 1

    if num_transitions == 0:
        print('DFS returned no traces -- perhaps your trace-filter is too restrictive?', file=sys.stderr)
        sys.exit(1)

    outbuf.write(harness.stop_benchmark())
    outbuf.write('}\n')
    outbuf.write('return 0;\n')
    outbuf.write('}\n')

    if len(args) > 1:
        with open(args[1], 'w') as f:
            f.write(outbuf.getvalue())
    else:
        print(outbuf.getvalue())

    if 'run' in opt:
        if 'sleep' in opt:
            run_timeout = num_transitions * opt['sleep'] / 1000
        else:
            run_timeout = num_transitions * 10 / 1000
        monitor = runner.get_monitor(opt['arch'], callback = harness.parser_cb)
        runner.build(opt['arch'], opt['app'], opt['run'].split())
        runner.flash(opt['arch'], opt['app'], opt['run'].split())
        if opt['arch'] != 'posix':
            try:
                slept = 0
                while True:
                    time.sleep(5)
                    slept += 5
                    print('[MON] approx. {:.0f}% done'.format(slept * 100 / run_timeout))
            except KeyboardInterrupt:
                pass
            lines = monitor.get_lines()
            monitor.close()
        else:
            print('[MON] Will run benchmark for {:.0f} seconds'.format(2 * run_timeout))
            lines = monitor.run(int(2 * run_timeout))
        print(lines)

    sys.exit(0)
