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
import sys
import io
import yaml
from automata import PTA
from harness import OnboardTimerHarness

opt = {}

if __name__ == '__main__':

    try:
        optspec = (
            'depth= '
            'instance= '
            'sleep= '
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

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    modelfile = args[0]

    with open(modelfile, 'r') as f:
        if '.json' in modelfile:
            pta = PTA.from_json(json.load(f))
        else:
            pta = PTA.from_yaml(yaml.safe_load(f))

    harness = OnboardTimerHarness('GPIO::p1_0')

    outbuf = io.StringIO()

    outbuf.write('#include "arch.h"\n')
    if 'includes' in pta.codegen:
        for include in pta.codegen['includes']:
            outbuf.write('#include "{}"\n'.format(include))
    outbuf.write(harness.global_code())

    outbuf.write('void loop(void)\n')
    outbuf.write('{\n')

    outbuf.write(harness.start_benchmark())

    class_prefix = ''
    if 'instance' in opt:
        class_prefix = '{}.'.format(opt['instance'])
    elif 'intance' in pta.codegen:
        class_prefix = '{}.'.format(pta.codegen['instance'])

    for run in pta.dfs(opt['depth'], with_arguments = True):
        outbuf.write(harness.start_run())
        for transition, arguments in run:
            outbuf.write('// {} -> {}\n'.format(transition.origin.name, transition.destination.name))
            if transition.is_interrupt:
                outbuf.write('// wait for {} interrupt\n'.format(transition.name))
                transition_code = '// TODO add startTransition / stopTransition calls to interrupt routine'
            else:
                transition_code = '{}{}({});'.format(class_prefix, transition.name, ', '.join(map(str, arguments)))
            outbuf.write(harness.pass_transition(pta.get_transition_id(transition), transition_code))

            if 'sleep' in opt:
                outbuf.write('arch.delay_ms({:d});\n'.format(opt['sleep']))

        outbuf.write(harness.stop_run())
        outbuf.write('\n')

    outbuf.write(harness.stop_benchmark())
    outbuf.write('}\n')
    outbuf.write('int main(void)\n')
    outbuf.write('{\n')
    for driver in ('arch', 'gpio', 'kout'):
        outbuf.write('{}.setup();\n'.format(driver))
    if 'setup' in pta.codegen:
        for call in pta.codegen['setup']:
            outbuf.write(call)
    outbuf.write('arch.idle_loop();\n')
    outbuf.write('return 0;\n')
    outbuf.write('}\n')

    if len(args) > 1:
        with open(args[1], 'w') as f:
            f.write(outbuf.getvalue())
    else:
        print(outbuf.getvalue())
    sys.exit(0)
