#!/usr/bin/env python3
"""
Generate a driver/library benchmark based on DFA/PTA traces.

Usage:
PYTHONPATH=lib bin/generate-dfa-benchmark.py [options] <pta/dfa definition>

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

    print('#include "arch.h"')
    if 'includes' in pta.codegen:
        for include in pta.codegen['includes']:
            print('#include "{}"'.format(include))
    print(harness.global_code())

    print('void loop(void)')
    print('{')

    print(harness.start_benchmark())

    class_prefix = ''
    if 'instance' in opt:
        class_prefix = '{}.'.format(opt['instance'])
    elif 'intance' in pta.codegen:
        class_prefix = '{}.'.format(pta.codegen['instance'])

    for run in pta.dfs(opt['depth'], with_arguments = True):
        print(harness.start_run())
        for transition, arguments in run:
            print('// {} -> {}'.format(transition.origin.name, transition.destination.name))
            if transition.is_interrupt:
                print('// wait for {} interrupt'.format(transition.name))
                transition_code = '// TODO add startTransition / stopTransition calls to interrupt routine'
            else:
                transition_code = '{}{}({});'.format(class_prefix, transition.name, ', '.join(map(str, arguments)))
            print(harness.pass_transition(pta.get_transition_id(transition), transition_code))

            if 'sleep' in opt:
                print('arch.delay_ms({:d});'.format(opt['sleep']))

        print(harness.stop_run())
        print()

    print(harness.stop_benchmark())
    print('}\n')
    print('int main(void)')
    print('{')
    for driver in ('arch', 'gpio', 'kout'):
        print('{}.setup();'.format(driver))
    if 'setup' in pta.codegen:
        for call in pta.codegen['setup']:
            print(call)
    print('arch.idle_loop();')
    print('return 0;')
    print('}')
    sys.exit(0)
