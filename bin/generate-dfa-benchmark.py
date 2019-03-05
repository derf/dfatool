#!/usr/bin/env python3

import getopt
import json
import re
import sys
import yaml
from automata import PTA
from harness import TransitionHarness

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

    harness = TransitionHarness('GPIO::p1_0')

    print('#include "arch.h"')
    if pta.header:
        print('#include "{}"'.format(pta.header))
    print(harness.global_code())

    print('void loop(void)')
    print('{')

    print(harness.start_benchmark())

    class_prefix = ''
    if 'instance' in opt:
        class_prefix = '{}.'.format(opt['instance'])
    elif pta.instance:
        class_prefix = '{}.'.format(pta.instance)

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
    print('arch.idle_loop();')
    print('return 0;')
    print('}')
    sys.exit(0)
