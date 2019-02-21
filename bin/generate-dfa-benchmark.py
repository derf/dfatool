#!/usr/bin/env python3

import getopt
import json
import re
import sys
from automata import PTA

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
        pta = PTA.from_json(json.load(f))

    for run in pta.dfs(opt['depth'], with_arguments = True):
        for transition, arguments in run:
            if transition.is_interrupt:
                print('// wait for {} interrupt'.format(transition.name))
            else:
                if 'instance' in opt:
                    print('{}.{}({});'.format(opt['instance'], transition.name, ', '.join(arguments)))
                else:
                    print('{}({});'.format(transition.name, ', '.join(arguments)))

            if 'sleep' in opt:
                print('arch.delay_ms({:d});'.format(opt['sleep']))

        print()

    sys.exit(0)
