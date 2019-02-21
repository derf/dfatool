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
        )
        raw_opts, args = getopt.getopt(sys.argv[1:], "", optspec.split(' '))

        for option, parameter in raw_opts:
            optname = re.sub(r'^--', '', option)
            opt[optname] = parameter

        if 'depth' in opt:
            opt['depth'] = int(opt['depth'])
        else:
            opt['depth'] = 3

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    modelfile = args[0]

    with open(modelfile, 'r') as f:
        pta = PTA.from_json(json.load(f))

    for run in pta.dfs(opt['depth'], with_arguments = True):
        for function_name, arguments in run:
            if 'instance' in opt:
                print('{}.{}({});'.format(opt['instance'], function_name, ', '.join(arguments)))
            else:
                print('{}({});'.format(function_name, ', '.join(arguments)))
        print()

    sys.exit(0)
