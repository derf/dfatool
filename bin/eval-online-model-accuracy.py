#!/usr/bin/env python3
"""
Evaluate accuracy of online model for DFA/PTA traces.

Usage:
PYTHONPATH=lib bin/eval-online-model-accuracy.py [options] <pta/dfa definition>

Options:
--accounting=static_state|static_state_immediate|static_statetransition|static_statetransition_immedate
    Select accounting method

--depth=<depth> (default: 3)
    Maximum number of function calls per run

--sleep=<ms> (default: 0)
    How long to sleep between simulated function calls.

--trace-filter=<transition,transition,transition,...>[ <transition,transition,transition,...> ...]
    Only consider traces whose beginning matches one of the provided transition sequences.
    E.g. --trace-filter='init,foo init,bar' will only consider traces with init as first and foo or bar as second transition,
    and --trace-filter='init,foo,$ init,bar,$' will only consider the traces init -> foo and init -> bar.
"""

import getopt
import json
import re
import runner
import sys
import time
import io
import yaml
from aspectc import Repo
from automata import PTA
from codegen import *
from harness import OnboardTimerHarness
from dfatool import regression_measures

opt = dict()

if __name__ == '__main__':

    try:
        optspec = (
            'accounting= '
            'arch= '
            'app= '
            'depth= '
            'dummy= '
            'instance= '
            'repeat= '
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
        else:
            opt['sleep'] = 0

        if 'trace-filter' in opt:
            trace_filter = []
            for trace in opt['trace-filter'].split():
                trace_filter.append(trace.split(','))
            opt['trace-filter'] = trace_filter
        else:
            opt['trace-filter'] = None

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    modelfile = args[0]

    with open(modelfile, 'r') as f:
        if '.json' in modelfile:
            pta = PTA.from_json(json.load(f))
        else:
            pta = PTA.from_yaml(yaml.safe_load(f))

    enum = dict()
    if '.json' not in modelfile:
        with open(modelfile, 'r') as f:
            driver_definition = yaml.safe_load(f)
        if 'dummygen' in driver_definition and 'enum' in driver_definition['dummygen']:
            enum = driver_definition['dummygen']['enum']

    repo = Repo('/home/derf/var/projects/multipass/build/repo.acp')

    pta.set_random_energy_model()

    runs = list(pta.dfs(opt['depth'], with_arguments = True, with_parameters = True, trace_filter = opt['trace-filter'], sleep = opt['sleep']))

    num_transitions = len(runs)

    if len(runs) == 0:
        print('DFS returned no traces -- perhaps your trace-filter is too restrictive?', file=sys.stderr)
        sys.exit(1)

    real_energies = list()
    real_durations = list()
    model_energies = list()
    for run in runs:
        accounting_method = get_simulated_accountingmethod(opt['accounting'])(pta, 1e6, 'uint32_t', 'uint32_t', 'uint32_t', 'uint32_t')
        real_energy, real_duration, _, _ = pta.simulate(run, accounting = accounting_method)
        model_energy = accounting_method.get_energy()
        real_energies.append(real_energy)
        real_durations.append(real_duration)
        model_energies.append(model_energy)
        print('actual energy {:.0f} µJ, modeled energy {:.0f} µJ'.format(real_energy / 1e6, model_energy / 1e6))

    measures = regression_measures(np.array(model_energies), np.array(real_energies))
    print('SMAPE {:.0f}%, MAE {}'.format(measures['smape'], measures['mae']))

    sys.exit(0)
