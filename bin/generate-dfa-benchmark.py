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
--accounting=static_state|static_state_immediate|static_statetransition|static_statetransition_immedate
    Select accounting method for dummy driver generation

--dummy=<class name>
    Generate and use a dummy driver for online energy model overhead evaluation

--depth=<depth> (default: 3)
    Maximum number of function calls per run

--repeat=<count> (default: 0)
    Repeat benchmark runs <count> times. When 0, benchmark runs are repeated
    indefinitely and must be explicitly terminated with Ctrl+C / SIGINT

--instance=<name>
    Override the name of the class instance used for benchmarking

--sleep=<ms> (default: 0)
    How long to sleep between function calls.

--shrink
    Decrease amount of parameter values used in state space exploration
    (only use minimum and maximum for numeric values)

--trace-filter=<transition,transition,transition,...>[ <transition,transition,transition,...> ...]
    Only consider traces whose beginning matches one of the provided transition sequences.
    E.g. --trace-filter='init,foo init,bar' will only consider traces with init as first and foo or bar as second transition,
    and --trace-filter='init,foo,$ init,bar,$' will only consider the traces init -> foo and init -> bar.
        
"""

import getopt
import json
import os
import re
import runner
import sys
import tarfile
import time
import io
import yaml
from aspectc import Repo
from automata import PTA
from codegen import *
from harness import OnboardTimerHarness, TransitionHarness
from utils import flatten

opt = dict()

def benchmark_from_runs(pta: PTA, runs: list, harness: OnboardTimerHarness, benchmark_id: int = 0, dummy = False, repeat = 0) -> io.StringIO:
    outbuf = io.StringIO()

    outbuf.write('#include "arch.h"\n')
    if dummy:
        outbuf.write('#include "driver/dummy.h"\n')
    elif 'includes' in pta.codegen:
        for include in pta.codegen['includes']:
            outbuf.write('#include "{}"\n'.format(include))
    outbuf.write(harness.global_code())

    outbuf.write('int main(void)\n')
    outbuf.write('{\n')

    for driver in ('arch', 'gpio', 'kout'):
        outbuf.write('{}.setup();\n'.format(driver))

    # There is a race condition between flashing the code and starting the UART log.
    # When starting the log before flashing, output from a previous benchmark may cause bogus data to be added.
    # When flashing first and then starting the log, the first log lines may be lost.
    # To work around this, we flash first, then start the log, and use this delay statement to ensure that no output is lost.
    # This is also useful to faciliate MIMOSA calibration after flashing
    outbuf.write('arch.delay_ms(12000);\n')

    if 'setup' in pta.codegen:
        for call in pta.codegen['setup']:
            outbuf.write(call)


    if repeat:
        outbuf.write('unsigned char i = 0;\n')
        outbuf.write('while (i++ < {}) {{\n'.format(repeat))
    else:
        outbuf.write('while (1) {\n')

    outbuf.write(harness.start_benchmark())

    class_prefix = ''
    if 'instance' in opt:
        class_prefix = '{}.'.format(opt['instance'])
    elif 'instance' in pta.codegen:
        class_prefix = '{}.'.format(pta.codegen['instance'])

    num_transitions = 0
    num_traces = 0
    for run in runs:
        outbuf.write(harness.start_run())
        harness.start_trace()
        param = pta.get_initial_param_dict()
        for transition, arguments, parameter in run:
            num_transitions += 1
            harness.append_transition(transition.name, param, arguments)
            harness.append_state(transition.destination.name, parameter.copy())
            outbuf.write('// {} -> {}\n'.format(transition.origin.name, transition.destination.name))
            if transition.is_interrupt:
                outbuf.write('// wait for {} interrupt\n'.format(transition.name))
                transition_code = '// TODO add startTransition / stopTransition calls to interrupt routine'
            else:
                transition_code = '{}{}({});'.format(class_prefix, transition.name, ', '.join(map(str, arguments)))
            outbuf.write(harness.pass_transition(pta.get_transition_id(transition), transition_code, transition = transition))

            param = parameter

            outbuf.write('// current parameters: {}\n'.format(', '.join(map(lambda kv: '{}={}'.format(*kv), param.items()))))

            if opt['sleep']:
                outbuf.write('arch.delay_ms({:d}); // {}\n'.format(opt['sleep'], transition.destination.name))

        outbuf.write(harness.stop_run(num_traces))
        if dummy:
            outbuf.write('kout << "[Energy] " << {}getEnergy() << endl;\n'.format(class_prefix))
        outbuf.write('\n')
        num_traces += 1

    outbuf.write(harness.stop_benchmark())
    outbuf.write('}\n')

    # Ensure logging can be terminated after the specified number of measurements
    outbuf.write(harness.start_benchmark())

    outbuf.write('while(1) { }\n')
    outbuf.write('return 0;\n')
    outbuf.write('}\n')

    return outbuf

def run_benchmark(application_file: str, pta: PTA, runs: list, arch: str, app: str, run_args: list, harness: object, sleep: int = 0, repeat: int = 0, run_offset: int = 0, runs_total: int = 0, dummy = False):
    outbuf = benchmark_from_runs(pta, runs, harness, dummy = dummy, repeat = repeat)
    with open(application_file, 'w') as f:
        f.write(outbuf.getvalue())
        print('[MAKE] building benchmark with {:d} runs'.format(len(runs)))

    # assume an average of 10ms per transition. Mind the 10s start delay.
    run_timeout = 10 + num_transitions * (sleep+10) / 1000

    if repeat:
        run_timeout *= repeat

    needs_split = False
    try:
        runner.build(arch, app, run_args)
    except RuntimeError:
        if len(runs) > 50:
            # Application is too large -> split up runs
            needs_split = True
        else:
            # Unknown error
            raise

    # This has been deliberately taken out of the except clause to avoid nested exception handlers
    # (they lead to pretty interesting tracebacks which are probably more confusing than helpful)
    if needs_split:
        print('[MAKE] benchmark code is too large, splitting up')
        mid = len(runs) // 2
        # Previously prepared trace data is useless
        harness.reset()
        results = run_benchmark(application_file, pta, runs[:mid], arch, app, run_args, harness.copy(), sleep, repeat, run_offset = run_offset, runs_total = runs_total, dummy = dummy)
        results.extend(run_benchmark(application_file, pta, runs[mid:], arch, app, run_args, harness.copy(), sleep, repeat, run_offset = run_offset + mid, runs_total = runs_total, dummy = dummy))
        return results

    runner.flash(arch, app, run_args)
    monitor = runner.get_monitor(arch, callback = harness.parser_cb, mimosa = {'shunt': 82})

    if arch == 'posix':
        print('[RUN] Will run benchmark for {:.0f} seconds'.format(run_timeout))
        lines = monitor.run(int(run_timeout))
        return [(runs, harness, lines)]

    try:
        slept = 0
        while repeat == 0 or not harness.done:
            time.sleep(5)
            slept += 5
            print('[RUN] {:d}/{:d} ({:.0f}%), current benchmark at {:.0f}%'.format(run_offset, runs_total, run_offset * 100 / runs_total, slept * 100 / run_timeout))
    except KeyboardInterrupt:
        pass
    monitor.close()

    return [(runs, harness, monitor)]


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
            'shrink '
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

        if 'repeat' in opt:
            opt['repeat'] = int(opt['repeat'])
        else:
            opt['repeat'] = 0

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

    if 'shrink' in opt:
        pta.shrink_argument_values()

    if 'timer-pin' in opt:
        timer_pin = opt['timer-pin']
    else:
        timer_pin = 'GPIO::p1_0'

    if 'dummy' in opt:

        enum = dict()
        if '.json' not in modelfile:
            with open(modelfile, 'r') as f:
                driver_definition = yaml.safe_load(f)
            if 'dummygen' in driver_definition and 'enum' in driver_definition['dummygen']:
                enum = driver_definition['dummygen']['enum']

        repo = Repo('/home/derf/var/projects/multipass/build/repo.acp')

        pta.set_random_energy_model()

        if 'accounting' in opt:
            accounting_object = get_accountingmethod(opt['accounting'])(opt['dummy'], pta)
        else:
            accounting_object = None
        drv = MultipassDriver(opt['dummy'], pta, repo.class_by_name[opt['dummy']], enum=enum, accounting=accounting_object)
        with open('/home/derf/var/projects/multipass/src/driver/dummy.cc', 'w') as f:
            f.write(drv.impl)
        with open('/home/derf/var/projects/multipass/include/driver/dummy.h', 'w') as f:
            f.write(drv.header)

    runs = list(pta.dfs(opt['depth'], with_arguments = True, with_parameters = True, trace_filter = opt['trace-filter']))

    num_transitions = len(runs)

    if len(runs) == 0:
        print('DFS returned no traces -- perhaps your trace-filter is too restrictive?', file=sys.stderr)
        sys.exit(1)

    need_return_values = False
    if next(filter(lambda x: len(x.return_value_handlers), pta.transitions), None):
        need_return_values = True

    harness = OnboardTimerHarness(gpio_pin = timer_pin, pta = pta, counter_limits = runner.get_counter_limits_us(opt['arch']), log_return_values = need_return_values, repeat = opt['repeat'])
    harness = TransitionHarness(gpio_pin = timer_pin, pta = pta, log_return_values = need_return_values, repeat = opt['repeat'], post_transition_delay_us = 20)

    if len(args) > 1:
        results = run_benchmark(args[1], pta, runs, opt['arch'], opt['app'], opt['run'].split(), harness, opt['sleep'], opt['repeat'], runs_total = len(runs), dummy = 'dummy' in opt)
        json_out = {
            'opt' : opt,
            'pta' : pta.to_json(),
            'traces' : list(map(lambda x: x[1].traces, results)),
            'raw_output' : list(map(lambda x: x[2].get_lines(), results)),
            'files' : list(map(lambda x: x[2].get_files(), results)),
            'configs' : list(map(lambda x: x[2].get_config(), results)),
        }
        extra_files = flatten(json_out['files'])
        if 'instance' in pta.codegen:
            output_prefix = time.strftime('/home/derf/var/ess/aemr/data/%Y%m%d-%H%M%S-') + pta.codegen['instance']
        else:
            output_prefix = time.strftime('/home/derf/var/ess/aemr/data/%Y%m%d-%H%M%S-ptalog')
        if len(extra_files):
            with open('ptalog.json', 'w') as f:
                json.dump(json_out, f)
            with tarfile.open('{}.tar'.format(output_prefix), 'w') as tar:
                tar.add('ptalog.json')
                for extra_file in extra_files:
                    tar.add(extra_file)
            print(' --> {}.tar'.format(output_prefix))
            os.remove('ptalog.json')
            for extra_file in extra_files:
                os.remove(extra_file)
        else:
            with open('{}.json'.format(output_prefix), 'w') as f:
                json.dump(json_out, f)
            print(' --> {}.json'.format(output_prefix))
    else:
        outbuf = benchmark_from_runs(pta, runs, harness, repeat = repeat)
        print(outbuf.getvalue())

    sys.exit(0)
