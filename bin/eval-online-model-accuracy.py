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
import re
import sys
import itertools
import yaml
from dfatool.automata import PTA
from dfatool.codegen import get_simulated_accountingmethod
from dfatool.model import regression_measures
import numpy as np

opt = dict()

if __name__ == "__main__":

    try:
        optspec = (
            "accounting= "
            "arch= "
            "app= "
            "depth= "
            "dummy= "
            "instance= "
            "repeat= "
            "run= "
            "sleep= "
            "timer-pin= "
            "trace-filter= "
            "timer-freq= "
            "timer-type= "
            "timestamp-type= "
            "energy-type= "
            "power-type= "
            "timestamp-granularity= "
            "energy-granularity= "
            "power-granularity= "
        )
        raw_opts, args = getopt.getopt(sys.argv[1:], "", optspec.split(" "))

        opt_default = {
            "depth": 3,
            "sleep": 0,
            "timer-freq": 1e6,
            "timer-type": "uint16_t",
            "timestamp-type": "uint16_t",
            "energy-type": "uint32_t",
            "power-type": "uint16_t",
            "timestamp-granularity": 1e-6,
            "power-granularity": 1e-6,
            "energy-granularity": 1e-12,
        }

        for option, parameter in raw_opts:
            optname = re.sub(r"^--", "", option)
            opt[optname] = parameter

        for key in "depth sleep".split():
            if key in opt:
                opt[key] = int(opt[key])
            else:
                opt[key] = opt_default[key]

        for (
            key
        ) in "timer-freq timestamp-granularity energy-granularity power-granularity".split():
            if key in opt:
                opt[key] = float(opt[key])
            else:
                opt[key] = opt_default[key]

        for key in "timer-type timestamp-type energy-type power-type".split():
            if key not in opt:
                opt[key] = opt_default[key]

        if "trace-filter" in opt:
            trace_filter = []
            for trace in opt["trace-filter"].split():
                trace_filter.append(trace.split(","))
            opt["trace-filter"] = trace_filter
        else:
            opt["trace-filter"] = None

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    modelfile = args[0]

    pta = PTA.from_file(modelfile)

    enum = dict()
    if ".json" not in modelfile:
        with open(modelfile, "r") as f:
            driver_definition = yaml.safe_load(f)
        if "dummygen" in driver_definition and "enum" in driver_definition["dummygen"]:
            enum = driver_definition["dummygen"]["enum"]

    pta.set_random_energy_model()

    runs = list(
        pta.dfs(
            opt["depth"],
            with_arguments=True,
            with_parameters=True,
            trace_filter=opt["trace-filter"],
            sleep=opt["sleep"],
        )
    )

    num_transitions = len(runs)

    if len(runs) == 0:
        print(
            "DFS returned no traces -- perhaps your trace-filter is too restrictive?",
            file=sys.stderr,
        )
        sys.exit(1)

    real_energies = list()
    real_durations = list()
    model_energies = list()
    for run in runs:
        accounting_method = get_simulated_accountingmethod(opt["accounting"])(
            pta,
            opt["timer-freq"],
            opt["timer-type"],
            opt["timestamp-type"],
            opt["power-type"],
            opt["energy-type"],
        )
        real_energy, real_duration, _, _ = pta.simulate(
            run, accounting=accounting_method
        )
        model_energy = accounting_method.get_energy()
        real_energies.append(real_energy)
        real_durations.append(real_duration)
        model_energies.append(model_energy)

    measures = regression_measures(np.array(model_energies), np.array(real_energies))
    print("SMAPE {:.0f}%, MAE {}".format(measures["smape"], measures["mae"]))

    timer_freqs = [1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6]
    timer_types = (
        timestamp_types
    ) = power_types = energy_types = "uint8_t uint16_t uint32_t uint64_t".split()

    def config_weight(timer_freq, timer_type, ts_type, power_type, energy_type):
        base_weight = 0
        for var_type in timer_type, ts_type, power_type, energy_type:
            if var_type == "uint8_t":
                base_weight += 1
            elif var_type == "uint16_t":
                base_weight += 2
            elif var_type == "uint32_t":
                base_weight += 4
            elif var_type == "uint64_t":
                base_weight += 8
        return base_weight

    # sys.exit(0)

    mean_errors = list()
    for timer_freq, timer_type, ts_type, power_type, energy_type in itertools.product(
        timer_freqs, timer_types, timestamp_types, power_types, energy_types
    ):
        real_energies = list()
        real_durations = list()
        model_energies = list()
        # duration in µs
        # Bei kurzer Dauer (z.B. nur [1e2]) performt auch uint32_t für Energie gut, sonst nicht so (weil overflow)
        for sleep_duration in [1e2, 1e3, 1e4, 1e5, 1e6]:
            runs = pta.dfs(
                opt["depth"],
                with_arguments=True,
                with_parameters=True,
                trace_filter=opt["trace-filter"],
                sleep=sleep_duration,
            )
            for run in runs:
                accounting_method = get_simulated_accountingmethod(opt["accounting"])(
                    pta, timer_freq, timer_type, ts_type, power_type, energy_type
                )
                real_energy, real_duration, _, _ = pta.simulate(
                    run, accounting=accounting_method
                )
                model_energy = accounting_method.get_energy()
                real_energies.append(real_energy)
                real_durations.append(real_duration)
                model_energies.append(model_energy)
        measures = regression_measures(
            np.array(model_energies), np.array(real_energies)
        )
        mean_errors.append(
            (
                (timer_freq, timer_type, ts_type, power_type, energy_type),
                config_weight(timer_freq, timer_type, ts_type, power_type, energy_type),
                measures,
            )
        )

    mean_errors.sort(key=lambda x: x[1])
    mean_errors.sort(key=lambda x: x[2]["mae"])

    for result in mean_errors:
        config, weight, measures = result
        print("{}  -> {:.0f}% / {}".format(config, measures["smape"], measures["mae"]))

    sys.exit(0)
