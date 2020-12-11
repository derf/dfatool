#!/usr/bin/env python3
"""
Generate a driver/library benchmark based on DFA/PTA traces.

Usage:
bin/generate-dfa-benchmark.py [options] <pta/dfa definition> [output.cc]

generate-dfa-benchmarks reads in a DFA definition and generates runs
(i.e., all words accepted by the DFA up to a configurable length). Each symbol
corresponds to a function call. If arguments are specified in the DFA
definition, each symbol corresponds to a function call with a specific set of
arguments (so all argument combinations are present in the generated runs).

It expects to be called from a multipass instance and writes data to ../data.
Recommended setup:

> mkdir -p data/cache
> git clone .../dfatool
> git clone .../multipass
> cd multipass
> ../dfatool/bin/generate-dfa-benchmark.py ... src/app/aemr/main.cc

Options:
--accounting=static_state|static_state_immediate|static_statetransition|static_statetransition_immedate[,opt1=val1,opt2=val2,...]
    Select accounting method for dummy driver generation.
    May be followed by a list of key=value driver options, e.g. energy_type=uint64_t

--data=<data path>
    Directory in which the measurements will be stored. Default: ../data

--dummy=<class name>
    Generate and use a dummy driver for online energy model overhead evaluation

--depth=<depth> (default: 3)
    Maximum number of function calls per run

--repeat=<count> (default: 0)
    Repeat benchmark runs <count> times. When 0, benchmark runs are repeated
    indefinitely and must be explicitly terminated with Ctrl+C / SIGINT

--instance=<name>
    Override the name of the class instance used for benchmarking

--mimosa=[k=v,k=v,...]
    Perform energy measurements with MIMOSA. Takes precedence over --timing and --energytrace.
    mimosa options are key-value pairs. Possible settings with defaults:
    offset = 130 (mysterious 0V offset)
    shunt = 330 (measurement shunt in ohms)
    voltage = 3.3 (VCC provided to DUT)

--sleep=<ms> (default: 0)
    How long to sleep between function calls.

--shrink
    Decrease amount of parameter values used in state space exploration
    (only use minimum and maximum for numeric values)

--timing
    Perform timing measurements using on-chip counters (no external hardware
    required)

--energytrace=[k=v,k=v,...]
    Perform energy measurements using MSP430 EnergyTrace hardware. Includes --timing.
    Additional configuration settings:
    sync = bar (Barcode mode (default): synchronize measurements via barcodes embedded in the energy trace)
    sync = la (Logic Analyzer mode (WIP): An external logic analyzer captures transition timing)
    sync = timing (Timing mode (WIP): The on-board cycle counter captures transition timing)

--trace-filter=<transition,transition,transition,...>[ <transition,transition,transition,...> ...]
    Only consider traces whose beginning matches one of the provided transition sequences.
    E.g. --trace-filter='init,foo init,bar' will only consider traces with init as first and foo or bar as second transition,
    and --trace-filter='init,foo,$ init,bar,$' will only consider the traces init -> foo and init -> bar.

EXAMPLES

Perform timing measurements of nRF24L01+ function calls:

../dfatool/bin/generate-dfa-benchmark.py --timer-pin=GPIO::p1_0 --sleep=200 --repeat=3 --depth=10 --arch=msp430fr5994lp --timing --trace-filter='setup,setAutoAck,setDataRate,setPALevel,write,$' model/driver/nrf24l01.dfa src/app/aemr/main.cc

Perform timing measurements of BME680 funtion calls:

../dfatool/bin/generate-dfa-benchmark.py --timer-pin=GPIO::p1_0 --sleep=200 --repeat=3 --depth=10 --arch=msp430fr5994lp --timing --trace-filter='init,configure,setSensorSettings,setPowerMode,setSensorMode,getSensorData,$' --shrink model/driver/bme680.dfa src/app/aemr/main.cc

"""

import getopt
import json
import logging
import os
import re
import sys
import tarfile
import time
import io
import yaml
from dfatool import runner
from dfatool.aspectc import Repo
from dfatool.automata import PTA
from dfatool.codegen import get_accountingmethod, MultipassDriver
from dfatool.harness import OnboardTimerHarness, TransitionHarness
from dfatool.utils import flatten

opt = dict()


def benchmark_from_runs(
    pta: PTA,
    runs: list,
    harness: OnboardTimerHarness,
    benchmark_id: int = 0,
    dummy=False,
    repeat=0,
) -> io.StringIO:
    outbuf = io.StringIO()

    outbuf.write('#include "arch.h"\n')
    outbuf.write('#include "driver/gpio.h"\n')
    if dummy:
        outbuf.write('#include "driver/dummy.h"\n')
    elif "includes" in pta.codegen:
        for include in pta.codegen["includes"]:
            outbuf.write('#include "{}"\n'.format(include))
    outbuf.write(harness.global_code())

    outbuf.write("int main(void)\n")
    outbuf.write("{\n")

    for driver in ("arch", "gpio", "kout"):
        outbuf.write("{}.setup();\n".format(driver))

    # There is a race condition between flashing the code and starting the UART log.
    # When starting the log before flashing, output from a previous benchmark may cause bogus data to be added.
    # When flashing first and then starting the log, the first log lines may be lost.
    # To work around this, we flash first, then start the log, and use this delay statement to ensure that no output is lost.
    # This is also useful to faciliate MIMOSA calibration after flashing
    # For MIMOSA, the DUT is disconnected from power during calibration, so
    # it must be set up after the calibration delay.
    # For energytrace, the device is connected to VCC and set up before
    # the initialization delay to -- this puts it into a well-defined state and
    # decreases pre-sync power consumption
    if "energytrace" not in opt:
        if "mimosa" in opt:
            outbuf.write("arch.delay_ms(12000);\n")
        else:
            outbuf.write("arch.delay_ms(2000);\n")
        # Output some newlines to ensure the parser can determine the start of the first real output line
        outbuf.write("kout << endl << endl;\n")

    if "setup" in pta.codegen:
        for call in pta.codegen["setup"]:
            outbuf.write(call)

    if "energytrace" in opt:
        outbuf.write("for (unsigned char i = 0; i < 10; i++) {\n")
        outbuf.write("arch.sleep_ms(250);\n}\n")
        # Output some newlines to ensure the parser can determine the start of the first real output line
        outbuf.write("kout << endl << endl;\n")

    if repeat:
        outbuf.write("unsigned char i = 0;\n")
        outbuf.write("while (i++ < {}) {{\n".format(repeat))
    else:
        outbuf.write("while (1) {\n")

    outbuf.write(harness.start_benchmark())

    class_prefix = ""
    if "instance" in opt:
        class_prefix = "{}.".format(opt["instance"])
    elif "instance" in pta.codegen:
        class_prefix = "{}.".format(pta.codegen["instance"])

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
            outbuf.write(
                "// {} -> {}\n".format(
                    transition.origin.name, transition.destination.name
                )
            )
            if transition.is_interrupt:
                outbuf.write("// wait for {} interrupt\n".format(transition.name))
                transition_code = "// TODO add startTransition / stopTransition calls to interrupt routine"
            else:
                transition_code = "{}{}({});".format(
                    class_prefix, transition.name, ", ".join(map(str, arguments))
                )
            outbuf.write(
                harness.pass_transition(
                    pta.get_transition_id(transition),
                    transition_code,
                    transition=transition,
                )
            )

            param = parameter

            outbuf.write(
                "// current parameters: {}\n".format(
                    ", ".join(map(lambda kv: "{}={}".format(*kv), param.items()))
                )
            )

            if "delay_after_ms" in transition.codegen:
                if "energytrace" in opt:
                    outbuf.write(
                        "arch.sleep_ms({:d}); // {} -- delay mandated by codegen.delay_after_ms\n".format(
                            transition.codegen["delay_after_ms"],
                            transition.destination.name,
                        )
                    )
                else:
                    outbuf.write(
                        "arch.delay_ms({:d}); // {} -- delay mandated by codegen.delay_after_ms\n".format(
                            transition.codegen["delay_after_ms"],
                            transition.destination.name,
                        )
                    )
            elif opt["sleep"]:
                if "energytrace" in opt:
                    outbuf.write(f"// -> {transition.destination.name}\n")
                    outbuf.write(target.sleep_ms(opt["sleep"]))
                else:
                    outbuf.write(f"// -> {transition.destination.name}\n")
                    outbuf.write("arch.delay_ms({:d});\n".format(opt["sleep"]))

        outbuf.write(harness.stop_run(num_traces))
        if dummy:
            outbuf.write(
                'kout << "[Energy] " << {}getEnergy() << endl;\n'.format(class_prefix)
            )
        outbuf.write("\n")
        num_traces += 1

    outbuf.write(harness.stop_benchmark())
    outbuf.write("}\n")

    # Ensure logging can be terminated after the specified number of measurements
    outbuf.write(harness.start_benchmark())

    outbuf.write("while(1) { }\n")
    outbuf.write("return 0;\n")
    outbuf.write("}\n")

    return outbuf


def run_benchmark(
    application_file: str,
    pta: PTA,
    runs: list,
    arch: str,
    app: str,
    run_args: list,
    harness: object,
    sleep: int = 0,
    repeat: int = 0,
    run_offset: int = 0,
    runs_total: int = 0,
    dummy=False,
):
    if "mimosa" in opt or "energytrace" in opt:
        outbuf = benchmark_from_runs(pta, runs, harness, dummy=dummy, repeat=1)
    else:
        outbuf = benchmark_from_runs(pta, runs, harness, dummy=dummy, repeat=repeat)
    with open(application_file, "w") as f:
        f.write(outbuf.getvalue())
        print("[MAKE] building benchmark with {:d} runs".format(len(runs)))

    # assume an average of 10ms per transition. Mind the 10s start delay.
    run_timeout = 10 + num_transitions * (sleep + 10) / 1000

    if repeat:
        run_timeout *= repeat

    needs_split = False
    if len(runs) > 1000:
        needs_split = True
    else:
        try:
            target.build(app, run_args)
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
        print("[MAKE] benchmark code is too large, splitting up")
        mid = len(runs) // 2
        # Previously prepared trace data is useless
        harness.reset()
        results = run_benchmark(
            application_file,
            pta,
            runs[:mid],
            arch,
            app,
            run_args,
            harness.copy(),
            sleep,
            repeat,
            run_offset=run_offset,
            runs_total=runs_total,
            dummy=dummy,
        )
        results.extend(
            run_benchmark(
                application_file,
                pta,
                runs[mid:],
                arch,
                app,
                run_args,
                harness.copy(),
                sleep,
                repeat,
                run_offset=run_offset + mid,
                runs_total=runs_total,
                dummy=dummy,
            )
        )
        return results

    if "mimosa" in opt or "energytrace" in opt:
        files = list()
        i = 0
        while i < opt["repeat"]:
            print(f"""[RUN] flashing benchmark {i+1}/{opt["repeat"]}""")
            target.flash(app, run_args)
            if "mimosa" in opt:
                monitor = target.get_monitor(
                    callback=harness.parser_cb, mimosa=opt["mimosa"]
                )
            elif "energytrace" in opt:
                monitor = target.get_monitor(
                    callback=harness.parser_cb, energytrace=opt["energytrace"]
                )

            sync_error = False
            try:
                slept = 0
                while not harness.done:
                    # possible race condition: if the benchmark completes at this
                    # exact point, it sets harness.done and unsets harness.synced.
                    if (
                        slept > 30
                        and slept < 40
                        and not harness.synced
                        and not harness.done
                    ):
                        print(
                            "[RUN] has been unsynced for more than 30 seconds, assuming error. Retrying."
                        )
                        sync_error = True
                        break
                    if harness.abort:
                        print("[RUN] harness encountered an error. Retrying")
                        sync_error = True
                        break
                    time.sleep(5)
                    slept += 5
                    print(
                        "[RUN] {:d}/{:d} ({:.0f}%) at trace {:d}".format(
                            run_offset,
                            runs_total,
                            run_offset * 100 / runs_total,
                            harness.trace_id,
                        )
                    )
            except KeyboardInterrupt:
                pass

            monitor.close()

            if sync_error:
                for filename in monitor.get_files():
                    os.remove(filename)
                harness.undo(i)
            else:
                files.append(monitor.get_files())
                i += 1

            harness.restart()

        return [(runs, harness, monitor, files)]
    else:
        target.flash(app, run_args)
        monitor = target.get_monitor(callback=harness.parser_cb)

        if arch == "posix":
            print("[RUN] Will run benchmark for {:.0f} seconds".format(run_timeout))
            lines = monitor.run(int(run_timeout))
            return [(runs, harness, lines, list())]

        try:
            slept = 0
            while not harness.done:
                time.sleep(5)
                slept += 5
                print(
                    "[RUN] {:d}/{:d} ({:.0f}%), current benchmark at {:.0f}%".format(
                        run_offset,
                        runs_total,
                        run_offset * 100 / runs_total,
                        slept * 100 / run_timeout,
                    )
                )
        except KeyboardInterrupt:
            pass
        monitor.close()

        return [(runs, harness, monitor, list())]


if __name__ == "__main__":

    try:
        optspec = (
            "accounting= "
            "arch= "
            "arch-flags= "
            "app= "
            "data= "
            "depth= "
            "dummy= "
            "energytrace= "
            "instance= "
            "log-level= "
            "mimosa= "
            "repeat= "
            "run= "
            "sleep= "
            "shrink "
            "timing "
            "timer-pin= "
            "trace-filter= "
        )
        raw_opts, args = getopt.getopt(sys.argv[1:], "", optspec.split(" "))

        for option, parameter in raw_opts:
            optname = re.sub(r"^--", "", option)
            opt[optname] = parameter

        if "app" not in opt:
            opt["app"] = "aemr"

        if "arch-flags" in opt:
            opt["arch-flags"] = opt["arch-flags"].split(",")
        else:
            opt["arch-flags"] = list()

        if "depth" in opt:
            opt["depth"] = int(opt["depth"])
        else:
            opt["depth"] = 3

        if "repeat" in opt:
            opt["repeat"] = int(opt["repeat"])
        else:
            opt["repeat"] = 0

        if "sleep" in opt:
            opt["sleep"] = int(opt["sleep"])
        else:
            opt["sleep"] = 0

        if "trace-filter" in opt:
            trace_filter = list()
            for trace in opt["trace-filter"].split():
                trace_filter.append(trace.split(","))
            opt["trace-filter"] = trace_filter
        else:
            opt["trace-filter"] = None

        if "log-level" in opt:
            numeric_level = getattr(logging, opt["log-level"].upper(), None)
            if not isinstance(numeric_level, int):
                print(f"Invalid log level: {args.log_level}", file=sys.stderr)
                sys.exit(1)
            logging.basicConfig(level=numeric_level)

        if "mimosa" in opt:
            if opt["mimosa"] == "":
                opt["mimosa"] = dict()
            else:
                opt["mimosa"] = dict(
                    map(lambda x: x.split("="), opt["mimosa"].split(","))
                )
            opt.pop("timing", None)
            if opt["repeat"] == 0:
                opt["repeat"] = 1

        if "energytrace" in opt:
            if opt["energytrace"] == "":
                opt["energytrace"] = dict()
            else:
                opt["energytrace"] = dict(
                    map(lambda x: x.split("="), opt["energytrace"].split(","))
                )
            opt.pop("timing", None)
            if opt["repeat"] == 0:
                opt["repeat"] = 1

        if "data" not in opt:
            opt["data"] = "../data"

        if "dummy" in opt:
            if opt["dummy"] == "":
                opt["dummy"] = dict()
            else:
                opt["dummy"] = dict(
                    map(lambda x: x.split("="), opt["dummy"].split(","))
                )

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    if "msp430fr" in opt["arch"]:
        if len(opt["arch-flags"]) == 0:
            opt["arch-flags"] = ["cpu-freq=8000000"]
        target = runner.Arch(opt["arch"], opt["arch-flags"])
    else:
        target = runner.Arch(opt["arch"])

    modelfile = args[0]

    pta = PTA.from_file(modelfile)
    run_flags = None

    if "shrink" in opt:
        pta.shrink_argument_values()

    if "timer-pin" in opt:
        timer_pin = opt["timer-pin"]
    else:
        timer_pin = None

    if "dummy" in opt:

        enum = dict()
        if ".json" not in modelfile:
            with open(modelfile, "r") as f:
                driver_definition = yaml.safe_load(f)
            if (
                "dummygen" in driver_definition
                and "enum" in driver_definition["dummygen"]
            ):
                enum = driver_definition["dummygen"]["enum"]

        if "class" in opt["dummy"]:
            class_name = opt["dummy"]["class"]
        else:
            class_name = driver_definition["codegen"]["class"]

        run_flags = ["drivers=dummy"]

        repo = Repo("../multipass/build/repo.acp")

        if "accounting" in opt and "getEnergy" not in map(
            lambda x: x.name, pta.transitions
        ):
            for state in pta.get_state_names():
                pta.add_transition(state, state, "getEnergy")

        pta.set_random_energy_model()

        if "accounting" in opt:
            if "," in opt["accounting"]:
                accounting_settings = opt["accounting"].split(",")
                accounting_name = accounting_settings[0]
                accounting_options = dict(
                    map(lambda x: x.split("="), accounting_settings[1:])
                )
                accounting_object = get_accountingmethod(accounting_name)(
                    class_name, pta, **accounting_options
                )
            else:
                accounting_object = get_accountingmethod(opt["accounting"])(
                    class_name, pta
                )
        else:
            accounting_object = None
        drv = MultipassDriver(
            class_name,
            pta,
            repo.class_by_name[class_name],
            enum=enum,
            accounting=accounting_object,
        )
        with open("../multipass/src/driver/dummy.cc", "w") as f:
            f.write(drv.impl)
        with open("../multipass/include/driver/dummy.h", "w") as f:
            f.write(drv.header)

    if ".json" not in modelfile:
        with open(modelfile, "r") as f:
            driver_definition = yaml.safe_load(f)
        if "codegen" in driver_definition and "flags" in driver_definition["codegen"]:
            if run_flags is None:
                run_flags = driver_definition["codegen"]["flags"]
    if "run" in opt:
        run_flags.extend(opt["run"].split())

    runs = list(
        pta.dfs(
            opt["depth"],
            with_arguments=True,
            with_parameters=True,
            trace_filter=opt["trace-filter"],
        )
    )

    num_transitions = len(runs)

    if len(runs) == 0:
        print(
            "DFS returned no traces -- perhaps your trace-filter is too restrictive?",
            file=sys.stderr,
        )
        sys.exit(1)

    need_return_values = False
    if next(filter(lambda x: len(x.return_value_handlers), pta.transitions), None):
        # A PTA transition indicates that its return value determines an online
        # parameter (e.g. Nrf24l01 getObserveTx, which is used to determine the
        # actual number of transmission retries after a write operation)
        need_return_values = True
    # elif 'accounting' in opt:
    #    # getEnergy() returns energy data. Log it.
    #    need_return_values = True

    if "mimosa" in opt:
        harness = TransitionHarness(
            gpio_pin=timer_pin,
            pta=pta,
            log_return_values=need_return_values,
            repeat=1,
            post_transition_delay_us=20,
        )
    elif "energytrace" in opt:
        # Use barcode sync by default
        gpio_mode = "bar"
        energytrace_sync = None
        if "sync" in opt["energytrace"] and opt["energytrace"]["sync"] != "bar":
            gpio_mode = "around"
            energytrace_sync = "led"
        harness = OnboardTimerHarness(
            gpio_pin=timer_pin,
            gpio_mode=gpio_mode,
            pta=pta,
            counter_limits=target.get_counter_limits_us(run_flags),
            log_return_values=need_return_values,
            repeat=1,
            energytrace_sync=energytrace_sync,
            remove_nop_from_timings=False,  # kein einfluss auf ungenauigkeiten
        )
    elif "timing" in opt:
        harness = OnboardTimerHarness(
            gpio_pin=timer_pin,
            pta=pta,
            counter_limits=target.get_counter_limits_us(run_flags),
            log_return_values=need_return_values,
            repeat=opt["repeat"],
        )

    if len(args) > 1:
        results = run_benchmark(
            args[1],
            pta,
            runs,
            opt["arch"],
            opt["app"],
            run_flags,
            harness,
            opt["sleep"],
            opt["repeat"],
            runs_total=len(runs),
            dummy="dummy" in opt,
        )
        json_out = {
            "opt": opt,
            "pta": pta.to_json(),
            "traces": list(map(lambda x: x[1].traces, results)),
            "raw_output": list(map(lambda x: x[2].get_lines(), results)),
            "files": list(map(lambda x: x[3], results)),
            "configs": list(map(lambda x: x[2].get_config(), results)),
        }
        extra_files = flatten(map(flatten, json_out["files"]))
        if "instance" in pta.codegen:
            output_prefix = (
                opt["data"] + time.strftime("/%Y%m%d-%H%M%S-") + pta.codegen["instance"]
            )
        else:
            output_prefix = opt["data"] + time.strftime("/%Y%m%d-%H%M%S-ptalog")
        if len(extra_files):
            with open("ptalog.json", "w") as f:
                json.dump(json_out, f)
            with tarfile.open("{}.tar".format(output_prefix), "w") as tar:
                tar.add("ptalog.json")
                for extra_file in extra_files:
                    tar.add(extra_file)
            print(" --> {}.tar".format(output_prefix))
            os.remove("ptalog.json")
            for extra_file in extra_files:
                os.remove(extra_file)
        else:
            with open("{}.json".format(output_prefix), "w") as f:
                json.dump(json_out, f)
            print(" --> {}.json".format(output_prefix))
    else:
        outbuf = benchmark_from_runs(pta, runs, harness, repeat=opt["repeat"])
        print(outbuf.getvalue())

    sys.exit(0)
