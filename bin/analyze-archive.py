#!/usr/bin/env python3
"""
analyze-archive - generate PTA energy model from dfatool benchmark traces

analyze-archive generates a PTA energy model from one or more annotated
traces generated by dfatool. By default, it does nothing else.

Cross-Validation help:
    If <method> is "montecarlo": Randomly divide data into 2/3 training and 1/3
    validation, <count> times. Reported model quality is the average of all
    validation runs. Data is partitioned without regard for parameter values,
    so a specific parameter combination may be present in both training and
    validation sets or just one of them.

    If <method> is "kfold": Perform k-fold cross validation with k=<count>.
    Divide data into 1-1/k training and 1/k validation, <count> times.
    In the first set, items 0, k, 2k, ... ard used for validation, in the
    second set, items 1, k+1, 2k+1, ... and so on.
    validation, <count> times. Reported model quality is the average of all
    validation runs. Data is partitioned without regard for parameter values,
    so a specific parameter combination may be present in both training and
    validation sets or just one of them.

Trace Export:
    Each JSON file lists all occurences of the corresponding state/transition in the
    benchmark's PTA trace. Each occurence contains the corresponding PTA
    parameters (if any) in 'parameter' and measurement results in 'offline'.
    As measurements are typically run repeatedly, 'offline' is in turn a list
    of measurements: offline[0]['uW'] is the power trace of the first
    measurement of this state/transition, offline[1]['uW'] corresponds t the
    second measurement, etc. Values are provided in microwatts.
    For example, TX.json[0].offline[0].uW corresponds to the first measurement
    of the first TX state in the benchmark, and TX.json[5].offline[2].uW
    corresponds to the third measurement of the sixth TX state in the benchmark.
    WARNING: Several GB of RAM and disk space are required for complex measurements.
             (JSON files may grow very large -- we trade efficiency for easy handling)
"""

import argparse
import json
import logging
import random
import re
import sys
import time
import dfatool.cli
import dfatool.utils
import dfatool.functions as df
from dfatool import plotter
from dfatool.loader import RawData, pta_trace_to_aggregate
from dfatool.model import PTAModel
from dfatool.validation import CrossValidator
from dfatool.automata import PTA


def print_model_quality(results):
    for state_or_tran in results.keys():
        print()
        for key, result in results[state_or_tran].items():
            if "smape" in result:
                print(
                    "{:20s} {:15s} {:.2f}% / {:.0f}".format(
                        state_or_tran, key, result["smape"], result["mae"]
                    )
                )
            else:
                print("{:20s} {:15s} {:.0f}".format(state_or_tran, key, result["mae"]))


def model_summary_table(result_list):
    buf = "transition duration"
    for results in result_list:
        if len(buf):
            buf += "  |||  "
        buf += dfatool.cli.format_quality_measures(results["duration_by_trace"])
    print(buf)
    buf = "total energy       "
    for results in result_list:
        if len(buf):
            buf += "  |||  "
        buf += dfatool.cli.format_quality_measures(results["energy_by_trace"])
    print(buf)
    buf = "rel total energy   "
    for results in result_list:
        if len(buf):
            buf += "  |||  "
        buf += dfatool.cli.format_quality_measures(results["rel_energy_by_trace"])
    print(buf)
    buf = "state-only energy  "
    for results in result_list:
        if len(buf):
            buf += "  |||  "
        buf += dfatool.cli.format_quality_measures(results["state_energy_by_trace"])
    print(buf)
    buf = "transition timeout "
    for results in result_list:
        if len(buf):
            buf += "  |||  "
        buf += dfatool.cli.format_quality_measures(results["timeout_by_trace"])
    print(buf)


def get_kconfig(model):
    buf = str()
    for param_name in model.parameters:
        unique_values = set()
        is_relevant = False
        for name in model.names:
            unique_values.update(
                model.attr_by_name[name]["power"].stats.distinct_values_by_param_name[
                    param_name
                ]
            )
            for attr in model.attr_by_name[name].values():
                # FIXME this indicates whether it might depend on the parameter, not whether it actually uses it (there's no API for that yet)
                if attr.stats.depends_on_param(param_name):
                    is_relevant = True
        unique_values.discard(None)
        if not unique_values or not is_relevant:
            # unused by the model
            continue

        buf += f"config {param_name}\n"
        buf += f'  prompt "{param_name}"\n'
        if unique_values == {0, 1}:
            buf += "  bool\n"
        elif all(map(dfatool.utils.is_numeric, unique_values)):
            buf += "  int\n"
            buf += f"  range {min(unique_values)} {max(unique_values)}\n"
        else:
            buf += "  string\n"
            buf += f"  #!accept [{unique_values}]\n"

    return buf


def plot_traces(preprocessed_data, sot_name):
    traces = list()
    timestamps = list()
    for trace in preprocessed_data:
        for state_or_transition in trace["trace"]:
            if state_or_transition["name"] == sot_name:
                timestamps.extend(
                    map(lambda x: x["plot"][0], state_or_transition["offline"])
                )
                traces.extend(
                    map(lambda x: x["plot"][1], state_or_transition["offline"])
                )
    if len(traces) == 0:
        print(
            f"""Did not find traces for state or transition {sot_name}. Abort.""",
            file=sys.stderr,
        )
        sys.exit(2)

    if len(traces) > 40:
        print(f"""Truncating plot to 40 of {len(traces)} traces (random sample)""")
        indexes = random.sample(range(len(traces)), 40)
        timestamps = [timestamps[i] for i in indexes]
        traces = [traces[i] for i in indexes]

    plotter.plot_xy(
        timestamps, traces, xlabel="t [s]", ylabel="P [W]", title=sot_name, family=True
    )


if __name__ == "__main__":
    ignored_trace_indexes = []
    safe_functions_enabled = False
    function_override = {}
    show_quality = []
    pta = None
    energymodel_export_file = None
    trace_export_dir = None
    xv_count = 10

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    dfatool.cli.add_standard_arguments(parser)
    parser.add_argument(
        "--no-cache", action="store_true", help="Do not load cached measurement results"
    )
    parser.add_argument(
        "--plot-traces",
        metavar="NAME",
        type=str,
        help="Plot power trace for state or transition NAME. X axis is wrong for non-MIMOSA measurements",
    )
    parser.add_argument(
        "--remove-outliers",
        action="store_true",
        help="Remove outliers exceeding the configured z score (default: 10)",
    )
    parser.add_argument(
        "--z-score",
        type=int,
        default=10,
        help="Configure z score for outlier detection (and optional removel)",
    )
    parser.add_argument(
        "--show-quality",
        choices=["table", "summary"],
        action="append",
        default=list(),
        help="table: show LUT, model, and static prediction error for each state/transition and attribute.\n"
        "summary: show static/fitted/lut SMAPE and MAE for each attribute, averaged over all states/transitions.",
    )
    parser.add_argument(
        "--ignored-trace-indexes",
        metavar="<i1,i2,...>",
        type=str,
        help="Specify traces which should be ignored due to bogus data. "
        "1 is the first trace, 2 the second, and so on.",
    )
    parser.add_argument(
        "--export-traces",
        metavar="DIRECTORY",
        type=str,
        help="Export power traces of all states and transitions to DIRECTORY. "
        "Creates a JSON file for each state and transition.",
    )
    parser.add_argument(
        "--with-safe-functions",
        action="store_true",
        help="Include 'safe' functions (safe_log, safe_inv, safe_sqrt) which are also defined for 0 and -1. "
        "This allows a greater range of functions to be tried during fitting.",
    )
    parser.add_argument(
        "--hwmodel",
        metavar="FILE",
        type=str,
        help="Load DFA hardware model from JSON or YAML FILE",
    )
    parser.add_argument(
        "--export-pta-dot",
        metavar="FILE",
        type=str,
        help="Export PTA representation suitable for Graphviz dot to FILE",
    )
    parser.add_argument(
        "--export-energymodel",
        metavar="FILE",
        type=str,
        help="Export JSON energy model to FILE. Works out of the box for v1+, requires --hwmodel for v0",
    )
    parser.add_argument(
        "--export-webconf",
        metavar="FILE",
        type=str,
        help="Export KConfig model to FILE.Kconfig and energy model to FILE.js. Works out of the box for v1+, requires --hwmodel for v0",
    )
    parser.add_argument(
        "--with-substates",
        metavar="PELT_CONFIG",
        type=str,
        help="Perform substate analysis",
    )
    parser.add_argument(
        "--force-tree",
        action="store_true",
        help="Build regression tree without checking whether static/analytic functions are sufficient.",
    )
    parser.add_argument("measurement", nargs="+")

    args = parser.parse_args()
    dfatool.cli.sanity_check(args)

    if args.log_level:
        numeric_level = getattr(logging, args.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            print(f"Invalid log level: {args.log_level}", file=sys.stderr)
            sys.exit(1)
        logging.basicConfig(level=numeric_level)

    if args.ignored_trace_indexes:
        ignored_trace_indexes = list(map(int, args.ignored_trace_indexes.split(",")))
        if 0 in ignored_trace_indexes:
            logging.error("arguments to --ignored-trace-indexes start from 1")

    if args.function_override:
        for function_desc in args.function_override.split(";"):
            state_or_tran, attribute, *function_str = function_desc.split(" ")
            function_override[(state_or_tran, attribute)] = " ".join(function_str)

    show_quality = args.show_quality

    if args.filter_param:
        args.filter_param = list(
            map(lambda x: x.split("="), args.filter_param.split(","))
        )
    else:
        args.filter_param = list()

    if args.with_safe_functions is not None:
        safe_functions_enabled = True

    if args.hwmodel:
        pta = PTA.from_file(args.hwmodel)

    raw_data = RawData(
        args.measurement,
        with_traces=(
            args.export_traces is not None
            or args.plot_traces is not None
            or args.with_substates is not None
        ),
        skip_cache=args.no_cache,
    )

    if args.info:
        print(" ".join(raw_data.filenames) + ":")
        data_source = "???"
        if raw_data.ptalog:
            options = " --".join(
                map(lambda kv: f"{kv[0]}={str(kv[1])}", raw_data.ptalog["opt"].items())
            )
            print(f"    Options: --{options}")
        if raw_data.version <= 1:
            data_source = "MIMOSA"
        elif raw_data.version == 2:
            if raw_data.ptalog and "sync" in raw_data.ptalog["opt"]["energytrace"]:
                data_source = "MSP430 EnergyTrace, sync={}".format(
                    raw_data.ptalog["opt"]["energytrace"]["sync"]
                )
            else:
                data_source = "MSP430 EnergyTrace"
        elif raw_data.version == 3:
            data_source = "Keysight"
        print(f"    Data source ID: {raw_data.version} ({data_source})")

    preprocessed_data = raw_data.get_preprocessed_data()

    if args.info:
        print(
            f"""    Valid Runs: {raw_data.preprocessing_stats["num_valid"]}/{raw_data.preprocessing_stats["num_runs"]}"""
        )
        state_durations = map(
            lambda x: str(x["state_duration"]), raw_data.setup_by_fileno
        )
        print(f"""    State Duration: {" / ".join(state_durations)} ms""")

    if args.export_traces:
        uw_per_sot = dict()
        for trace in preprocessed_data:
            for state_or_transition in trace["trace"]:
                name = state_or_transition["name"]
                if name not in uw_per_sot:
                    uw_per_sot[name] = list()
                for elem in state_or_transition["offline"]:
                    elem["plot"] = list(elem["plot"])
                uw_per_sot[name].append(state_or_transition)
        for name, data in uw_per_sot.items():
            target = f"{args.export_traces}/{name}.json"
            print(f"exporting {target} ...")
            with open(target, "w") as f:
                json.dump(data, f)

    if args.with_substates is not None:
        arg_dict = dict()
        if args.with_substates != "":
            for kv in args.with_substates.split(","):
                k, v = kv.split("=")
                try:
                    arg_dict[k] = float(v)
                except ValueError:
                    arg_dict[k] = v
        args.with_substates = arg_dict

    if args.plot_traces:
        plot_traces(preprocessed_data, args.plot_traces)

    if raw_data.preprocessing_stats["num_valid"] == 0:
        print("No valid data available. Abort.", file=sys.stderr)
        sys.exit(2)

    if pta is None and raw_data.pta is not None:
        pta = PTA.from_json(raw_data.pta)

    by_name, parameters, arg_count = pta_trace_to_aggregate(
        preprocessed_data, ignored_trace_indexes
    )

    if args.ignore_param:
        args.ignore_param = args.ignore_param.split(",")
        dfatool.utils.ignore_param(by_name, parameters, args.ignore_param)

    dfatool.utils.filter_aggregate_by_param(by_name, parameters, args.filter_param)

    if args.param_shift:
        param_shift = dfatool.cli.parse_param_shift(args.param_shift)
        dfatool.utils.shift_param_in_aggregate(by_name, parameters, param_shift)

    if args.normalize_nfp:
        norm = dfatool.cli.parse_nfp_normalization(args.normalize_nfp)
        dfatool.utils.normalize_nfp_in_aggregate(by_name, norm)

    dfatool.utils.detect_outliers_in_aggregate(
        by_name, z_limit=args.z_score, remove_outliers=args.remove_outliers
    )

    constructor_start = time.time()
    model = PTAModel(
        by_name,
        parameters,
        arg_count,
        traces=preprocessed_data,
        function_override=function_override,
        pta=pta,
        pelt=args.with_substates,
        force_tree=args.force_tree,
    )
    constructor_duration = time.time() - constructor_start

    if args.info:
        dfatool.cli.print_info_by_name(model, by_name)

    if args.export_pgf_unparam:
        dfatool.cli.export_pgf_unparam(model, args.export_pgf_unparam)

    if args.export_json_unparam:
        dfatool.cli.export_json_unparam(model, args.export_json_unparam)

    if args.cross_validate:
        xv_method, xv_count = args.cross_validate.split(":")
        xv_count = int(xv_count)
        xv = CrossValidator(
            PTAModel, by_name, parameters, arg_count, force_tree=args.force_tree
        )
        xv.parameter_aware = args.parameter_aware_cross_validation
    else:
        xv_method = None

    if args.plot_unparam:
        for kv in args.plot_unparam.split(";"):
            state_or_trans, attribute, ylabel = kv.split(":")
            fname = "param_y_{}_{}.pdf".format(state_or_trans, attribute)
            plotter.plot_y(
                model.by_name[state_or_trans][attribute],
                xlabel="measurement #",
                ylabel=ylabel,
                output=fname,
            )

    if args.boxplot_unparam:
        plotter.boxplot(
            model.names,
            [model.by_name[name]["power"] for name in model.names],
            xlabel="State/Transition",
            ylabel="Average Power [µW]",
            output=f"{args.boxplot_unparam}power.pdf",
        )
        plotter.boxplot(
            model.transitions,
            [model.by_name[name]["duration"] for name in model.transitions],
            xlabel="Transition",
            ylabel="Duration [µs]",
            output=f"{args.boxplot_unparam}duration.pdf",
        )
        for name in model.names:
            plotter.boxplot(
                [name],
                [model.by_name[name]["power"]],
                xlabel="State/Transition",
                ylabel="Average Power [µW]",
                output=f"{args.boxplot_unparam}{name}-power.pdf",
            )
        for trans in model.transitions:
            plotter.boxplot(
                [trans],
                [model.by_name[trans]["duration"]],
                xlabel="Transition",
                ylabel="duration [µs]",
                output=f"{args.boxplot_unparam}{trans}-duration.pdf",
            )

    static_model = model.get_static()
    if "static" in args.show_model or "all" in args.show_model:
        print("--- simple static model ---")
        for state in model.states:
            for attribute in model.attributes(state):
                dfatool.cli.print_static(
                    model,
                    static_model,
                    state,
                    attribute,
                    with_dependence="all" in args.show_model,
                )
        if args.with_substates:
            for submodel in model.submodel_by_name.values():
                for substate in submodel.states:
                    for subattribute in submodel.attributes(substate):
                        dfatool.cli.print_static(
                            submodel,
                            submodel.get_static(),
                            substate,
                            subattribut,
                            with_dependence="all" in args.show_model,
                        )

        for trans in model.transitions:
            if "energy" in model.attributes(trans):
                try:
                    print(
                        "{:10s}: {:.0f} / {:.0f} / {:.0f} pJ  ({:.2f} / {:.2f} / {:.2f})".format(
                            trans,
                            static_model(trans, "energy"),
                            static_model(trans, "rel_energy_prev"),
                            static_model(trans, "rel_energy_next"),
                            model.attr_by_name[trans][
                                "energy"
                            ].stats.generic_param_dependence_ratio(),
                            model.attr_by_name[trans][
                                "rel_energy_prev"
                            ].stats.generic_param_dependence_ratio(),
                            model.attr_by_name[trans][
                                "rel_energy_next"
                            ].stats.generic_param_dependence_ratio(),
                        )
                    )
                except KeyError:
                    print(
                        "{:10s}: {:.0f} pJ  ({:.2f})".format(
                            trans,
                            static_model(trans, "energy"),
                            model.attr_by_name[trans][
                                "energy"
                            ].stats.generic_param_dependence_ratio(),
                        )
                    )
            else:
                try:
                    print(
                        "{:10s}: {:.0f} / {:.0f} / {:.0f} pJ  (E=P·t)".format(
                            trans,
                            static_model(trans, "power")
                            * static_model(trans, "duration"),
                            static_model(trans, "rel_power_prev")
                            * static_model(trans, "duration"),
                            static_model(trans, "rel_power_next")
                            * static_model(trans, "duration"),
                        )
                    )
                except KeyError:
                    print(
                        "{:10s}: {:.0f} pJ  (E=P·t)".format(
                            trans,
                            static_model(trans, "power")
                            * static_model(trans, "duration"),
                        )
                    )
            print(
                "{:10s}: {:.0f} µs  ({:.2f})".format(
                    trans,
                    static_model(trans, "duration"),
                    model.attr_by_name[trans][
                        "duration"
                    ].stats.generic_param_dependence_ratio(),
                )
            )
            try:
                print(
                    "{:10s}: {:.0f} / {:.0f} / {:.0f} µW  ({:.2f} / {:.2f} / {:.2f})".format(
                        trans,
                        static_model(trans, "power"),
                        static_model(trans, "rel_power_prev"),
                        static_model(trans, "rel_power_next"),
                        model.attr_by_name[trans][
                            "power"
                        ].stats.generic_param_dependence_ratio(),
                        model.attr_by_name[trans][
                            "rel_power_prev"
                        ].stats.generic_param_dependence_ratio(),
                        model.attr_by_name[trans][
                            "rel_power_next"
                        ].stats.generic_param_dependence_ratio(),
                    )
                )
            except KeyError:
                print(
                    "{:10s}: {:.0f} pJ  ({:.2f})".format(
                        trans,
                        static_model(trans, "power"),
                        model.attr_by_name[trans][
                            "power"
                        ].stats.generic_param_dependence_ratio(),
                    )
                )

    if xv_method == "montecarlo":
        static_quality, _ = xv.montecarlo(lambda m: m.get_static(), xv_count)
    elif xv_method == "kfold":
        static_quality, _ = xv.kfold(lambda m: m.get_static(), xv_count)
    else:
        static_quality = model.assess(static_model)

    if len(args.show_model):
        print("--- LUT ---")
    lut_model = model.get_param_lut()
    lut_quality = model.assess(lut_model)

    if len(args.show_model):
        print("--- param model ---")

    # get_fitted_sub -> with sub-state detection and modeling
    fit_start_time = time.time()
    param_model, param_info = model.get_fitted(
        safe_functions_enabled=safe_functions_enabled
    )
    fit_duration = time.time() - fit_start_time

    if "paramdetection" in args.show_model or "all" in args.show_model:
        for name in model.names:
            for attribute in model.attributes(name):
                info = param_info(name, attribute)
                print(
                    "{:10s} {:10s} non-param stddev {:f}".format(
                        name,
                        attribute,
                        model.attr_by_name[name][attribute].stats.std_static,
                    )
                )
                print(
                    "{:10s} {:10s} param-lut stddev {:f}".format(
                        name,
                        attribute,
                        model.attr_by_name[name][attribute].stats.std_param_lut,
                    )
                )
                for param in sorted(
                    model.attr_by_name[name][attribute].stats.std_by_param.keys()
                ):
                    print(
                        "{:10s} {:10s} {:10s} stddev {:f}".format(
                            name,
                            attribute,
                            param,
                            model.attr_by_name[name][attribute].stats.std_by_param[
                                param
                            ],
                        )
                    )
                for arg_index in range(model.attr_by_name[name][attribute].arg_count):
                    print(
                        "{:10s} {:10s} {:10s} stddev {:f}".format(
                            name,
                            attribute,
                            f"arg{arg_index}",
                            model.attr_by_name[name][attribute].stats.std_by_arg[
                                arg_index
                            ],
                        )
                    )
                if type(info) is df.AnalyticFunction:
                    for param_name in sorted(info.fit_by_param.keys(), key=str):
                        param_fit = info.fit_by_param[param_name]["results"]
                        for function_type in sorted(param_fit.keys()):
                            function_rmsd = param_fit[function_type]["rmsd"]
                            print(
                                "{:10s} {:10s} {:10s} mean {:10s} RMSD {:.0f}".format(
                                    name,
                                    attribute,
                                    str(param_name),
                                    function_type,
                                    function_rmsd,
                                )
                            )

    if "param" in args.show_model or "all" in args.show_model:
        for state in model.states:
            for attribute in model.attributes(state):
                info = param_info(state, attribute)
                dfatool.cli.print_model(
                    f"{state:10s} {attribute:15s}", info, model.parameters
                )
        for trans in model.transitions:
            for attribute in model.attributes(trans):
                info = param_info(trans, attribute)
                dfatool.cli.print_model(
                    f"{trans:10s} {attribute:15s}", info, model.parameters
                )
        if args.with_substates:
            for submodel in model.submodel_by_name.values():
                sub_param_model, sub_param_info = submodel.get_fitted()
                for substate in submodel.states:
                    for subattribute in submodel.attributes(substate):
                        info = sub_param_info(substate, subattribute)
                        if type(info) is df.AnalyticFunction:
                            print(
                                "{:10s} {:15s}: {}".format(
                                    substate, subattribute, info.model_function
                                )
                            )
                            print("{:10s} {:15s}  {}".format("", "", info.model_args))

    if args.with_substates:
        for state in model.states:
            if (
                type(model.attr_by_name[state]["power"].model_function)
                is df.SubstateFunction
            ):
                # sub-state models need to know the duration of the state / transition. only needed for eval.
                model.attr_by_name[state]["power"].model_function.static_duration = (
                    raw_data.setup_by_fileno[0]["state_duration"] * 1e3
                )

    if xv_method == "montecarlo":
        xv.export_filename = args.export_xv
        analytic_quality, xv_analytic_models = xv.montecarlo(
            lambda m: m.get_fitted()[0], xv_count
        )
    elif xv_method == "kfold":
        xv.export_filename = args.export_xv
        analytic_quality, xv_analytic_models = xv.kfold(
            lambda m: m.get_fitted()[0], xv_count
        )
    else:
        if args.export_raw_predictions:
            analytic_quality, raw_results = model.assess(param_model, return_raw=True)
            with open(args.export_raw_predictions, "w") as f:
                json.dump(raw_results, f, cls=dfatool.utils.NpEncoder)
        else:
            analytic_quality = model.assess(param_model)
        xv_analytic_models = None

    if "table" in show_quality or "all" in show_quality:
        dfatool.cli.model_quality_table(
            lut=lut_quality,
            model=analytic_quality,
            static=static_quality,
            model_info=param_info,
            xv_method=xv_method,
            error_metric=args.error_metric,
        )
        if args.with_substates:
            for submodel in model.submodel_by_name.values():
                sub_static_model = submodel.get_static()
                sub_static_quality = submodel.assess(sub_static_model)
                sub_lut_model = submodel.get_param_lut()
                sub_lut_quality = submodel.assess(sub_lut_model)
                sub_param_model, sub_param_info = submodel.get_fitted()
                sub_analytic_quality = submodel.assess(sub_param_model)
                dfatool.cli.model_quality_table(
                    lut=sub_lut_quality,
                    model=sub_analytic_quality,
                    static=sub_static_quality,
                    model_info=sub_param_info,
                    error_metric=args.error_metric,
                )

    if "overall" in show_quality or "all" in show_quality:
        print("overall state static/param/lut MAE assuming equal state distribution:")
        print(
            "    {:6.1f}  /  {:6.1f}  /  {:6.1f}  µW".format(
                model.assess_states(static_model),
                model.assess_states(param_model),
                model.assess_states(lut_model),
            )
        )
        distrib = dict()
        num_states = len(model.states)
        p95_state = None
        for state in model.states:
            distrib[state] = 1.0 / num_states

        if "STANDBY1" in model.states:
            p95_state = "STANDBY1"
        elif "SLEEP" in model.states:
            p95_state = "SLEEP"

        if p95_state is not None:
            for state in distrib.keys():
                distrib[state] = 0.05 / (num_states - 1)
            distrib[p95_state] = 0.95

            print(f"overall state static/param/lut MAE assuming 95% {p95_state}:")
            print(
                "    {:6.1f}  /  {:6.1f}  /  {:6.1f}  µW".format(
                    model.assess_states(static_model, distribution=distrib),
                    model.assess_states(param_model, distribution=distrib),
                    model.assess_states(lut_model, distribution=distrib),
                )
            )

    if "summary" in show_quality or "all" in show_quality:
        model_summary_table(
            [
                model.assess_on_traces(static_model),
                model.assess_on_traces(param_model),
                model.assess_on_traces(lut_model),
            ]
        )

    if args.show_model_size:
        dfatool.cli.print_model_size(model)

    if args.boxplot_param:
        dfatool.cli.boxplot_param(args, model)

    if args.plot_param:
        for kv in args.plot_param.split(";"):
            try:
                state_or_trans, attribute, param_name, *function = kv.split(":")
            except ValueError:
                print(
                    "Usage: --plot-param='state_or_trans attribute param_name [additional function spec]'",
                    file=sys.stderr,
                )
                sys.exit(1)
            if len(function):
                function = df.gplearn_to_function(" ".join(function))
            else:
                function = None
            plotter.plot_param(
                model,
                state_or_trans,
                attribute,
                model.param_index(param_name),
                extra_function=function,
                output=f"{state_or_trans}-{attribute}-{param_name}.pdf",
                show=not args.non_interactive,
            )

    if args.export_dref:
        dref = raw_data.to_dref()
        dref.update(
            model.to_dref(
                static_quality,
                lut_quality,
                analytic_quality,
                xv_models=xv_analytic_models,
            )
        )
        dref["constructor duration"] = (constructor_duration, r"\second")
        dref["regression duration"] = (fit_duration, r"\second")
        dfatool.cli.export_dataref(
            args.export_dref, dref, precision=args.dref_precision
        )

    if args.export_json:
        with open(args.export_json, "w") as f:
            json.dump(model.to_json(), f, sort_keys=True, cls=dfatool.utils.NpEncoder)

    if args.export_webconf:
        if not pta:
            print(
                "Note: v0 measurements do not embed the PTA used for benchmark generation. Estimating PTA from recorded observations."
            )
        json_model = model.to_json()
        json_model_str = json.dumps(
            json_model, indent=2, sort_keys=True, cls=dfatool.utils.NpEncoder
        )
        for function_str, function_body in model.webconf_function_map():
            json_model_str = json_model_str.replace(function_str, function_body)

        buf = "class watModel {\n"
        buf += f"model = {json_model_str};\n"
        buf += "};"

        with open(f"{args.export_webconf}.js", "w") as f:
            f.write(buf)
        with open(f"{args.export_webconf}.kconfig", "w") as f:
            f.write(get_kconfig(model))

    if args.export_energymodel:
        if not pta:
            print(
                "Note: v0 measurements do not embed the PTA used for benchmark generation. Estimating PTA from recorded observations."
            )
        json_model = model.to_json()
        with open(args.export_energymodel, "w") as f:
            json.dump(
                json_model, f, indent=2, sort_keys=True, cls=dfatool.utils.NpEncoder
            )

    if args.export_dot:
        dfatool.cli.export_dot(model, args.export_dot)

    if args.export_pta_dot:
        if not pta:
            print(
                "Note: v0 measurements do not embed the PTA used for benchmark generation. Estimating PTA from recorded observations."
            )
        json_model = model.to_json()
        with open(args.export_pta_dot, "w") as f:
            f.write(model.to_dot())

    sys.exit(0)
