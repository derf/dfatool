#!/usr/bin/env python3
"""
analyze-timing -- generate analytic energy model from annotated OnboardTimerHarness traces.

Usage:
PYTHONPATH=lib bin/analyze-timing.py [options] <tracefiles ...>

analyze-timing generates an analytic energy model (``AnalyticModel``)from one or more annotated
traces generated by generate-dfa-benchmark using OnboardTimerHarness. By default, it does nothing else --
use one of the --plot-* or --show-* options to examine the generated model.

Options:
--plot-unparam=<name>:<attribute>:<Y axis label>[;<name>:<attribute>:<label>;...]
    Plot all mesurements for <name> <attribute> without regard for parameter values.
    X axis is measurement number/id.

--plot-param=<name> <attribute> <parameter> [gplearn function][;<name> <attribute> <parameter> [function];...]
    Plot measurements for <name> <attribute> by <parameter>.
    X axis is parameter value.
    Plots the model function as one solid line for each combination of non-<parameter>
    parameters. Also plots the corresponding measurements.
    If gplearn function is set, it is plotted using dashed lines.

--param-info
    Show parameter names and values

--show-models=<static|paramdetection|param|all|tex>
    static: show static model values as well as parameter detection heuristic
    paramdetection: show stddev of static/lut/fitted model
    param: show parameterized model functions and regression variable values
    all: all of the above
    tex: print tex/pgfplots-compatible model data on stdout

--show-quality=<table|summary|all|tex>
    table: show static/fitted/lut SMAPE and MAE for each name and attribute
    summary: show static/fitted/lut SMAPE and MAE for each attribute, averaged over all states/transitions
    all: all of the above
    tex: print tex/pgfplots-compatible model quality data on stdout

--ignored-trace-indexes=<i1,i2,...>
    Specify traces which should be ignored due to bogus data. 1 is the first
    trace, 2 the second, and so on.

--cross-validate=<method>:<count>
    Perform cross validation when computing model quality.
    Only works with --show-quality=table at the moment.
    If <method> is "montecarlo": Randomly divide data into 2/3 training and 1/3
    validation, <count> times. Reported model quality is the average of all
    validation runs. Data is partitioned without regard for parameter values,
    so a specific parameter combination may be present in both training and
    validation sets or just one of them.

--function-override=<name attribute function>[;<name> <attribute> <function>;...]
    Manually specify the function to fit for <name> <attribute>. A function
    specified this way bypasses parameter detection: It is always assigned,
    even if the model seems to be independent of the parameters it references.

--with-safe-functions
    If set, include "safe" functions (safe_log, safe_inv, safe_sqrt) which are
    also defined for cases such as safe_inv(0) or safe_sqrt(-1). This allows
    a greater range of functions to be tried during fitting.

--hwmodel=<hwmodel.json>
    Load DFA hardware model from JSON

--export-energymodel=<model.json>
    Export energy model. Requires --hwmodel.

--filter-param=<parameter name>=<parameter value>[,<parameter name>=<parameter value>...]
    Only consider measurements where <parameter name> is <parameter value>
    All other measurements (including those where it is None, that is, has
    not been set yet) are discarded. Note that this may remove entire
    function calls from the model.
"""

import getopt
import json
import re
import sys
from dfatool import plotter
from dfatool.dfatool import AnalyticModel, TimingData, pta_trace_to_aggregate
from dfatool.dfatool import gplearn_to_function
from dfatool.dfatool import CrossValidator
from dfatool.utils import filter_aggregate_by_param
from dfatool.parameters import prune_dependent_parameters

opt = dict()


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


def format_quality_measures(result):
    if "smape" in result:
        return "{:6.2f}% / {:9.0f}".format(result["smape"], result["mae"])
    else:
        return "{:6}    {:9.0f}".format("", result["mae"])


def model_quality_table(result_lists, info_list):
    for state_or_tran in result_lists[0]["by_name"].keys():
        for key in result_lists[0]["by_name"][state_or_tran].keys():
            buf = "{:20s} {:15s}".format(state_or_tran, key)
            for i, results in enumerate(result_lists):
                info = info_list[i]
                buf += "  |||  "
                if info is None or info(state_or_tran, key):
                    result = results["by_name"][state_or_tran][key]
                    buf += format_quality_measures(result)
                else:
                    buf += "{:6}----{:9}".format("", "")
            print(buf)


def print_text_model_data(model, pm, pq, lm, lq, am, ai, aq):
    print("")
    print(r"key attribute $1 - \frac{\sigma_X}{...}$")
    for state_or_tran in model.by_name.keys():
        for attribute in model.attributes(state_or_tran):
            print(
                "{} {} {:.8f}".format(
                    state_or_tran,
                    attribute,
                    model.stats.generic_param_dependence_ratio(
                        state_or_tran, attribute
                    ),
                )
            )

    print("")
    print(r"key attribute parameter $1 - \frac{...}{...}$")
    for state_or_tran in model.by_name.keys():
        for attribute in model.attributes(state_or_tran):
            for param in model.parameters():
                print(
                    "{} {} {} {:.8f}".format(
                        state_or_tran,
                        attribute,
                        param,
                        model.stats.param_dependence_ratio(
                            state_or_tran, attribute, param
                        ),
                    )
                )
            if state_or_tran in model._num_args:
                for arg_index in range(model._num_args[state_or_tran]):
                    print(
                        "{} {} {:d} {:.8f}".format(
                            state_or_tran,
                            attribute,
                            arg_index,
                            model.stats.arg_dependence_ratio(
                                state_or_tran, attribute, arg_index
                            ),
                        )
                    )


if __name__ == "__main__":

    ignored_trace_indexes = []
    discard_outliers = None
    safe_functions_enabled = False
    function_override = {}
    show_models = []
    show_quality = []
    hwmodel = None
    energymodel_export_file = None
    xv_method = None
    xv_count = 10

    try:
        optspec = (
            "plot-unparam= plot-param= show-models= show-quality= "
            "ignored-trace-indexes= discard-outliers= function-override= "
            "filter-param= "
            "cross-validate= "
            "corrcoef param-info "
            "with-safe-functions hwmodel= export-energymodel="
        )
        raw_opts, args = getopt.getopt(sys.argv[1:], "", optspec.split(" "))

        for option, parameter in raw_opts:
            optname = re.sub(r"^--", "", option)
            opt[optname] = parameter

        if "ignored-trace-indexes" in opt:
            ignored_trace_indexes = list(
                map(int, opt["ignored-trace-indexes"].split(","))
            )
            if 0 in ignored_trace_indexes:
                print("[E] arguments to --ignored-trace-indexes start from 1")

        if "discard-outliers" in opt:
            discard_outliers = float(opt["discard-outliers"])

        if "function-override" in opt:
            for function_desc in opt["function-override"].split(";"):
                state_or_tran, attribute, *function_str = function_desc.split(" ")
                function_override[(state_or_tran, attribute)] = " ".join(function_str)

        if "show-models" in opt:
            show_models = opt["show-models"].split(",")

        if "show-quality" in opt:
            show_quality = opt["show-quality"].split(",")

        if "cross-validate" in opt:
            xv_method, xv_count = opt["cross-validate"].split(":")
            xv_count = int(xv_count)

        if "with-safe-functions" in opt:
            safe_functions_enabled = True

        if "hwmodel" in opt:
            with open(opt["hwmodel"], "r") as f:
                hwmodel = json.load(f)

        if "corrcoef" not in opt:
            opt["corrcoef"] = False

        if "filter-param" in opt:
            opt["filter-param"] = list(
                map(lambda x: x.split("="), opt["filter-param"].split(","))
            )
        else:
            opt["filter-param"] = list()

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    raw_data = TimingData(args)

    preprocessed_data = raw_data.get_preprocessed_data()
    by_name, parameters, arg_count = pta_trace_to_aggregate(
        preprocessed_data, ignored_trace_indexes
    )

    prune_dependent_parameters(by_name, parameters)

    filter_aggregate_by_param(by_name, parameters, opt["filter-param"])

    model = AnalyticModel(
        by_name,
        parameters,
        arg_count,
        use_corrcoef=opt["corrcoef"],
        function_override=function_override,
    )

    if xv_method:
        xv = CrossValidator(AnalyticModel, by_name, parameters, arg_count)

    if "param-info" in opt:
        for state in model.names:
            print("{}:".format(state))
            for param in model.parameters:
                print(
                    "    {} = {}".format(
                        param, model.stats.distinct_values[state][param]
                    )
                )

    if "plot-unparam" in opt:
        for kv in opt["plot-unparam"].split(";"):
            state_or_trans, attribute, ylabel = kv.split(":")
            fname = "param_y_{}_{}.pdf".format(state_or_trans, attribute)
            plotter.plot_y(
                model.by_name[state_or_trans][attribute],
                xlabel="measurement #",
                ylabel=ylabel,
            )

    if len(show_models):
        print("--- simple static model ---")
    static_model = model.get_static()
    if "static" in show_models or "all" in show_models:
        for trans in model.names:
            print("{:10s}: {:.0f} µs".format(trans, static_model(trans, "duration")))
            for param in model.parameters:
                print(
                    "{:10s}  dependence on {:15s}: {:.2f}".format(
                        "",
                        param,
                        model.stats.param_dependence_ratio(trans, "duration", param),
                    )
                )
                if model.stats.has_codependent_parameters(trans, "duration", param):
                    print(
                        "{:24s}  co-dependencies: {:s}".format(
                            "",
                            ", ".join(
                                model.stats.codependent_parameters(
                                    trans, "duration", param
                                )
                            ),
                        )
                    )
                    for param_dict in model.stats.codependent_parameter_value_dicts(
                        trans, "duration", param
                    ):
                        print("{:24s}  parameter-aware for {}".format("", param_dict))
        # import numpy as np
        # safe_div = np.vectorize(lambda x,y: 0. if x == 0 else 1 - x/y)
        # ratio_by_value = safe_div(model.stats.stats['write']['duration']['lut_by_param_values']['max_retry_count'], model.stats.stats['write']['duration']['std_by_param_values']['max_retry_count'])
        # err_mode = np.seterr('warn')
        # dep_by_value = ratio_by_value > 0.5
        # np.seterr(**err_mode)
        # Eigentlich sollte hier ein paar mal True stehen, ist aber nicht so...
        # und warum ist da eine non-power-of-two Zahl von True-Einträgen in der Matrix? 3 stück ist komisch...
        # print(dep_by_value)

    if xv_method == "montecarlo":
        static_quality = xv.montecarlo(lambda m: m.get_static(), xv_count)
    else:
        static_quality = model.assess(static_model)

    if len(show_models):
        print("--- LUT ---")
    lut_model = model.get_param_lut()

    if xv_method == "montecarlo":
        lut_quality = xv.montecarlo(lambda m: m.get_param_lut(fallback=True), xv_count)
    else:
        lut_quality = model.assess(lut_model)

    if len(show_models):
        print("--- param model ---")

    param_model, param_info = model.get_fitted(
        safe_functions_enabled=safe_functions_enabled
    )

    if "paramdetection" in show_models or "all" in show_models:
        for transition in model.names:
            for attribute in ["duration"]:
                info = param_info(transition, attribute)
                print(
                    "{:10s} {:10s} non-param stddev {:f}".format(
                        transition,
                        attribute,
                        model.stats.stats[transition][attribute]["std_static"],
                    )
                )
                print(
                    "{:10s} {:10s} param-lut stddev {:f}".format(
                        transition,
                        attribute,
                        model.stats.stats[transition][attribute]["std_param_lut"],
                    )
                )
                for param in sorted(
                    model.stats.stats[transition][attribute]["std_by_param"].keys()
                ):
                    print(
                        "{:10s} {:10s} {:10s} stddev {:f}".format(
                            transition,
                            attribute,
                            param,
                            model.stats.stats[transition][attribute]["std_by_param"][
                                param
                            ],
                        )
                    )
                    print(
                        "{:10s} {:10s} dependence on {:15s}: {:.2f}".format(
                            transition,
                            attribute,
                            param,
                            model.stats.param_dependence_ratio(
                                transition, attribute, param
                            ),
                        )
                    )
                for i, arg_stddev in enumerate(
                    model.stats.stats[transition][attribute]["std_by_arg"]
                ):
                    print(
                        "{:10s} {:10s} arg{:d} stddev {:f}".format(
                            transition, attribute, i, arg_stddev
                        )
                    )
                    print(
                        "{:10s} {:10s} dependence on arg{:d}: {:.2f}".format(
                            transition,
                            attribute,
                            i,
                            model.stats.arg_dependence_ratio(transition, attribute, i),
                        )
                    )
                if info is not None:
                    for param_name in sorted(info["fit_result"].keys(), key=str):
                        param_fit = info["fit_result"][param_name]["results"]
                        for function_type in sorted(param_fit.keys()):
                            function_rmsd = param_fit[function_type]["rmsd"]
                            print(
                                "{:10s} {:10s} {:10s} mean {:10s} RMSD {:.0f}".format(
                                    transition,
                                    attribute,
                                    str(param_name),
                                    function_type,
                                    function_rmsd,
                                )
                            )

    if "param" in show_models or "all" in show_models:
        for trans in model.names:
            for attribute in ["duration"]:
                if param_info(trans, attribute):
                    print(
                        "{:10s}: {:10s}: {}".format(
                            trans,
                            attribute,
                            param_info(trans, attribute)["function"].model_function,
                        )
                    )
                    print(
                        "{:10s}  {:10s}  {}".format(
                            "",
                            "",
                            param_info(trans, attribute)["function"].model_args,
                        )
                    )

    if xv_method == "montecarlo":
        analytic_quality = xv.montecarlo(lambda m: m.get_fitted()[0], xv_count)
    else:
        analytic_quality = model.assess(param_model)

    if "tex" in show_models or "tex" in show_quality:
        print_text_model_data(
            model,
            static_model,
            static_quality,
            lut_model,
            lut_quality,
            param_model,
            param_info,
            analytic_quality,
        )

    if "table" in show_quality or "all" in show_quality:
        model_quality_table(
            [static_quality, analytic_quality, lut_quality], [None, param_info, None]
        )

    if "plot-param" in opt:
        for kv in opt["plot-param"].split(";"):
            state_or_trans, attribute, param_name, *function = kv.split(" ")
            if len(function):
                function = gplearn_to_function(" ".join(function))
            else:
                function = None
            plotter.plot_param(
                model,
                state_or_trans,
                attribute,
                model.param_index(param_name),
                extra_function=function,
            )

    sys.exit(0)
