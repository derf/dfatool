#!/usr/bin/env python3
"""
analyze-archive -- generate PTA energy model from annotated legacy MIMOSA traces.

Usage:
PYTHONPATH=lib bin/analyze-archive.py [options] <tracefiles ...>

analyze-archive generates a PTA energy model from one or more annotated
traces generated by MIMOSA/dfatool-legacy. By default, it does nothing else --
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

--plot-traces=<name>
    Plot power trace for state or transition <name>.

--export-traces=<directory>
    Export power traces of all states and transitions to <directory>.
    Creates a JSON file for each state and transition. Each JSON file
    lists all occurences of the corresponding state/transition in the
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

--param-info
    Show parameter names and values

--show-models=<static|paramdetection|param|all|tex|html>
    static: show static model values as well as parameter detection heuristic
    paramdetection: show stddev of static/lut/fitted model
    param: show parameterized model functions and regression variable values
    all: all of the above
    tex: print tex/pgfplots-compatible model data on stdout
    html: print model and quality data as HTML table on stdout

--show-quality=<table|summary|all|tex|html>
    table: show static/fitted/lut SMAPE and MAE for each name and attribute
    summary: show static/fitted/lut SMAPE and MAE for each attribute, averaged over all states/transitions
    all: all of the above
    tex: print tex/pgfplots-compatible model quality data on stdout

--ignored-trace-indexes=<i1,i2,...>
    Specify traces which should be ignored due to bogus data. 1 is the first
    trace, 2 the second, and so on.

--discard-outliers=
    not supported at the moment

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

--filter-param=<parameter name>=<parameter value>[,<parameter name>=<parameter value>...]
    Only consider measurements where <parameter name> is <parameter value>
    All other measurements (including those where it is None, that is, has
    not been set yet) are discarded. Note that this may remove entire
    function calls from the model.

--hwmodel=<hwmodel.json|hwmodel.dfa>
    Load DFA hardware model from JSON or YAML

--export-energymodel=<model.json>
    Export energy model. Works out of the box for v1 and v2 logfiles. Requires --hwmodel for v0 logfiles.
"""

import getopt
import json
import re
import sys
from dfatool import plotter
from dfatool.dfatool import PTAModel, RawData, pta_trace_to_aggregate
from dfatool.dfatool import gplearn_to_function
from dfatool.dfatool import CrossValidator
from dfatool.utils import filter_aggregate_by_param
from dfatool.automata import PTA

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


def model_summary_table(result_list):
    buf = "transition duration"
    for results in result_list:
        if len(buf):
            buf += "  |||  "
        buf += format_quality_measures(results["duration_by_trace"])
    print(buf)
    buf = "total energy       "
    for results in result_list:
        if len(buf):
            buf += "  |||  "
        buf += format_quality_measures(results["energy_by_trace"])
    print(buf)
    buf = "rel total energy   "
    for results in result_list:
        if len(buf):
            buf += "  |||  "
        buf += format_quality_measures(results["rel_energy_by_trace"])
    print(buf)
    buf = "state-only energy  "
    for results in result_list:
        if len(buf):
            buf += "  |||  "
        buf += format_quality_measures(results["state_energy_by_trace"])
    print(buf)
    buf = "transition timeout "
    for results in result_list:
        if len(buf):
            buf += "  |||  "
        buf += format_quality_measures(results["timeout_by_trace"])
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


def print_html_model_data(model, pm, pq, lm, lq, am, ai, aq):
    state_attributes = model.attributes(model.states()[0])

    print(
        "<table><tr><th>state</th><th>"
        + "</th><th>".join(state_attributes)
        + "</th></tr>"
    )
    for state in model.states():
        print("<tr>", end="")
        print("<td>{}</td>".format(state), end="")
        for attribute in state_attributes:
            unit = ""
            if attribute == "power":
                unit = "µW"
            print(
                "<td>{:.0f} {} ({:.1f}%)</td>".format(
                    pm(state, attribute), unit, pq["by_name"][state][attribute]["smape"]
                ),
                end="",
            )
        print("</tr>")
    print("</table>")

    trans_attributes = model.attributes(model.transitions()[0])
    if "rel_energy_prev" in trans_attributes:
        trans_attributes.remove("rel_energy_next")

    print(
        "<table><tr><th>transition</th><th>"
        + "</th><th>".join(trans_attributes)
        + "</th></tr>"
    )
    for trans in model.transitions():
        print("<tr>", end="")
        print("<td>{}</td>".format(trans), end="")
        for attribute in trans_attributes:
            unit = ""
            if attribute == "duration":
                unit = "µs"
            elif attribute in ["energy", "rel_energy_prev"]:
                unit = "pJ"
            print(
                "<td>{:.0f} {} ({:.1f}%)</td>".format(
                    pm(trans, attribute), unit, pq["by_name"][trans][attribute]["smape"]
                ),
                end="",
            )
        print("</tr>")
    print("</table>")


if __name__ == "__main__":

    ignored_trace_indexes = []
    discard_outliers = None
    safe_functions_enabled = False
    function_override = {}
    show_models = []
    show_quality = []
    pta = None
    energymodel_export_file = None
    trace_export_dir = None
    xv_method = None
    xv_count = 10

    try:
        optspec = (
            "plot-unparam= plot-param= plot-traces= param-info show-models= show-quality= "
            "ignored-trace-indexes= discard-outliers= function-override= "
            "export-traces= "
            "filter-param= "
            "cross-validate= "
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

        if "filter-param" in opt:
            opt["filter-param"] = list(
                map(lambda x: x.split("="), opt["filter-param"].split(","))
            )
        else:
            opt["filter-param"] = list()

        if "with-safe-functions" in opt:
            safe_functions_enabled = True

        if "hwmodel" in opt:
            pta = PTA.from_file(opt["hwmodel"])

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    raw_data = RawData(
        args, with_traces=("export-traces" in opt or "plot-traces" in opt)
    )

    preprocessed_data = raw_data.get_preprocessed_data()

    if "export-traces" in opt:
        uw_per_sot = dict()
        for trace in preprocessed_data:
            for state_or_transition in trace["trace"]:
                name = state_or_transition["name"]
                if name not in uw_per_sot:
                    uw_per_sot[name] = list()
                for elem in state_or_transition["offline"]:
                    elem["uW"] = list(elem["uW"])
                uw_per_sot[name].append(state_or_transition)
        for name, data in uw_per_sot.items():
            target = f"{opt['export-traces']}/{name}.json"
            print(f"exporting {target} ...")
            with open(target, "w") as f:
                json.dump(data, f)

    if "plot-traces" in opt:
        traces = list()
        for trace in preprocessed_data:
            for state_or_transition in trace["trace"]:
                if state_or_transition["name"] == opt["plot-traces"]:
                    traces.extend(
                        map(lambda x: x["uW"], state_or_transition["offline"])
                    )
        plotter.plot_y(
            traces,
            xlabel="t [1e-5 s]",
            ylabel="P [uW]",
            title=opt["plot-traces"],
            family=True,
        )

    if raw_data.preprocessing_stats["num_valid"] == 0:
        print("No valid data available. Abort.")
        sys.exit(2)

    if pta is None and raw_data.pta is not None:
        pta = PTA.from_json(raw_data.pta)

    by_name, parameters, arg_count = pta_trace_to_aggregate(
        preprocessed_data, ignored_trace_indexes
    )

    filter_aggregate_by_param(by_name, parameters, opt["filter-param"])

    model = PTAModel(
        by_name,
        parameters,
        arg_count,
        traces=preprocessed_data,
        discard_outliers=discard_outliers,
        function_override=function_override,
        pta=pta,
    )

    if xv_method:
        xv = CrossValidator(PTAModel, by_name, parameters, arg_count)

    if "param-info" in opt:
        for state in model.states():
            print("{}:".format(state))
            for param in model.parameters():
                print(
                    "    {} = {}".format(
                        param, model.stats.distinct_values[state][param]
                    )
                )
        for transition in model.transitions():
            print("{}:".format(transition))
            for param in model.parameters():
                print(
                    "    {} = {}".format(
                        param, model.stats.distinct_values[transition][param]
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
                output=fname,
            )

    if len(show_models):
        print("--- simple static model ---")
    static_model = model.get_static()
    if "static" in show_models or "all" in show_models:
        for state in model.states():
            print(
                "{:10s}: {:.0f} µW  ({:.2f})".format(
                    state,
                    static_model(state, "power"),
                    model.stats.generic_param_dependence_ratio(state, "power"),
                )
            )
            for param in model.parameters():
                print(
                    "{:10s}  dependence on {:15s}: {:.2f}".format(
                        "",
                        param,
                        model.stats.param_dependence_ratio(state, "power", param),
                    )
                )
                if model.stats.has_codependent_parameters(state, "power", param):
                    print(
                        "{:24s}  co-dependencies: {:s}".format(
                            "",
                            ", ".join(
                                model.stats.codependent_parameters(
                                    state, "power", param
                                )
                            ),
                        )
                    )
                    for param_dict in model.stats.codependent_parameter_value_dicts(
                        state, "power", param
                    ):
                        print("{:24s}  parameter-aware for {}".format("", param_dict))

        for trans in model.transitions():
            # Mean power is not a typical transition attribute, but may be present for debugging or analysis purposes
            try:
                print(
                    "{:10s}: {:.0f} µW  ({:.2f})".format(
                        trans,
                        static_model(trans, "power"),
                        model.stats.generic_param_dependence_ratio(trans, "power"),
                    )
                )
            except KeyError:
                pass
            try:
                print(
                    "{:10s}: {:.0f} / {:.0f} / {:.0f} pJ  ({:.2f} / {:.2f} / {:.2f})".format(
                        trans,
                        static_model(trans, "energy"),
                        static_model(trans, "rel_energy_prev"),
                        static_model(trans, "rel_energy_next"),
                        model.stats.generic_param_dependence_ratio(trans, "energy"),
                        model.stats.generic_param_dependence_ratio(
                            trans, "rel_energy_prev"
                        ),
                        model.stats.generic_param_dependence_ratio(
                            trans, "rel_energy_next"
                        ),
                    )
                )
            except KeyError:
                print(
                    "{:10s}: {:.0f} pJ  ({:.2f})".format(
                        trans,
                        static_model(trans, "energy"),
                        model.stats.generic_param_dependence_ratio(trans, "energy"),
                    )
                )
            print("{:10s}: {:.0f} µs".format(trans, static_model(trans, "duration")))

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
        for state in model.states_and_transitions():
            for attribute in model.attributes(state):
                info = param_info(state, attribute)
                print(
                    "{:10s} {:10s} non-param stddev {:f}".format(
                        state,
                        attribute,
                        model.stats.stats[state][attribute]["std_static"],
                    )
                )
                print(
                    "{:10s} {:10s} param-lut stddev {:f}".format(
                        state,
                        attribute,
                        model.stats.stats[state][attribute]["std_param_lut"],
                    )
                )
                for param in sorted(
                    model.stats.stats[state][attribute]["std_by_param"].keys()
                ):
                    print(
                        "{:10s} {:10s} {:10s} stddev {:f}".format(
                            state,
                            attribute,
                            param,
                            model.stats.stats[state][attribute]["std_by_param"][param],
                        )
                    )
                if info is not None:
                    for param_name in sorted(info["fit_result"].keys(), key=str):
                        param_fit = info["fit_result"][param_name]["results"]
                        for function_type in sorted(param_fit.keys()):
                            function_rmsd = param_fit[function_type]["rmsd"]
                            print(
                                "{:10s} {:10s} {:10s} mean {:10s} RMSD {:.0f}".format(
                                    state,
                                    attribute,
                                    str(param_name),
                                    function_type,
                                    function_rmsd,
                                )
                            )

    if "param" in show_models or "all" in show_models:
        if not model.stats.can_be_fitted():
            print(
                "[!] measurements have insufficient distinct numeric parameters for fitting. A parameter-aware model is not available."
            )
        for state in model.states():
            for attribute in model.attributes(state):
                if param_info(state, attribute):
                    print(
                        "{:10s}: {}".format(
                            state, param_info(state, attribute)["function"]._model_str
                        )
                    )
                    print(
                        "{:10s}  {}".format(
                            "",
                            param_info(state, attribute)["function"]._regression_args,
                        )
                    )
        for trans in model.transitions():
            for attribute in model.attributes(trans):
                if param_info(trans, attribute):
                    print(
                        "{:10s}: {:10s}: {}".format(
                            trans,
                            attribute,
                            param_info(trans, attribute)["function"]._model_str,
                        )
                    )
                    print(
                        "{:10s}  {:10s}  {}".format(
                            "",
                            "",
                            param_info(trans, attribute)["function"]._regression_args,
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

    if "html" in show_models or "html" in show_quality:
        print_html_model_data(
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

    if "overall" in show_quality or "all" in show_quality:
        print("overall static/param/lut MAE assuming equal state distribution:")
        print(
            "    {:6.1f}  /  {:6.1f}  /  {:6.1f}  µW".format(
                model.assess_states(static_model),
                model.assess_states(param_model),
                model.assess_states(lut_model),
            )
        )
        print("overall static/param/lut MAE assuming 95% STANDBY1:")
        distrib = {"STANDBY1": 0.95, "POWERDOWN": 0.03, "TX": 0.01, "RX": 0.01}
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

    if "export-energymodel" in opt:
        if not pta:
            print("[E] --export-energymodel requires --hwmodel to be set")
            sys.exit(1)
        json_model = model.to_json()
        with open(opt["export-energymodel"], "w") as f:
            json.dump(json_model, f, indent=2, sort_keys=True)

    sys.exit(0)
