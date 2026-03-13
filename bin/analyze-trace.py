#!/usr/bin/env python3

"""
analyze-trace - Generate a performance-aware behaviour model from log files

foo
"""

import argparse
import dfatool.cli
import dfatool.plotter
import dfatool.utils
import dfatool.functions as df
from dfatool.behaviour import SDKBehaviourModel
from dfatool.loader import Logfile
from dfatool.model import AnalyticModel
from dfatool.validation import CrossValidator, _xv_partitions_kfold
from functools import reduce
import logging
import json
import re
import sys
import time


def parse_trace(filename):
    return _parse_logfile(filename, is_trace=True)


def parse_log(filename):
    return _parse_logfile(filename, is_trace=True)[0]


def _parse_logfile(filename, is_trace):
    loader = Logfile()

    if filename.endswith("xz"):
        import lzma

        with lzma.open(filename, "rt") as f:
            return loader.load(f, is_trace=is_trace)
    with open(filename, "r") as f:
        return loader.load(f, is_trace=is_trace)


def join_annotations(ref, base, new):
    offset = len(ref)
    return base + list(map(lambda x: x.apply_offset(offset), new))


def format_eq(equality):
    if equality == "<=":
        return "≤"
    if equality == "==":
        return "="
    if equality == ">=":
        return "≥"
    return equality


def format_condition(key, eq, value):
    if type(value) is tuple:
        if len(value) == 2:
            return f"{key}∈({value[0]},{value[1]})"
        if value[0] < value[-1] and value[-1] - value[0] == len(value) - 1:
            return f"{key}∈({value[0]},…,{value[-1]})"
        for i, entry in enumerate(value):
            if i == 0:
                buf = str(value[0])
            elif entry - 1 != value[i - 1]:
                if (
                    i >= 3
                    and value[i - 1] == value[i - 2] + 1
                    and value[i - 1] == value[i - 3] + 2
                ):
                    buf += f",…,{value[i-1]}"
                elif value[i - 1] == value[i - 2] + 1:
                    buf += f",{value[i-1]}"
                buf += f",{entry}"
        buf += f",{value[-1]}"
        return f"{key}∈({buf})"
    return f"{key}{format_eq(eq)}{value}"


def format_guard(guard):
    return "∧".join(map(lambda kv: format_condition(*kv), guard))


def get_expected_trace(bm, observations, annotation):
    if annotation.kernels:
        expected_trace = (
            observations[annotation.start.offset : annotation.kernels[0].offset]
            + observations[annotation.kernels[-1].offset : annotation.end.offset]
        )
    else:
        expected_trace = observations[annotation.start.offset : annotation.end.offset]
    return expected_trace


def assess_trace(bm, observations, annotation):
    predicted_trace = bm.get_trace(annotation.start.name, annotation.end.param)
    predicted_trace = list(map(lambda edge_param: edge_param[0], predicted_trace))
    expected_trace = get_expected_trace(bm, observations, annotation)
    expected_trace = (
        ["__init__"]
        + list(
            map(
                lambda entry: f"""{entry["name"]} @ {entry["place"]}""",
                expected_trace,
            )
        )
        + ["__end__"]
    )
    return expected_trace == predicted_trace


def assess_trace_args(
    bm,
    wfcfg_model,
    function_model,
    observations,
    annotation,
    bm_param_names,
    fun_param_names,
):
    predicted_trace = bm.get_trace(annotation.start.name, annotation.end.param)
    expected_trace = ["__init__"] + get_expected_trace(bm, observations, annotation)
    predicted_args = list()
    expected_args = list()
    reliable = True
    for i, (callsite, param) in enumerate(predicted_trace):
        if not " @ " in callsite:
            continue
        if i >= len(expected_trace):
            reliable = False
            break
        call, _ = callsite.split(" @ ")
        workload_tuple = tuple(dfatool.utils.param_dict_to_list(param, bm_param_names))
        for arg_name, expected_value in expected_trace[i]["param"].items():
            predicted_value = round(
                wfcfg_model(callsite, arg_name, param=workload_tuple)
            )
            predicted_args.append(predicted_value)
            expected_args.append(expected_value)

    return predicted_args, expected_args, reliable


def assess_trace_nfps(
    bm,
    wfcfg_model,
    function_model,
    observations,
    annotation,
    bm_param_names,
    fun_param_names,
):
    predicted_trace = bm.get_trace(annotation.start.name, annotation.end.param)
    expected_trace = ["__init__"] + get_expected_trace(bm, observations, annotation)
    reliable = True

    # Baseline: NFP value predicted using function_model and expected_trace[i]["param"][arg_name]
    base_attr = dict()

    # Prediction: NFP value predicted using function_model and wfcfg_model
    pred_attr = dict()

    # Expectation: NFP value observed in ground truth
    exp_attr = dict()
    for i, (callsite, param) in enumerate(predicted_trace):
        if not " @ " in callsite:
            continue
        if i >= len(expected_trace):
            reliable = False
            break
        call, _ = callsite.split(" @ ")
        workload_tuple = tuple(dfatool.utils.param_dict_to_list(param, bm_param_names))
        base_param = dict()
        call_param = dict()
        for arg_name, arg_value in expected_trace[i]["param"].items():
            base_param[arg_name] = arg_value
            call_param[arg_name] = round(
                wfcfg_model(callsite, arg_name, param=workload_tuple)
            )
        base_param_tuple = tuple(
            dfatool.utils.param_dict_to_list(base_param, fun_param_names)
        )
        pred_param_tuple = tuple(
            dfatool.utils.param_dict_to_list(call_param, fun_param_names)
        )
        for attr_name, expected_value in expected_trace[i]["attribute"].items():
            baseline_value = function_model(call, attr_name, param=base_param_tuple)
            predicted_value = function_model(call, attr_name, param=pred_param_tuple)
            if attr_name not in pred_attr:
                base_attr[attr_name] = {"all": list(), "by_call": dict()}
                pred_attr[attr_name] = {"all": list(), "by_call": dict()}
                exp_attr[attr_name] = {"all": list(), "by_call": dict()}
            base_attr[attr_name]["all"].append(baseline_value)
            pred_attr[attr_name]["all"].append(predicted_value)
            exp_attr[attr_name]["all"].append(expected_value)
            if call not in base_attr[attr_name]["by_call"]:
                base_attr[attr_name]["by_call"][call] = list()
                pred_attr[attr_name]["by_call"][call] = list()
                exp_attr[attr_name]["by_call"][call] = list()
            base_attr[attr_name]["by_call"][call].append(baseline_value)
            pred_attr[attr_name]["by_call"][call].append(predicted_value)
            exp_attr[attr_name]["by_call"][call].append(expected_value)
    return base_attr, pred_attr, exp_attr, reliable


def main():
    timing = dict()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    dfatool.cli.add_standard_arguments(parser)
    parser.add_argument(
        "--with-function-models",
        action="store_true",
        help="Learn and evaluate function-level prediction models as well",
    )
    parser.add_argument(
        "--show-wfcfg",
        action="store_true",
        help="Show Weighted Featured Control Flow Graph model",
    )
    parser.add_argument("--unroll-loops", action="store_true", help="Unroll loops")
    parser.add_argument(
        "logfiles",
        nargs="+",
        type=str,
        help="Path to benchmark output (.txt or .txt.xz)",
    )
    args = parser.parse_args()
    dfatool.cli.sanity_check(args)

    if args.ignore_param:
        args.ignore_param = args.ignore_param.split(",")
    else:
        args.ignore_param = list()

    def pprint(msg):
        if args.progress:
            print(f"{msg} ...")

    if args.log_level:
        numeric_level = getattr(logging, args.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            print(f"Invalid log level: {args.log_level}", file=sys.stderr)
            sys.exit(1)
        logging.basicConfig(
            level=numeric_level,
            format="{asctime} {levelname}:{name}:{message}",
            style="{",
        )

    #
    # Model for function call arguments → latency
    #

    if args.with_function_models:
        pprint("Loading traces for per-function performance models")
        observations = reduce(
            lambda a, b: a + b,
            map(parse_log, args.logfiles),
        )
        pprint("Parsing traces for per-function performance models")
        function_by_name, function_parameter_names = (
            dfatool.utils.observations_to_by_name(observations)
        )

    pprint("Loading traces for behaviour model")
    observations, annotations = reduce(
        lambda a, b: (a[0] + b[0], join_annotations(a[0], a[1], b[1])),
        map(parse_trace, args.logfiles),
    )

    if args.ignore_param:
        for observation in observations:
            for arg in args.ignore_param:
                observation["param"].pop(arg, None)

    pprint("Building FCFG (states, edges, feature guards)")
    bm = SDKBehaviourModel(
        observations,
        annotations,
        unroll_loops=args.unroll_loops,
        show_progress=args.progress,
    )

    if args.show_wfcfg:
        for name in sorted(bm.delta_by_name.keys()):
            for t_from, t_to_set in bm.delta_by_name[name].items():
                i_to_transition = dict()
                delta_param_sets = list()
                to_names = list()

                for t_to in sorted(t_to_set):
                    delta_params = bm.delta_param_by_name[name][(t_from, t_to)]
                    delta_param_sets.append(delta_params)
                    to_names.append(t_to)
                    n_confs = len(delta_params)
                    symbol = " "
                    if bm.transition_guard[t_from] is None:
                        # invalid model
                        guard = "?"
                    elif t_to not in bm.transition_guard[t_from]:
                        guard = "⊥"
                    elif not bm.transition_guard[t_from][t_to]:
                        guard = "⊤"
                    else:
                        guard = " ∨ ".join(
                            map(format_guard, bm.transition_guard[t_from][t_to])
                        )
                    print(f"{name}  {t_from}  →  {t_to}  {symbol}  ({guard})")

                for i in range(len(delta_param_sets)):
                    for j in range(i + 1, len(delta_param_sets)):
                        if not delta_param_sets[i].isdisjoint(delta_param_sets[j]):
                            intersection = delta_param_sets[i].intersection(
                                delta_param_sets[j]
                            )
                            logging.error(
                                f"Outbound transitions of <{t_from}> are not deterministic: <{to_names[i]}> and <{to_names[j]}> are both taken for {intersection}"
                            )
                            raise RuntimeError(
                                f"Outbound transitions of <{t_from}> are not deterministic"
                            )

                print("")

    #
    # Model for workload → function call arguments
    #

    pprint("Parsing traces for behaviour model")
    by_name, parameter_names = dfatool.utils.observations_to_by_name(
        bm.meta_observations
    )
    bm.cleanup()

    if args.filter_observation:
        args.filter_observation = list(
            map(lambda x: tuple(x.split(":")), args.filter_observation.split(","))
        )
        dfatool.utils.filter_aggregate_by_observation(by_name, args.filter_observation)
        if args.with_function_models:
            logging.error(
                "--with-function-models and --filter-observation are mutually exclusive"
            )

    pprint("Building WFCFG (argument prediction)")
    model = AnalyticModel(by_name, parameter_names)

    if args.with_function_models:
        pprint("Building CART for function performance prediction")
        dfatool.utils.ignore_param(
            function_by_name, function_parameter_names, args.ignore_param
        )
        function_model = AnalyticModel(
            function_by_name, function_parameter_names, model_type="cart"
        )
        function_cart, _ = function_model.get_fitted()

    # BM-specific
    trace_dref = dict()
    n_correct = 0
    n_wrong = 0
    for annotation in annotations:
        if assess_trace(bm, observations, annotation):
            n_correct += 1
        else:
            n_wrong += 1
    trace_dref["trace/accuracy/training"] = (
        n_correct / (n_correct + n_wrong) * 100,
        r"\percent",
    )
    print(
        f"{n_correct / (n_correct + n_wrong) * 100 :.1f}% of training traces predicted correctly"
    )

    n_correct = 0
    n_wrong = 0
    for training, validation in _xv_partitions_kfold(len(annotations)):
        training_annotations = list(map(lambda i: annotations[i], training))
        validation_annotations = list(map(lambda i: annotations[i], validation))
        bm_xv = SDKBehaviourModel(
            observations, training_annotations, show_progress=args.progress
        )
        bm_xv.cleanup()
        for annotation in validation_annotations:
            if assess_trace(bm_xv, observations, annotation):
                n_correct += 1
            else:
                n_wrong += 1
        del bm_xv
    trace_dref["trace/accuracy/validation"] = (
        n_correct / (n_correct + n_wrong) * 100,
        r"\percent",
    )
    print(
        f"{n_correct / (n_correct + n_wrong) * 100 :.1f}% of validation traces predicted correctly"
    )
    # /BM-specific

    if args.info:
        dfatool.cli.print_info_by_name(model, by_name)

    if args.information_gain:
        dfatool.cli.print_information_gain_by_name(model, by_name)

    if args.export_csv_unparam:
        dfatool.cli.export_csv_unparam(
            model, args.export_csv_unparam, dialect=args.export_csv_dialect
        )

    if args.export_pgf_unparam:
        dfatool.cli.export_pgf_unparam(model, args.export_pgf_unparam)

    if args.export_json_unparam:
        dfatool.cli.export_json_unparam(model, args.export_json_unparam)

    if args.plot_unparam:
        for kv in args.plot_unparam.split(";"):
            state_or_trans, attribute, ylabel = kv.split(":")
            fname = "param_y_{}_{}.pdf".format(state_or_trans, attribute)
            dfatool.plotter.plot_y(
                model.by_name[state_or_trans][attribute],
                xlabel="measurement #",
                ylabel=ylabel,
                # output=fname,
                show=not args.non_interactive,
            )

    if args.boxplot_unparam:
        title = None
        if args.filter_param:
            title = "filter: " + ", ".join(
                map(lambda kv: f"{kv[0]}={kv[1]}", args.filter_param)
            )
        for name in model.names:
            attr_names = sorted(model.attributes(name))
            dfatool.plotter.boxplot(
                attr_names,
                [model.by_name[name][attr] for attr in attr_names],
                xlabel="Attribute",
                output=f"{args.boxplot_unparam}{name}.pdf",
                title=title,
                show=not args.non_interactive,
            )
            for attribute in attr_names:
                dfatool.plotter.boxplot(
                    [attribute],
                    [model.by_name[name][attribute]],
                    output=f"{args.boxplot_unparam}{name}-{attribute}.pdf",
                    title=title,
                    show=not args.non_interactive,
                )

    if args.boxplot_param:
        dfatool.cli.boxplot_param(args, model)

    if args.cross_validate:
        xv_method, xv_count = args.cross_validate.split(":")
        xv_count = int(xv_count)
        xv = CrossValidator(
            AnalyticModel,
            by_name,
            parameter_names,
            force_tree=args.force_tree,
            compute_stats=not args.skip_param_stats,
            show_progress=args.progress,
        )
        xv.parameter_aware = args.parameter_aware_cross_validation
    else:
        xv_method = None
        xv_count = None

    static_model = model.get_static()

    ts = time.time()
    lut_model = model.get_param_lut()
    timing["get lut"] = time.time() - ts

    if lut_model is None:
        lut_quality = None
    else:
        ts = time.time()
        lut_quality = model.assess(lut_model, with_sum=args.add_total_observation)
        timing["assess lut"] = time.time() - ts

    ts = time.time()
    param_model, param_info = model.get_fitted()
    timing["get model"] = time.time() - ts

    if args.with_function_models:
        exp_args = list()
        pred_args = list()
        ok = True
        for annotation in annotations:
            pred, exp, ok = assess_trace_args(
                bm,
                param_model,
                function_cart,
                observations,
                annotation,
                parameter_names,
                function_parameter_names,
            )
            pred_args += pred
            exp_args += exp
            if not ok:
                ok = False
        arg_err = dfatool.utils.regression_measures(pred_args, exp_args)
        smape = arg_err["smape"]
        annot = "" if ok else " (UNRELIABLE)"
        trace_dref["callsite-arg/error/training/smape"] = smape
        print(f"{smape:.1f}% training callsite argument prediction error{annot}")

        pred_args = list()
        exp_args = list()
        ok = True
        for training, validation in _xv_partitions_kfold(len(annotations)):
            training_annotations = list(map(lambda i: annotations[i], training))
            validation_annotations = list(map(lambda i: annotations[i], validation))
            bm_xv = SDKBehaviourModel(
                observations, training_annotations, show_progress=args.progress
            )
            bm_xv.cleanup()
            for annotation in validation_annotations:
                pred, exp, ok = assess_trace_args(
                    bm_xv,
                    param_model,
                    function_cart,
                    observations,
                    annotation,
                    parameter_names,
                    function_parameter_names,
                )
                pred_args += pred
                exp_args += exp
                if not ok:
                    ok = False
        arg_err = dfatool.utils.regression_measures(pred_args, exp_args)
        smape = arg_err["smape"]
        annot = "" if ok else " (UNRELIABLE)"
        trace_dref["callsite-arg/error/validation/smape"] = smape
        print(f"{smape:.1f}% validation callsite argument prediction error{annot}")

        base_attr = dict()
        pred_attr = dict()
        exp_attr = dict()
        val_pred_attr = dict()
        val_exp_attr = dict()
        t_ok = True
        v_ok = True
        for annotation in annotations:
            a_base_attr, a_pred_attr, a_exp_attr, ok = assess_trace_nfps(
                bm,
                param_model,
                function_cart,
                observations,
                annotation,
                parameter_names,
                function_parameter_names,
            )
            if not ok:
                t_ok = False
            for attr_name in a_pred_attr.keys():
                if attr_name not in pred_attr:
                    base_attr[attr_name] = {"all": list(), "by_call": dict()}
                    pred_attr[attr_name] = {"all": list(), "by_call": dict()}
                    exp_attr[attr_name] = {"all": list(), "by_call": dict()}
                base_attr[attr_name]["all"] += a_base_attr[attr_name]["all"]
                pred_attr[attr_name]["all"] += a_pred_attr[attr_name]["all"]
                exp_attr[attr_name]["all"] += a_exp_attr[attr_name]["all"]
                for call in a_base_attr[attr_name]["by_call"].keys():
                    if call not in base_attr[attr_name]["by_call"]:
                        base_attr[attr_name]["by_call"][call] = list()
                        pred_attr[attr_name]["by_call"][call] = list()
                        exp_attr[attr_name]["by_call"][call] = list()
                    base_attr[attr_name]["by_call"][call] += a_base_attr[attr_name][
                        "by_call"
                    ][call]
                    pred_attr[attr_name]["by_call"][call] += a_pred_attr[attr_name][
                        "by_call"
                    ][call]
                    exp_attr[attr_name]["by_call"][call] += a_exp_attr[attr_name][
                        "by_call"
                    ][call]

        for training, validation in _xv_partitions_kfold(len(annotations)):
            training_annotations = list(map(lambda i: annotations[i], training))
            validation_annotations = list(map(lambda i: annotations[i], validation))
            bm_xv = SDKBehaviourModel(
                observations, training_annotations, show_progress=args.progress
            )
            bm_xv.cleanup()
            for annotation in validation_annotations:
                _, a_pred_attr, a_exp_attr, ok = assess_trace_nfps(
                    bm_xv,
                    param_model,
                    function_cart,
                    observations,
                    annotation,
                    parameter_names,
                    function_parameter_names,
                )
                if not ok:
                    v_ok = False
                for attr_name in a_pred_attr.keys():
                    if attr_name not in val_pred_attr:
                        val_pred_attr[attr_name] = {"all": list(), "by_call": dict()}
                        val_exp_attr[attr_name] = {"all": list(), "by_call": dict()}
                    val_pred_attr[attr_name]["all"] += a_pred_attr[attr_name]["all"]
                    val_exp_attr[attr_name]["all"] += a_exp_attr[attr_name]["all"]
                    for call in a_pred_attr[attr_name]["by_call"].keys():
                        if call not in val_pred_attr[attr_name]["by_call"]:
                            val_pred_attr[attr_name]["by_call"][call] = list()
                            val_exp_attr[attr_name]["by_call"][call] = list()
                        val_pred_attr[attr_name]["by_call"][call] += a_pred_attr[
                            attr_name
                        ]["by_call"][call]
                        val_exp_attr[attr_name]["by_call"][call] += a_exp_attr[
                            attr_name
                        ]["by_call"][call]

        for attr_name in pred_attr.keys():
            print()

            arg_err = dfatool.utils.regression_measures(
                base_attr[attr_name]["all"], exp_attr[attr_name]["all"]
            )
            smape_base = arg_err["smape"]
            trace_dref[f"nfp/error/{attr_name}/baseline/smape"] = smape_base
            print(f"{smape_base:5.1f}%   baseline {attr_name} prediction error")

            arg_err = dfatool.utils.regression_measures(
                pred_attr[attr_name]["all"], exp_attr[attr_name]["all"]
            )
            smape = arg_err["smape"]
            annot = "" if t_ok else " (UNRELIABLE)"
            trace_dref[f"nfp/error/{attr_name}/training/smape"] = smape
            trace_dref[f"nfp/error/{attr_name}/trainingOverBaseline/smape"] = (
                smape - smape_base
            )
            print(f"{smape:5.1f}%   training {attr_name} prediction error{annot}")

            arg_err = dfatool.utils.regression_measures(
                val_pred_attr[attr_name]["all"], val_exp_attr[attr_name]["all"]
            )
            smape = arg_err["smape"]
            annot = "" if v_ok else " (UNRELIABLE)"
            trace_dref[f"nfp/error/{attr_name}/validation/smape"] = smape
            trace_dref[f"nfp/error/{attr_name}/validationOverBaseline/smape"] = (
                smape - smape_base
            )
            print(f"{smape:5.1f}% validation {attr_name} prediction error{annot}")
            for call in val_pred_attr[attr_name]["by_call"].keys():
                n_measurements = len(val_pred_attr[attr_name]["by_call"][call])
                arg_err = dfatool.utils.regression_measures(
                    val_pred_attr[attr_name]["by_call"][call],
                    val_exp_attr[attr_name]["by_call"][call],
                )
                smape = arg_err["smape"]
                print(
                    f"    {smape:5.1f}% validation {attr_name} prediction error for {n_measurements:3d}× {call}"
                )
        print()

    ts = time.time()
    if xv_method == "montecarlo":
        static_quality, _ = xv.montecarlo(
            lambda m: m.get_static(),
            xv_count,
            static=True,
            with_sum=args.add_total_observation,
        )
        xv.export_filename = args.export_xv
        analytic_quality, _ = xv.montecarlo(
            lambda m: m.get_fitted()[0], xv_count, with_sum=args.add_total_observation
        )
    elif xv_method == "kfold":
        static_quality, _ = xv.kfold(
            lambda m: m.get_static(),
            xv_count,
            static=True,
            with_sum=args.add_total_observation,
        )
        xv.export_filename = args.export_xv
        analytic_quality, _ = xv.kfold(
            lambda m: m.get_fitted()[0], xv_count, with_sum=args.add_total_observation
        )
    else:
        static_quality = model.assess(static_model, with_sum=args.add_total_observation)
        if args.export_raw_predictions:
            analytic_quality, raw_results = model.assess(param_model, return_raw=True)
            with open(args.export_raw_predictions, "w") as f:
                json.dump(raw_results, f, cls=dfatool.utils.NpEncoder)
        else:
            analytic_quality = model.assess(
                param_model, with_sum=args.add_total_observation
            )
    timing["assess model"] = time.time() - ts

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

    if "static" in args.show_model or "all" in args.show_model:
        print("--- static model ---")
        for name in sorted(model.names):
            for attribute in sorted(model.attributes(name)):
                dfatool.cli.print_static(
                    model,
                    static_model,
                    name,
                    attribute,
                    with_dependence="all" in args.show_model,
                    num_format=args.show_model_format,
                )

    if "param" in args.show_model or "all" in args.show_model:
        print("--- param model ---")
        for name in sorted(model.names):
            for attribute in sorted(model.attributes(name)):
                info = param_info(name, attribute)
                dfatool.cli.print_model(
                    f"{name:10s} {attribute:15s}",
                    info,
                    num_format=args.show_model_format,
                )

    if args.show_model_error:
        dfatool.cli.model_quality_table(
            lut=lut_quality,
            model=analytic_quality,
            static=static_quality,
            model_info=param_info,
            xv_method=xv_method,
            xv_count=xv_count,
            error_metric=args.error_metric,
            load_model=args.load_json,
        )

    if args.show_model_complexity:
        dfatool.cli.print_model_complexity(model)

    if args.export_dot:
        dfatool.cli.export_dot(model, args.export_dot)

    if args.export_dref or args.export_pseudo_dref:
        dref = model.to_dref(
            static_quality,
            lut_quality,
            analytic_quality,
            with_sum=args.add_total_observation,
        )
        for key, value in timing.items():
            dref[f"timing/{key}"] = (value, r"\second")
        for key, value in trace_dref.items():
            dref[f"trace/{key}"] = value

        if args.information_gain:
            for name in model.names:
                for attr in model.attributes(name):
                    mutual_information = model.mutual_information(name, attr)
                    for param in model.parameters:
                        if param in mutual_information:
                            dref[f"mutual information/{name}/{attr}/{param}"] = (
                                mutual_information[param]
                            )

        if args.export_pseudo_dref:
            dfatool.cli.export_pseudo_dref(
                args.export_pseudo_dref, dref, precision=args.dref_precision
            )
        if args.export_dref:
            dfatool.cli.export_dataref(
                args.export_dref, dref, precision=args.dref_precision
            )

    if args.export_json:
        with open(args.export_json, "w") as f:
            json.dump(
                model.to_json(
                    static_error=static_quality,
                    lut_error=lut_quality,
                    model_error=analytic_quality,
                ),
                f,
                sort_keys=True,
                cls=dfatool.utils.NpEncoder,
                indent=2,
            )

    if args.plot_param:
        for kv in args.plot_param.split(";"):
            try:
                state_or_trans, attribute, param_name = kv.split(":")
            except ValueError:
                print(
                    "Usage: --plot-param='state_or_trans:attribute:param_name'",
                    file=sys.stderr,
                )
                sys.exit(1)
            dfatool.plotter.plot_param(
                model,
                state_or_trans,
                attribute,
                model.param_index(param_name),
                title=state_or_trans,
                ylabel=attribute,
                xlabel=param_name,
                output=f"{state_or_trans}-{attribute}-{param_name}.pdf",
                show=not args.non_interactive,
                verbose_legend=args.plot_verbose_legend,
            )


if __name__ == "__main__":
    main()
