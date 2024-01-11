#!/usr/bin/env python3

import dfatool.functions as df
import dfatool.plotter
import logging
import numpy as np
import os
import sys

logger = logging.getLogger(__name__)


def sanity_check(args):
    if args.force_tree and bool(int(os.getenv("DFATOOL_FIT_FOL", "0"))):
        print(
            "--force-tree and DFATOOL_FIT_FOL=1 are mutually exclusive", file=sys.stderr
        )
        sys.exit(1)


def print_static(model, static_model, name, attribute, with_dependence=False):
    unit = "  "
    if attribute == "power":
        unit = "µW"
    elif attribute == "duration":
        unit = "µs"
    elif attribute == "substate_count":
        unit = "su"
    print(
        "{:10s}: {:28s} : {:.2f} {:s}  ({:.2f})".format(
            name,
            attribute,
            static_model(name, attribute),
            unit,
            model.attr_by_name[name][attribute].stats.generic_param_dependence_ratio(),
        )
    )
    if with_dependence:
        for param in model.parameters:
            print(
                "{:10s}  {:13s} {:15s}: {:.2f}".format(
                    "",
                    "dependence on",
                    param,
                    model.attr_by_name[name][attribute].stats.param_dependence_ratio(
                        param
                    ),
                )
            )


def print_info_by_name(model, by_name):
    for name in model.names:
        attr = list(model.attributes(name))[0]
        print(f"{name}:")
        print(f"""    Number of Measurements: {len(by_name[name][attr])}""")
        for param in model.parameters:
            print(
                "    Parameter {} ∈ {}".format(
                    param,
                    model.attr_by_name[name][attr].stats.distinct_values_by_param_name[
                        param
                    ],
                )
            )
        if name in model._num_args:
            for i in range(model._num_args[name]):
                print(
                    "    Argument  {} ∈ {}".format(
                        i,
                        model.attr_by_name[name][
                            attr
                        ].stats.distinct_values_by_param_index[
                            len(model.parameters) + i
                        ],
                    )
                )
        for attr in sorted(model.attributes(name)):
            print(
                "    Observation {} ∈ [{:.2f}, {:.2f}]".format(
                    attr,
                    model.attr_by_name[name][attr].min(),
                    model.attr_by_name[name][attr].max(),
                )
            )


def print_analyticinfo(prefix, info):
    model_function = info.model_function.removeprefix("0 + ")
    for i in range(len(info.model_args)):
        model_function = model_function.replace(
            f"regression_arg({i})", str(info.model_args[i])
        )
    model_function = model_function.replace("+ -", "- ")
    print(f"{prefix}: {model_function}")


def print_staticinfo(prefix, info):
    print(f"{prefix}: {info.value}")


def print_cartinfo(prefix, info, feature_names):
    _print_cartinfo(prefix, info.to_json(feature_names=feature_names), feature_names)


def _print_cartinfo(prefix, model, feature_names):
    if model["type"] == "static":
        print(f"""{prefix}: {model["value"]}""")
    else:
        _print_cartinfo(
            f"""{prefix} {model["paramName"]}≤{model["paramDecisionValue"]} """,
            model["left"],
            feature_names,
        )
        _print_cartinfo(
            f"""{prefix} {model["paramName"]}>{model["paramDecisionValue"]} """,
            model["right"],
            feature_names,
        )


def print_splitinfo(param_names, info, prefix=""):
    if type(info) is df.SplitFunction:
        for k, v in info.child.items():
            if info.param_index < len(param_names):
                param_name = param_names[info.param_index]
            else:
                param_name = f"arg{info.param_index - len(param_names)}"
            print_splitinfo(param_names, v, f"{prefix} {param_name}={k}")
    elif type(info) is df.AnalyticFunction:
        print_analyticinfo(prefix, info)
    elif type(info) is df.StaticFunction:
        print(f"{prefix}: {info.value}")
    else:
        print(f"{prefix}: UNKNOWN")


def print_model(prefix, info, feature_names):
    if type(info) is df.StaticFunction:
        print_staticinfo(prefix, info)
    elif type(info) is df.AnalyticFunction:
        print_analyticinfo(prefix, info)
    elif type(info) is df.FOLFunction:
        print_analyticinfo(prefix, info)
    elif type(info) is df.CARTFunction:
        print_cartinfo(prefix, info, feature_names)
    elif type(info) is df.SplitFunction:
        print_splitinfo(feature_names, info, prefix)
    else:
        print(f"{prefix}: {type(info)} UNIMPLEMENTED")


def print_model_complexity(model):
    key_len = len("Key")
    attr_len = len("Attribute")
    for name in model.names:
        if len(name) > key_len:
            key_len = len(name)
        for attr in model.attributes(name):
            if len(attr) > attr_len:
                attr_len = len(attr)
    for name in model.names:
        for attribute in model.attributes(name):
            mf = model.attr_by_name[name][attribute].model_function
            prefix = f"{name:{key_len}s} {attribute:{attr_len}s}: {mf.get_complexity_score():7d}"
            try:
                num_nodes = mf.get_number_of_nodes()
                max_depth = mf.get_max_depth()
                print(f"{prefix}  ({num_nodes:6d} nodes @ {max_depth:3d} max depth)")
            except AttributeError:
                print(prefix)


def format_quality_measures(result, error_metric="smape", col_len=8):
    if error_metric in result and result[error_metric] is not np.nan:
        if error_metric.endswith("pe"):
            unit = "%"
        else:
            unit = " "
        return f"{result[error_metric]:{col_len-1}.2f}{unit}"
    else:
        return f"""{result["mae"]:{col_len-1}.0f} """


def model_quality_table(
    lut, model, static, model_info, xv_method=None, xv_count=None, error_metric="smape"
):
    key_len = len("Key")
    attr_len = len("Attribute")
    for key in static.keys():
        if len(key) > key_len:
            key_len = len(key)
        for attr in static[key].keys():
            if len(attr) > attr_len:
                attr_len = len(attr)

    if xv_method == "kfold":
        xv_header = "kfold XV"
    elif xv_method == "montecarlo":
        xv_header = "MC XV"
    elif xv_method:
        xv_header = "XV"
    else:
        xv_header = "training"

    if xv_method is not None:
        print(
            f"Model error ({error_metric}) after cross validation ({xv_method}, {xv_count}):"
        )
    else:
        print(f"Model error ({error_metric}) on training data:")

    print(
        f"""{"":>{key_len}s} {"":>{attr_len}s}   {"training":>8s}   {xv_header:>8s}   {xv_header:>8s}"""
    )
    print(
        f"""{"Key":>{key_len}s} {"Attribute":>{attr_len}s}   {"LUT":>8s}   {"model":>8s}   {"static":>8s}"""
    )
    for key in sorted(static.keys()):
        for attr in sorted(static[key].keys()):
            buf = f"{key:>{key_len}s} {attr:>{attr_len}s}"
            for results, info in ((lut, None), (model, model_info), (static, None)):
                buf += "   "
                if results is not None and (
                    info is None
                    or (
                        key != "energy_Pt"
                        and type(info(key, attr)) is not df.StaticFunction
                    )
                    or (
                        key == "energy_Pt"
                        and (
                            type(info(key, "power")) is not df.StaticFunction
                            or type(info(key, "duration")) is not df.StaticFunction
                        )
                    )
                ):
                    result = results[key][attr]
                    buf += format_quality_measures(result, error_metric=error_metric)
                else:
                    buf += f"""{"----":>7s} """
            if type(model_info(key, attr)) is not df.StaticFunction:
                if model[key][attr]["mae"] > static[key][attr]["mae"]:
                    buf += "  :-("
                elif (
                    lut is not None
                    and model[key][attr]["mae"] <= 2 * lut[key][attr]["mae"]
                    and static[key][attr]["mae"] > 4 * lut[key][attr]["mae"]
                ):
                    buf += "  :-D"
                elif (
                    lut is not None
                    and static[key][attr]["mae"] - model[key][attr]["mae"]
                    > model[key][attr]["mae"] - lut[key][attr]["mae"]
                    and static[key][attr]["mae"] > 1.1 * lut[key][attr]["mae"]
                ):
                    buf += "  :-)"
            print(buf)


def export_dataref(dref_file, dref, precision=None):
    with open(dref_file, "w") as f:
        for k, v in sorted(dref.items()):
            if type(v) is not tuple:
                v = (v, None)
            if v[1] is None:
                prefix = r"\drefset{"
            else:
                prefix = r"\drefset" + f"[unit={v[1]}]" + "{"
            if type(v[0]) in (float, np.float64) and precision is not None:
                print(f"{prefix}/{k}" + "}{" + f"{v[0]:.{precision}f}" + "}", file=f)
            else:
                print(f"{prefix}/{k}" + "}{" + str(v[0]) + "}", file=f)


def export_dot(model, dot_prefix):
    for name in model.names:
        for attribute in model.attributes(name):
            dot_model = model.attr_by_name[name][attribute].to_dot()
            if dot_model is None:
                logger.debug(f"{name} {attribute} does not have a dot model")
            elif type(dot_model) is list:
                # A Forest
                for i, tree in enumerate(dot_model):
                    filename = f"{dot_prefix}{name}-{attribute}.{i:03d}.dot"
                    with open(filename, "w") as f:
                        print(tree, file=f)
                filename = filename.replace(f".{len(dot_model)-1:03d}.", ".*.")
                logger.info(f"Dot exports of model saved to {filename}")
            else:
                filename = f"{dot_prefix}{name}-{attribute}.dot"
                with open(filename, "w") as f:
                    print(dot_model, file=f)
                logger.info(f"Dot export of model saved to {filename}")


def export_pgf_unparam(model, pgf_prefix):
    for name in model.names:
        for attribute in model.attributes(name):
            filename = f"{pgf_prefix}{name}-{attribute}.txt"
            with open(filename, "w") as f:
                print(
                    "measurement value "
                    + " ".join(model.parameters)
                    + " "
                    + " ".join(
                        map(lambda x: f"arg{x}", range(model._num_args.get(name, 0)))
                    ),
                    file=f,
                )
                for i, value in enumerate(model.attr_by_name[name][attribute].data):
                    parameters = list()
                    for param in model.attr_by_name[name][attribute].param_values[i]:
                        if param is None:
                            parameters.append("{}")
                        else:
                            parameters.append(str(param))
                    parameters = " ".join(parameters)
                    print(f"{i} {value} {parameters}", file=f)
            logger.info(f"PGF unparam data saved to {filename}")


def export_json_unparam(model, filename):
    import json
    from dfatool.utils import NpEncoder

    ret = {"paramNames": model.parameters, "byName": dict()}
    for name in model.names:
        ret["byName"][name] = dict()
        for attribute in model.attributes(name):
            ret["byName"][name][attribute] = {
                "paramValues": model.attr_by_name[name][attribute].param_values,
                "data": model.attr_by_name[name][attribute].data,
            }
    with open(filename, "w") as f:
        json.dump(ret, f, cls=NpEncoder)
    logger.info(f"JSON unparam data saved to {filename}")


def boxplot_param(args, model):
    title = None
    param_is_filtered = dict()
    if args.filter_param:
        title = "filter: " + ", ".join(
            map(lambda kv: f"{kv[0]}={kv[1]}", args.filter_param)
        )
        for param_name, _ in args.filter_param:
            param_is_filtered[param_name] = True
    by_param = model.get_by_param()
    for name in model.names:
        attr_names = sorted(model.attributes(name))
        param_keys = list(
            map(lambda kv: kv[1], filter(lambda kv: kv[0] == name, by_param.keys()))
        )
        param_desc = list(
            map(
                lambda param_key: ", ".join(
                    map(
                        lambda ip: f"{model.param_name(ip[0])}={ip[1]}",
                        filter(
                            lambda ip: model.param_name(ip[0]) not in param_is_filtered,
                            enumerate(param_key),
                        ),
                    )
                ),
                param_keys,
            )
        )
        for attribute in attr_names:
            dfatool.plotter.boxplot(
                param_desc,
                list(map(lambda k: by_param[(name, k)][attribute], param_keys)),
                output=f"{args.boxplot_param}{name}-{attribute}.pdf",
                title=title,
                ylabel=attribute,
                show=not args.non_interactive,
            )


def add_standard_arguments(parser):
    parser.add_argument(
        "--export-dot",
        metavar="PREFIX",
        type=str,
        help="Export tree-based model to {PREFIX}{name}-{attribute}.dot",
    )
    parser.add_argument(
        "--export-dref",
        metavar="FILE",
        type=str,
        help="Export model and model quality to LaTeX dataref file",
    )
    parser.add_argument(
        "--export-pgf-unparam",
        metavar="PREFIX",
        type=str,
        help="Export raw (parameter-independent) observations in tikz-pgf-compatible format to {PREFIX}{name}-{attribute}.txt",
    )
    parser.add_argument(
        "--export-json-unparam",
        metavar="FILENAME",
        type=str,
        help="Export raw (parameter-independent) observations in JSON format to FILENAME",
    )
    parser.add_argument(
        "--export-json",
        metavar="FILENAME",
        type=str,
        help="Export model in JSON format to FILENAME",
    )
    parser.add_argument(
        "--dref-precision",
        metavar="NDIG",
        type=int,
        help="Limit precision of dataref export to NDIG decimals",
    )
    parser.add_argument(
        "--plot-unparam",
        metavar="<name>:<attribute>:<Y axis label>[;<name>:<attribute>:<label>;...]",
        type=str,
        help="Plot all mesurements for <name> <attribute> without regard for parameter values. "
        "X axis is measurement number/id.",
    )
    parser.add_argument(
        "--plot-param",
        metavar="<name>:<attribute>:<parameter>[;<name>:<attribute>:<parameter>;...])",
        type=str,
        help="Plot measurements for <name> <attribute> by <parameter>. "
        "X axis is parameter value. "
        "Plots the model function as one solid line for each combination of non-<parameter> parameters. "
        "Also plots the corresponding measurements. ",
    )
    parser.add_argument(
        "--boxplot-unparam",
        metavar="PREFIX",
        type=str,
        help="Export boxplots of raw (parameter-independent) observations to {PREFIX}{name}-{attribute}.pdf",
    )
    parser.add_argument(
        "--boxplot-param",
        metavar="PREFIX",
        type=str,
        help="Export boxplots of observations to {PREFIX}{name}-{attribute}.pdf, with one boxplot per parameter combination",
    )
    parser.add_argument(
        "--non-interactive", action="store_true", help="Do not show interactive plots"
    )
    parser.add_argument(
        "--export-xv",
        metavar="FILE",
        type=str,
        help="Export raw cross-validation results to FILE for later analysis (e.g. to compare different modeling approaches by means of a t-test)",
    )
    parser.add_argument(
        "--export-raw-predictions",
        metavar="FILE",
        type=str,
        help="Export raw model error data (i.e., ground truth vs. model output) to FILE for later analysis (e.g. to compare different modeling approaches by means of a t-test)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show benchmark information (number of measurements, parameter values, ...)",
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        choices=["debug", "info", "warning", "error"],
        default="warning",
        help="Set log level",
    )
    parser.add_argument(
        "--show-model",
        choices=["static", "paramdetection", "param", "all"],
        action="append",
        default=list(),
        help="static: show static model values as well as parameter detection heuristic.\n"
        "paramdetection: show stddev of static/lut/fitted model\n"
        "param: show parameterized model functions and regression variable values\n"
        "all: all of the above",
    )
    parser.add_argument(
        "--show-model-error",
        action="store_true",
        help="Show model error compared to LUT (lower bound) and static (reference) models",
    )
    parser.add_argument(
        "--show-model-complexity",
        action="store_true",
        help="Show model complexity score and details (e.g. regression tree height and node count)",
    )
    parser.add_argument(
        "--cross-validate",
        metavar="<method>:<count>",
        type=str,
        help="Perform cross validation when computing model quality",
    )
    parser.add_argument(
        "--parameter-aware-cross-validation",
        action="store_true",
        help="Perform parameter-aware cross-validation: ensure that parameter values (and not just observations) are mutually exclusive between training and validation sets.",
    )
    parser.add_argument(
        "--param-shift",
        metavar="<key>=<+|-|*|/><value>|none-to-0;...",
        type=str,
        help="Adjust parameter values before passing them to model generation",
    )
    parser.add_argument(
        "--normalize-nfp",
        metavar="<newkey>=<oldkey>=<+|-|*|/><value>|none-to-0;...",
        type=str,
        help="Normalize observation values before passing them to model generation",
    )
    parser.add_argument(
        "--filter-param",
        metavar="<parameter name>=<parameter value>[,<parameter name>=<parameter value>...]",
        type=str,
        help="Only consider measurements where <parameter name> is <parameter value>. "
        "All other measurements (including those where it is None, that is, has not been set yet) are discarded. "
        "Note that this may remove entire function calls from the model.",
    )
    parser.add_argument(
        "--ignore-param",
        metavar="<parameter name>[,<parameter name>,...]",
        type=str,
        help="Ignore listed parameters during model generation",
    )
    parser.add_argument(
        "--function-override",
        metavar="<name> <attribute> <function>[;<name> <attribute> <function>;...]",
        type=str,
        help="Manually specify the function to fit for <name> <attribute>. "
        "A function specified this way bypasses parameter detection: "
        "It is always assigned, even if the model seems to be independent of the parameters it references.",
    )
    parser.add_argument(
        "--error-metric",
        metavar="METRIC",
        choices=[
            "mae",
            "mape",
            "smape",
            "p50",
            "p90",
            "p95",
            "p99",
            "msd",
            "rmsd",
            "ssr",
            "rsq",
        ],
        default="smape",
        help="Error metric to use in --show-quality reports. In case a metric is undefined for a particular set of ground truth and prediction entries, dfatool falls back to mae.\n"
        "MAE    : Mean Absolute Error\n"
        "MAPE   : Mean Absolute Percentage Error\n"
        "SMAPE  : Symmetric Mean Absolute Percentage Error\n"
        "p50    : Median (50th Percentile) Absolute Error\n"
        "p90    : 90th Percentile Absolute Error\n"
        "p95    : 95th Percentile Absolute Error\n"
        "p99    : 99th Percentile Absolute Error\n"
        "msd    : Mean Square Deviation\n"
        "rmsd   : Root Mean Square Deviation\n"
        "ssr    : Sum of Squared Residuals\n"
        "rsq    : R² Score",
    )


def parse_shift_function(param_name, param_shift):
    if param_shift.startswith("+"):
        param_shift_value = float(param_shift[1:])
        return lambda p: p + param_shift_value
    elif param_shift.startswith("-"):
        param_shift_value = float(param_shift[1:])
        return lambda p: p - param_shift_value
    elif param_shift.startswith("*"):
        param_shift_value = float(param_shift[1:])
        return lambda p: p * param_shift_value
    elif param_shift.startswith("/"):
        param_shift_value = float(param_shift[1:])
        return lambda p: p / param_shift_value
    elif param_shift == "none-to-0":
        return lambda p: p or 0
    else:
        raise ValueError(f"Unsupported shift operation {param_name}={param_shift}")


def parse_nfp_normalization(raw_normalization):
    norm_list = list()
    for norm_pair in raw_normalization.split(";"):
        new_name, old_name, norm_val = norm_pair.split("=")
        norm_function = parse_shift_function(new_name, norm_val)
        norm_list.append((new_name, old_name, norm_function))
    return norm_list


def parse_param_shift(raw_param_shift):
    shift_list = list()
    for shift_pair in raw_param_shift.split(";"):
        param_name, param_shift = shift_pair.split("=")
        param_shift_function = parse_shift_function(param_name, param_shift)
        shift_list.append((param_name, param_shift_function))
    return shift_list
