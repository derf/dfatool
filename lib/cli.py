#!/usr/bin/env python3

import dfatool.functions as df
import logging
import numpy as np
import os
import sys

logger = logging.getLogger(__name__)


def sanity_check(args):
    pass


def print_static(
    model, static_model, name, attribute, with_dependence=False, precision=2
):
    if precision is None:
        precision = 6
    unit = "  "
    if attribute == "power":
        unit = "µW"
    elif attribute == "duration":
        unit = "µs"
    elif attribute == "substate_count":
        unit = "su"
    if model.attr_by_name[name][attribute].stats:
        ratio = model.attr_by_name[name][
            attribute
        ].stats.generic_param_dependence_ratio()
        print(
            f"{name:10s}: {attribute:28s} : {static_model(name, attribute):.{precision}f} {unit:s}  ({ratio:.2f})"
        )
    else:
        print(
            f"{name:10s}: {attribute:28s} : {static_model(name, attribute):.{precision}f} {unit:s}"
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


def print_information_gain_by_name(model, by_name):
    for name in model.names:
        for attr in model.attributes(name):
            print(f"{name} {attr}:")
            mutual_information = model.mutual_information(name, attr)
            for param in model.parameters:
                if param in mutual_information:
                    print(f"    Parameter {param} : {mutual_information[param]:5.2f}")
                else:
                    print(f"    Parameter {param} :  -.--")


def print_analyticinfo(prefix, info, ndigits=None):
    model_function = info.model_function.removeprefix("0 + ")
    for i in range(len(info.model_args)):
        if ndigits is not None:
            model_function = model_function.replace(
                f"regression_arg({i})", str(round(info.model_args[i], ndigits=ndigits))
            )
        else:
            model_function = model_function.replace(
                f"regression_arg({i})", str(info.model_args[i])
            )
    model_function = model_function.replace("+ -", "- ")
    print(f"{prefix}: {model_function}")


def print_staticinfo(prefix, info, ndigits=None):
    if ndigits is not None:
        print(f"{prefix}: {round(info.value, ndigits)}")
    else:
        print(f"{prefix}: {info.value}")


def print_symreginfo(prefix, info):
    print(f"{prefix}: {str(info.regressor)}")


def print_cartinfo(prefix, info):
    _print_cartinfo(prefix, info.to_json())


def print_xgbinfo(prefix, info):
    for i, tree in enumerate(info.to_json()):
        _print_cartinfo(prefix + f"tree{i:03d} :", tree)


def print_lmtinfo(prefix, info):
    _print_lmtinfo(prefix, info.to_json())


def _print_lmtinfo(prefix, model):
    if model["type"] == "static":
        print(f"""{prefix}: {model["value"]}""")
    elif model["type"] == "scalarSplit":
        _print_lmtinfo(
            f"""{prefix} {model["paramName"]}≤{model["threshold"]} """,
            model["left"],
        )
        _print_lmtinfo(
            f"""{prefix} {model["paramName"]}>{model["threshold"]} """,
            model["right"],
        )
    else:
        model_function = model["functionStr"].removeprefix("0 + ")
        for i, coef in enumerate(model["regressionModel"]):
            model_function = model_function.replace(f"regression_arg({i})", str(coef))
        model_function = model_function.replace("+ -", "- ")
        print(f"{prefix}: {model_function}")


def _print_cartinfo(prefix, model):
    if model["type"] == "static":
        print(f"""{prefix}: {model["value"]}""")
    else:
        _print_cartinfo(
            f"""{prefix} {model["paramName"]}≤{model["threshold"]} """,
            model["left"],
        )
        _print_cartinfo(
            f"""{prefix} {model["paramName"]}>{model["threshold"]} """,
            model["right"],
        )


def print_splitinfo(info, prefix=""):
    if type(info) is df.SplitFunction:
        for k, v in sorted(info.child.items()):
            print_splitinfo(v, f"{prefix} {info.param_name}={k}")
    elif type(info) is df.ScalarSplitFunction:
        print_splitinfo(info.child_le, f"{prefix} {info.param_name}≤{info.threshold}")
        print_splitinfo(info.child_gt, f"{prefix} {info.param_name}>{info.threshold}")
    elif type(info) is df.AnalyticFunction:
        print_analyticinfo(prefix, info)
    elif type(info) is df.SymbolicRegressionFunction:
        print_symreginfo(prefix, info)
    elif type(info) is df.StaticFunction:
        print(f"{prefix}: {info.value}")
    else:
        print(f"{prefix}: UNKNOWN {type(info)}")


def print_model(prefix, info, precision=None):
    if type(info) is df.StaticFunction:
        print_staticinfo(prefix, info, ndigits=precision)
    elif type(info) is df.AnalyticFunction:
        print_analyticinfo(prefix, info, ndigits=precision)
    elif type(info) is df.FOLFunction:
        print_analyticinfo(prefix, info, ndigits=precision)
    elif type(info) is df.CARTFunction:
        print_cartinfo(prefix, info)
    elif type(info) is df.SplitFunction:
        print_splitinfo(info, prefix)
    elif type(info) is df.ScalarSplitFunction:
        print_splitinfo(info, prefix)
    elif type(info) is df.LMTFunction:
        print_lmtinfo(prefix, info)
    elif type(info) is df.LightGBMFunction:
        print_xgbinfo(prefix, info)
    elif type(info) is df.XGBoostFunction:
        print_xgbinfo(prefix, info)
    elif type(info) is df.SymbolicRegressionFunction:
        print_symreginfo(prefix, info)
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
    for name in sorted(model.names):
        for attribute in sorted(model.attributes(name)):
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
    lut,
    model,
    static,
    model_info,
    xv_method=None,
    xv_count=None,
    error_metric="smape",
    load_model=False,
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
    elif load_model:
        xv_header = "json"
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

                # special case for "TOTAL" (--add-total-observation)
                if attr == "TOTAL" and attr not in results[key]:
                    buf += f"""{"----":>7s} """
                elif results is not None and (
                    info is None
                    or (
                        attr != "energy_Pt"
                        and type(info(key, attr)) is not df.StaticFunction
                    )
                    or (
                        attr == "energy_Pt"
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
            if attr != "TOTAL" and type(model_info(key, attr)) is not df.StaticFunction:
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
        for k, v in sorted(os.environ.items(), key=lambda kv: kv[0]):
            if k.startswith("DFATOOL_"):
                print(f"% {k}='{v}'", file=f)
        for arg in sys.argv:
            print(f"% {arg}", file=f)
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


def export_csv_unparam(model, csv_prefix, dialect="excel"):
    import csv

    class ExcelLF(csv.Dialect):
        delimiter = ","
        quotechar = '"'
        doublequote = True
        skipinitialspace = False
        lineterminator = "\n"
        quoting = 0

    csv.register_dialect("excel-lf", ExcelLF)

    for name in sorted(model.names):
        filename = f"{csv_prefix}{name}.csv"
        with open(filename, "w") as f:
            writer = csv.writer(f, dialect=dialect)
            writer.writerow(
                ["measurement"] + model.parameters + sorted(model.attributes(name))
            )
            for i, param_tuple in enumerate(model.param_values(name)):
                row = [i] + param_tuple
                for attr in sorted(model.attributes(name)):
                    row.append(model.attr_by_name[name][attr].data[i])
                writer.writerow(row)
        logger.info(f"CSV unparam data saved to {filename}")


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
    import dfatool.plotter as dp

    title = None
    param_is_filtered = dict()
    if args.filter_param:
        title = "filter: " + " && ".join(
            map(lambda kv: f"{kv[0]} {kv[1]} {kv[2]}", args.filter_param)
        )
        for param_name, _, _ in args.filter_param:
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
            dp.boxplot(
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
        "--export-csv-unparam",
        metavar="PREFIX",
        type=str,
        help="Export raw (parameter-independent) observations in CSV format to {PREFIX}{name}-{attribute}.csv",
    )
    parser.add_argument(
        "--export-csv-dialect",
        metavar="DIALECT",
        type=str,
        choices=["excel", "excel-lf", "excel-tab", "unix"],
        default="excel",
        help="CSV dialect to use for --export-csv-unparam",
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
        "--load-json",
        metavar="FILENAME",
        type=str,
        help="Load model in JSON format from FILENAME",
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
        "--information-gain",
        action="store_true",
        help="Show information gain of parameters",
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
        "--show-model-precision",
        metavar="NDIG",
        type=int,
        default=2,
        help="Limit precision of model output to NDIG decimals",
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
        "--add-total-observation",
        action="store_true",
        help="Add a TOTAL observation for each <key> that consists of the sums of its <attribute> entries. This allows for cross-validation of behaviour models vs. non-behaviour-aware models.",
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
        metavar="<key>=<+|-|*|/><value>|none-to-0|categorical;...",
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
        metavar="<parameter name><condition>[;<parameter name><condition>...]",
        type=str,
        help="Only consider measurements where <parameter name> satisfies <condition>. "
        "<condition> may be <operator><parameter value> with operator being < / <= / = / >= / >, "
        "or ∈<parameter value>[,<parameter value>...]. "
        "All other measurements (including those where it is None, that is, has not been set yet) are discarded. "
        "Note that this may remove entire function calls from the model.",
    )
    parser.add_argument(
        "--filter-observation",
        metavar="<key>:<attribute>[,<key>:<attribute>...]",
        type=str,
        help="Only consider measurements of <key> <attribute>",
    )
    parser.add_argument(
        "--ignore-param",
        metavar="<parameter name>[,<parameter name>,...]",
        type=str,
        help="Ignore listed parameters during model generation",
    )
    parser.add_argument(
        "--function-override",
        metavar="<name>:<attribute>:<function>[;<name>:<attribute>:<function>;...]",
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
    parser.add_argument(
        "--skip-param-stats",
        action="store_true",
        help="Do not compute param stats that are required for RMT. Use this for high-dimensional feature spaces.",
    )
    parser.add_argument(
        "--force-tree",
        action="store_true",
        help="Build regression tree without checking whether static/analytic functions are sufficient.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars while executing compute-intensive tasks such as cross-validation.",
    )


def parse_filter_string(filter_string, parameter_names=None):
    if "<=" in filter_string:
        p, v = filter_string.split("<=")
        return (p, "≤", v)
    if ">=" in filter_string:
        p, v = filter_string.split(">=")
        return (p, "≥", v)
    if "!=" in filter_string:
        p, v = filter_string.split("!=")
        if parameter_names is None or p in parameter_names:
            return (p, "≠", v)
        # otherwise, '!' belongs to the parameter name and is not part of the condition.
    for op in ("<", ">", "≤", "≥", "=", "∈", "≠"):
        if op in filter_string:
            p, v = filter_string.split(op)
            return (p, op, v)
    raise ValueError(f"Cannot parse '{filter_string}'")


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
    elif param_shift == "categorical":
        return lambda p: "=" + str(p)
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
