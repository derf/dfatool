#!/usr/bin/env python3

from dfatool.functions import (
    SplitFunction,
    AnalyticFunction,
    StaticFunction,
    FOLFunction,
)
import numpy as np


def print_static(model, static_model, name, attribute):
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
    for param in model.parameters:
        print(
            "{:10s}  {:13s} {:15s}: {:.2f}".format(
                "",
                "dependence on",
                param,
                model.attr_by_name[name][attribute].stats.param_dependence_ratio(param),
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


def print_analyticinfo(prefix, info):
    empty = ""
    print(f"{prefix}: {info.model_function}")
    print(f"{empty:{len(prefix)}s}  {info.model_args}")


def print_splitinfo(param_names, info, prefix=""):
    if type(info) is SplitFunction:
        for k, v in info.child.items():
            if info.param_index < len(param_names):
                param_name = param_names[info.param_index]
            else:
                param_name = f"arg{info.param_index - len(param_names)}"
            print_splitinfo(param_names, v, f"{prefix} {param_name}={k}")
    elif type(info) is AnalyticFunction:
        print_analyticinfo(prefix, info)
    elif type(info) is StaticFunction:
        print(f"{prefix}: {info.value}")
    else:
        print(f"{prefix}: UNKNOWN")


def print_model_size(model):
    for name in model.names:
        for attribute in model.attributes(name):
            try:
                num_nodes = model.attr_by_name[name][
                    attribute
                ].model_function.get_number_of_nodes()
                max_depth = model.attr_by_name[name][
                    attribute
                ].model_function.get_max_depth()
                print(
                    f"{name:15s} {attribute:20s}: {num_nodes:6d} nodes @ {max_depth:3d} max depth"
                )
            except AttributeError:
                print(
                    f"{name:15s} {attribute:20s}: {model.attr_by_name[name][attribute].model_function}"
                )


def format_quality_measures(result):
    if "smape" in result:
        return "{:6.2f}% / {:9.0f}".format(result["smape"], result["mae"])
    else:
        return "{:6}    {:9.0f}".format("", result["mae"])


def model_quality_table(header, result_lists, info_list):
    print(
        "{:20s} {:15s}       {:19s}       {:19s}       {:19s}".format(
            "key",
            "attribute",
            header[0].center(19),
            header[1].center(19),
            header[2].center(19),
        )
    )
    for state_or_tran in sorted(result_lists[0].keys()):
        for key in sorted(result_lists[0][state_or_tran].keys()):
            buf = "{:20s} {:15s}".format(state_or_tran, key)
            for i, results in enumerate(result_lists):
                info = info_list[i]
                buf += "  |||  "
                if results is not None and (
                    info is None
                    or (
                        key != "energy_Pt"
                        and type(info(state_or_tran, key)) is not StaticFunction
                    )
                    or (
                        key == "energy_Pt"
                        and (
                            type(info(state_or_tran, "power")) is not StaticFunction
                            or type(info(state_or_tran, "duration"))
                            is not StaticFunction
                        )
                    )
                ):
                    result = results[state_or_tran][key]
                    buf += format_quality_measures(result)
                else:
                    buf += "{:7}----{:8}".format("", "")
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
            if not dot_model is None:
                with open(f"{dot_prefix}{name}-{attribute}.dot", "w") as f:
                    print(dot_model, file=f)


def export_pgf_unparam(model, pgf_prefix):
    for name in model.names:
        for attribute in model.attributes(name):
            with open(f"{pgf_prefix}{name}-{attribute}.txt", "w") as f:
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
        "--dref-precision",
        metavar="NDIG",
        type=int,
        help="Limit precision of dataref export to NDIG decimals",
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
        "--show-model-size",
        action="store_true",
        help="Show model size (e.g. regression tree height and node count)",
    )
    parser.add_argument(
        "--cross-validate",
        metavar="<method>:<count>",
        type=str,
        help="Perform cross validation when computing model quality. "
        "Only works with --show-quality=table at the moment.",
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
