#!/usr/bin/env python3

from dfatool.functions import SplitFunction, AnalyticFunction, StaticFunction


def print_static(model, static_model, name, attribute):
    unit = "  "
    if attribute == "power":
        unit = "µW"
    elif attribute == "duration":
        unit = "µs"
    elif attribute == "substate_count":
        unit = "su"
    print(
        "{:10s}: {:.0f} {:s}  ({:.2f})".format(
            name,
            static_model(name, attribute),
            unit,
            model.attr_by_name[name][attribute].stats.generic_param_dependence_ratio(),
        )
    )
    for param in model.parameters:
        print(
            "{:10s}  dependence on {:15s}: {:.2f}".format(
                "",
                param,
                model.attr_by_name[name][attribute].stats.param_dependence_ratio(param),
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
    for state_or_tran in result_lists[0].keys():
        for key in result_lists[0][state_or_tran].keys():
            buf = "{:20s} {:15s}".format(state_or_tran, key)
            for i, results in enumerate(result_lists):
                info = info_list[i]
                buf += "  |||  "
                if (
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


def add_standard_arguments(parser):
    parser.add_argument(
        "--export-dref",
        metavar="FILE",
        type=str,
        help="Export model and model quality to LaTeX dataref file",
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
        metavar="<key>=<+|-|*|/><value>;...",
        type=str,
        help="Adjust parameter values before passing them to model generation",
    )


def parse_param_shift(raw_param_shift):
    shift_list = list()
    for shift_pair in raw_param_shift.split(";"):
        param_name, param_shift = shift_pair.split("=")
        if param_shift.startswith("+"):
            param_shift_value = float(param_shift[1:])
            param_shift_function = lambda p: p + param_shift_value
        elif param_shift.startswith("-"):
            param_shift_value = float(param_shift[1:])
            param_shift_function = lambda p: p - param_shift_value
        elif param_shift.startswith("*"):
            param_shift_value = float(param_shift[1:])
            param_shift_function = lambda p: p * param_shift_value
        elif param_shift.startswith("/"):
            param_shift_value = float(param_shift[1:])
            param_shift_function = lambda p: p / param_shift_value
        else:
            raise ValueError(f"Unsupported shift operation {param_name}={param_shift}")
        shift_list.append((param_name, param_shift_function))
    return shift_list
