#!/usr/bin/env python3

from dfatool.functions import (
    SplitFunction,
    AnalyticFunction,
    StaticFunction,
)


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
