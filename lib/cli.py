#!/usr/bin/env python3


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
