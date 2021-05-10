#!/usr/bin/env python3
"""
analyze-config - generate NFP model from system config benchmarks

analyze-config generates an NFP model from benchmarks with various system
configs (.config entries generated from a common Kconfig definition). The
NFP model is not yet compatible with the type of models generated
by analyze-archive and analyze-timing
"""

import argparse
import json
import kconfiglib
import logging
import os

import numpy as np

from dfatool.functions import SplitFunction, StaticFunction
from dfatool.model import AnalyticModel, ModelAttribute
from dfatool.utils import NpEncoder


def make_config_vector(kconf, params, symbols, choices):
    config_vector = [None for i in params]
    for i, param in enumerate(params):
        if param in choices:
            choice = kconf.choices[choices.index(param)]
            if choice.selection:
                config_vector[i] = choice.selection.name
        else:
            config_vector[i] = kconf.syms[param].str_value
    return tuple(config_vector)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        choices=["debug", "info", "warning", "error"],
        default="warning",
        help="Set log level",
    )
    parser.add_argument(
        "--with-choice-node",
        action="store_true",
        help="Add special decisiontree nodes for Kconfig choice symbols",
    )
    parser.add_argument("kconfig_file")
    parser.add_argument("data_dir")

    args = parser.parse_args()

    if args.log_level:
        numeric_level = getattr(logging, args.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            print(f"Invalid log level: {args.log_level}", file=sys.stderr)
            sys.exit(1)
        logging.basicConfig(level=numeric_level)

    experiments = list()

    for direntry in os.listdir(args.data_dir):
        if "Multipass" in direntry:
            config_path = f"{args.data_dir}/{direntry}/.config"
            attr_path = f"{args.data_dir}/{direntry}/attributes.json"
            if os.path.exists(attr_path):
                experiments.append((config_path, attr_path))

    kconf = kconfiglib.Kconfig(args.kconfig_file)

    # TODO Optional neben bool auch choices unterstützen.
    # Später ebenfalls int u.ä. -> dfatool-modeling

    symbols = sorted(
        map(
            lambda sym: sym.name,
            filter(
                lambda sym: kconfiglib.TYPE_TO_STR[sym.type] == "bool",
                kconf.syms.values(),
            ),
        )
    )

    if args.with_choice_node:
        choices = list(map(lambda choice: choice.name, kconf.choices))
    else:
        choices = list()

    params = sorted(symbols + choices)

    by_name = {
        "multipass": {
            "isa": "state",
            "attributes": ["rom_usage", "ram_usage"],
            "rom_usage": list(),
            "ram_usage": list(),
            "param": list(),
        }
    }
    data = list()

    config_vectors = set()

    for config_path, attr_path in experiments:
        kconf.load_config(config_path)
        with open(attr_path, "r") as f:
            attr = json.load(f)

        config_vector = make_config_vector(kconf, params, symbols, choices)

        config_vectors.add(config_vector)
        by_name["multipass"]["rom_usage"].append(attr["total"]["ROM"])
        by_name["multipass"]["ram_usage"].append(attr["total"]["RAM"])
        by_name["multipass"]["param"].append(config_vector)
        data.append((config_vector, attr["total"]["ROM"], attr["total"]["RAM"]))

    print(
        "Processing {:d} unique configurations of {:d} total".format(
            len(config_vectors), len(experiments)
        )
    )

    print(
        "std of all data: {:5.0f} Bytes".format(np.std(list(map(lambda x: x[1], data))))
    )

    model = AnalyticModel(by_name, params, compute_stats=False)

    def get_min(this_symbols, this_data, data_index=1, threshold=100, level=0):

        rom_sizes = list(map(lambda x: x[data_index], this_data))

        if np.std(rom_sizes) < threshold or len(this_symbols) == 0:
            return StaticFunction(np.mean(rom_sizes))
            # sf.value_error["std"] = np.std(rom_sizes)

        mean_stds = list()
        for i, param in enumerate(this_symbols):

            unique_values = list(set(map(lambda vrr: vrr[0][i], this_data)))

            child_values = list()
            for value in unique_values:
                child_values.append(
                    list(filter(lambda vrr: vrr[0][i] == value, this_data))
                )

            if len(list(filter(len, child_values))) < 2:
                # this param only has a single value. there's no point in splitting.
                mean_stds.append(np.inf)
                continue

            children = list()
            for child in child_values:
                children.append(np.std(list(map(lambda x: x[1], child))))

            if np.any(np.isnan(children)):
                mean_stds.append(np.inf)
            else:
                mean_stds.append(np.mean(children))

        if np.all(np.isinf(mean_stds)):
            # all children have the same configuration. We shouldn't get here due to the threshold check above...
            logging.warning("Waht")
            return StaticFunction(np.mean(rom_sizes))

        symbol_index = np.argmin(mean_stds)
        symbol = this_symbols[symbol_index]
        new_symbols = this_symbols[:symbol_index] + this_symbols[symbol_index + 1 :]

        unique_values = list(set(map(lambda vrr: vrr[0][symbol_index], this_data)))

        child = dict()

        for value in unique_values:
            children = filter(lambda vrr: vrr[0][symbol_index] == value, this_data)
            children = list(
                map(
                    lambda x: (x[0][:symbol_index] + x[0][symbol_index + 1 :], *x[1:]),
                    children,
                )
            )
            if len(children):
                print(
                    f"Level {level} split on {symbol} == {value} has {len(children)} children"
                )
                child[value] = get_min(
                    new_symbols, children, data_index, threshold, level + 1
                )

        assert len(child.values()) >= 2

        return SplitFunction(np.mean(rom_sizes), symbol_index, child)

    model.attr_by_name["multipass"] = dict()
    model.attr_by_name["multipass"]["rom_usage"] = ModelAttribute(
        "multipass",
        "rom_usage",
        by_name["multipass"]["rom_usage"],
        by_name["multipass"]["param"],
        params,
    )
    model.attr_by_name["multipass"]["ram_usage"] = ModelAttribute(
        "multipass",
        "rom_usage",
        by_name["multipass"]["ram_usage"],
        by_name["multipass"]["param"],
        params,
    )

    model.attr_by_name["multipass"]["rom_usage"].model_function = get_min(
        params, data, 1, 100
    )
    model.attr_by_name["multipass"]["ram_usage"].model_function = get_min(
        params, data, 2, 20
    )

    with open("kconfigmodel.json", "w") as f:
        json_model = model.to_json(with_param_name=True, param_names=params)
        json.dump(json_model, f, sort_keys=True, cls=NpEncoder)


if __name__ == "__main__":
    main()
