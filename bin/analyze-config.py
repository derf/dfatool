#!/usr/bin/env python3

#  _                                   ____          _
# | |    ___  __ _  __ _  ___ _   _   / ___|___   __| | ___
# | |   / _ \/ _` |/ _` |/ __| | | | | |   / _ \ / _` |/ _ \
# | |__|  __/ (_| | (_| | (__| |_| | | |__| (_) | (_| |  __/
# |_____\___|\__, |\__,_|\___|\__, |  \____\___/ \__,_|\___|
#            |___/            |___/


"""
analyze-config - generate NFP model from system config benchmarks

analyze-config generates an NFP model from benchmarks with various system
configs (.config entries generated from a common Kconfig definition). The
NFP model is not yet compatible with the type of models generated
by analyze-archive and analyze-timing
"""

import argparse
import hashlib
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

    file_hash = hashlib.sha256()
    with open(args.kconfig_file, "rb") as f:
        kconfig_data = f.read()
        while len(kconfig_data) > 0:
            file_hash.update(kconfig_data)
            kconfig_data = f.read()

    kconfig_hash = str(file_hash.hexdigest())

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
        choices = sorted(map(lambda choice: choice.name, kconf.choices))
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
        data.append((config_vector, (attr["total"]["ROM"], attr["total"]["RAM"])))

    print(
        "Processing {:d} unique configurations of {:d} total".format(
            len(config_vectors), len(experiments)
        )
    )

    print(
        "std of all data: {:5.0f} Bytes".format(
            np.std(list(map(lambda x: x[1][0], data)))
        )
    )

    model = AnalyticModel(by_name, params, compute_stats=False)
    model.build_dtree("multipass", "rom_usage", 100)
    model.build_dtree("multipass", "ram_usage", 20)

    with open("kconfigmodel.json", "w") as f:
        json_model = model.to_json(with_param_name=True, param_names=params)
        json_model = json_model["name"]["multipass"]
        json_model["ram_usage"].update(
            {
                "unit": "B",
                "description": "RAM Usage",
                "minimize": True,
            }
        )
        json_model["rom_usage"].update(
            {
                "unit": "B",
                "description": "ROM Usage",
                "minimize": True,
            }
        )
        out_model = {
            "model": json_model,
            "modelType": "dfatool-kconfig",
            "kconfigHash": kconfig_hash,
            "project": "multipass",
            "symbols": symbols,
            "choices": choices,
        }
        json.dump(out_model, f, sort_keys=True, cls=NpEncoder)


if __name__ == "__main__":
    main()
