#!/usr/bin/env python3
"""
analyze-config - generate NFP model from system config benchmarks

analyze-connfig generates an NFP model from benchmarks with various system
configs (.config entries generated from a common Kconfig definition).
"""

import argparse
import json
import kconfiglib
import logging
import os

import numpy as np

from dfatool.model import AnalyticModel


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

    symbols = sorted(
        map(
            lambda sym: sym.name,
            filter(
                lambda sym: kconfiglib.TYPE_TO_STR[sym.type] == "bool",
                kconf.syms.values(),
            ),
        )
    )

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

        config_vector = tuple(map(lambda sym: kconf.syms[sym].tri_value == 2, symbols))
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

    model = AnalyticModel(by_name, symbols, compute_stats=False)

    class DTreeLeaf:
        def __init__(self, value, stddev):
            self.value = value
            self.stddev = stddev

        def __repr__(self):
            return f"<DTreeLeaf({self.value}, {self.stddev})>"

        def to_json(self):
            return {"value": self.value, "stddev": self.stddev}

    class DTreeNode:
        def __init__(self, symbol):
            self.symbol = symbol
            self.false_child = None
            self.true_child = None

        def set_false_child(self, child_node):
            self.false_child = child_node

        def set_true_child(self, child_node):
            self.true_child = child_node

        def __repr__(self):
            return f"<DTreeNode({self.false_child}, {self.true_child})>"

        def to_json(self):
            ret = {"symbol": self.symbol}
            if self.false_child:
                ret["false"] = self.false_child.to_json()
            else:
                ret["false"] = None
            if self.true_child:
                ret["true"] = self.true_child.to_json()
            else:
                ret["true"] = None
            return ret

    def get_min(this_symbols, this_data, level):

        rom_sizes = list(map(lambda x: x[1], this_data))

        if np.std(rom_sizes) < 100 or len(this_symbols) == 0:
            return DTreeLeaf(np.mean(rom_sizes), np.std(rom_sizes))

        mean_stds = list()
        for i, param in enumerate(this_symbols):
            enabled = list(filter(lambda vrr: vrr[0][i] == True, this_data))
            disabled = list(filter(lambda vrr: vrr[0][i] == False, this_data))

            enabled_std_rom = np.std(list(map(lambda x: x[1], enabled)))
            disabled_std_rom = np.std(list(map(lambda x: x[1], disabled)))
            children = [enabled_std_rom, disabled_std_rom]

            if np.any(np.isnan(children)):
                mean_stds.append(np.inf)
            else:
                mean_stds.append(np.mean(children))

        symbol_index = np.argmin(mean_stds)
        symbol = this_symbols[symbol_index]
        enabled = list(filter(lambda vrr: vrr[0][symbol_index] == True, this_data))
        disabled = list(filter(lambda vrr: vrr[0][symbol_index] == False, this_data))

        node = DTreeNode(symbol)

        new_symbols = this_symbols[:symbol_index] + this_symbols[symbol_index + 1 :]
        enabled = list(
            map(
                lambda x: (x[0][:symbol_index] + x[0][symbol_index + 1 :], *x[1:]),
                enabled,
            )
        )
        disabled = list(
            map(
                lambda x: (x[0][:symbol_index] + x[0][symbol_index + 1 :], *x[1:]),
                disabled,
            )
        )
        print(
            f"Level {level} split on {symbol} has {len(enabled)} children when enabled and {len(disabled)} children when disabled"
        )
        if len(enabled):
            node.set_true_child(get_min(new_symbols, enabled, level + 1))
        if len(disabled):
            node.set_false_child(get_min(new_symbols, disabled, level + 1))

        return node

    model = get_min(symbols, data, 0)

    output = {"model": model.to_json(), "symbols": symbols}

    with open("kconfigmodel.json", "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
