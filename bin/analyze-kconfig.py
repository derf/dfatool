#!/usr/bin/env python3

"""analyze-kconfig - Generate a model for KConfig selections

analyze-kconfig builds a model determining system attributes
(e.g. ROM or RAM usage) based on KConfig configuration variables.
Only boolean variables are supported at the moment.
"""

import argparse
import json
import kconfiglib
import logging
import os

import numpy as np

import dfatool.utils
from dfatool.loader import KConfigAttributes
from dfatool.model import AnalyticModel


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "--show-failing-symbols",
        action="store_true",
        help="Show Kconfig symbols related to build failures. Must be used with an experiment result directory.",
    )
    parser.add_argument(
        "--show-nop-symbols",
        action="store_true",
        help="Show Kconfig symbols which are only present in a single configuration. Must be used with an experiment result directory.",
    )
    parser.add_argument(
        "--force-tree",
        action="store_true",
        help="Build decision tree without checking for analytic functions first. Use this for large kconfig files.",
    )
    parser.add_argument(
        "--export-tree",
        type=str,
        help="Export kconfig-webconf model to file",
        metavar="FILE",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Show model results for symbols in .config file",
        metavar="FILE",
    )
    parser.add_argument(
        "--log-level",
        default=logging.INFO,
        type=lambda level: getattr(logging, level.upper()),
        help="Set log level",
    )
    parser.add_argument(
        "--info", action="store_true", help="Show Kconfig and benchmark information"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Restrict model generation to N random samples",
        metavar="N",
    )
    parser.add_argument("kconfig_path", type=str, help="Path to Kconfig file")
    parser.add_argument(
        "model",
        type=str,
        help="Path to experiment results directory or model.json file",
    )

    args = parser.parse_args()

    if isinstance(args.log_level, int):
        logging.basicConfig(level=args.log_level)
    else:
        print(f"Invalid log level. Setting log level to INFO.", file=sys.stderr)

    if os.path.isdir(args.model):
        attributes = KConfigAttributes(args.kconfig_path, args.model)

        if args.show_failing_symbols:
            show_failing_symbols(attributes)
        if args.show_nop_symbols:
            show_nop_symbols(attributes)

        observations = list()

        for param, attr in attributes.data:
            for key, value in attr.items():
                observations.append(
                    {
                        "name": key,
                        "param": param,
                        "attribute": value,
                    }
                )

        if args.sample_size:
            shuffled_data_indices = np.random.permutation(
                np.arange(len(attributes.data))
            )
            sample_indices = shuffled_data_indices[: args.sample_size]
            raise RuntimeError("Not Implemented")

        by_name, parameter_names = dfatool.utils.observations_to_by_name(observations)

        model = AnalyticModel(
            by_name, parameter_names, compute_stats=not args.force_tree
        )

        if args.force_tree:
            for name in model.names:
                for attr in model.by_name[name]["attributes"]:
                    # TODO specify correct threshold
                    model.build_dtree(name, attr, 20)
            model.fit_done = True

        param_model, param_info = model.get_fitted()
        analytic_quality = model.assess(param_model)

        print("Model Error on Training Data:")
        for name in model.names:
            for attribute, error in analytic_quality[name].items():
                mae = error["mae"]
                smape = error["smape"]
                print(f"{name:15s} {attribute:20s}  ± {mae:5.0}  /  {smape:5.1f}%")

    else:
        raise NotImplementedError()

    if args.info:
        for name in model.names:
            print(f"{name}:")
            print(f"""    Number of Measurements: {len(by_name[name]["param"])}""")
            for i, param in enumerate(model.parameters):
                param_values = model.distinct_param_values_by_name[name][i]
                print(f"    Parameter {param} ∈ {param_values}")

    if args.export_tree:
        with open("nfpkeys.json", "r") as f:
            nfpkeys = json.load(f)
        complete_json_model = model.to_json(
            with_param_name=True, param_names=parameter_names
        )
        json_model = dict()
        for name, attribute_data in complete_json_model["name"].items():
            for attribue, data in attribute_data.items():
                data.update(nfpkeys[name][attribute])
                json_model[attribute] = data
        with open(args.export_tree, "w") as f:
            json.dump(json_model, f, sort_keys=True, cls=dfatool.utils.NpEncoder)

    if args.config:
        kconf = kconfiglib.Kconfig(args.kconfig_path)
        kconf.load_config(args.config)
        print(f"Model result for .config: {model.value_for_config(kconf)}")

        for symbol in model.symbols:
            kconf2 = kconfiglib.Kconfig(args.kconfig_path)
            kconf2.load_config(args.config)
            kconf_sym = kconf2.syms[symbol]
            if kconf_sym.tri_value == 0 and 2 in kconf_sym.assignable:
                kconf_sym.set_value(2)
            elif kconf_sym.tri_value == 2 and 0 in kconf_sym.assignable:
                kconf_sym.set_value(0)
            else:
                continue

            # specific to multipass:
            # Do not suggest changes which affect the application
            skip = False
            num_changes = 0
            changed_symbols = list()
            for i, csymbol in enumerate(model.symbols):
                if kconf.syms[csymbol].tri_value != kconf2.syms[csymbol].tri_value:
                    num_changes += 1
                    changed_symbols.append(csymbol)
                    if (
                        csymbol.startswith("app_")
                        and kconf.syms[csymbol].tri_value
                        != kconf2.syms[csymbol].tri_value
                    ):
                        skip = True
                        break
            if skip:
                continue

            try:
                model_diff = model.value_for_config(kconf2) - model.value_for_config(
                    kconf
                )
                if kconf_sym.choice:
                    print(
                        f"Setting {kconf_sym.choice.name} to {kconf_sym.name} changes {num_changes:2d} symbols, model change: {model_diff:+5.0f}"
                    )
                else:
                    print(
                        f"Setting {symbol} to {kconf_sym.str_value} changes {num_changes:2d} symbols, model change: {model_diff:+5.0f}"
                    )
            except TypeError:
                if kconf_sym.choice:
                    print(
                        f"Setting {kconf_sym.choice.name} to {kconf_sym.name} changes {num_changes:2d} symbols, model is undefined"
                    )
                else:
                    print(
                        f"Setting {symbol} to {kconf_sym.str_value} changes {num_changes:2d} symbols, model is undefined"
                    )
            for changed_symbol in changed_symbols:
                print(
                    f"    {changed_symbol:30s} -> {kconf2.syms[changed_symbol].str_value}"
                )


def show_failing_symbols(data):
    for symbol in data.param_names:
        unique_values = list(set(map(lambda p: p[symbol], data.failures)))
        for value in unique_values:
            fail_count = len(list(filter(lambda p: p[symbol] == value, data.failures)))
            success_count = len(
                list(filter(lambda p: p[0][symbol] == value, data.data))
            )
            if success_count == 0 and fail_count > 0:
                print(
                    f"Setting {symbol} to '{value}' reliably causes the build to fail (count = {fail_count})"
                )


def show_nop_symbols(data):
    for symbol in data.symbol_names:
        true_count = len(
            list(filter(lambda config: config[symbol] == True, data.failures))
        ) + len(list(filter(lambda config: config[0][symbol] == True, data.data)))
        false_count = len(
            list(filter(lambda config: config[symbol] == False, data.failures))
        ) + len(list(filter(lambda config: config[0][symbol] == False, data.data)))
        if false_count == 0:
            print(f"Symbol {symbol} is never n")
        if true_count == 0:
            print(f"Symbol {symbol} is never y")
    pass


if __name__ == "__main__":
    main()
