#!/usr/bin/env python3

"""analyze-kconfig - Generate a model for KConfig selections

analyze-kconfig builds a model determining system attributes
(e.g. ROM or RAM usage) based on KConfig configuration variables.
Only boolean variables are supported at the moment.
"""

import argparse
import hashlib
import json
import kconfiglib
import logging
import os
import sys
import time

import numpy as np

import dfatool.cli
import dfatool.utils
import dfatool.functions as df
from dfatool.loader.kconfig import KConfigAttributes
from dfatool.model import AnalyticModel
from dfatool.validation import CrossValidator


def write_csv(f, model, attr, precision=None):
    model_attr = model.attr_by_name[attr]
    attributes = sorted(model_attr.keys())
    print(", ".join(model.parameters) + ",     " + ", ".join(attributes), file=f)

    if precision is not None:
        data_wrapper = lambda x: f"{x:.{precision}f}"
    else:
        data_wrapper = str

    # by convention, model_attr[attr].param_values is the same regardless of 'attr'
    for param_tuple in model_attr[attributes[0]].param_values:
        param_data = map(
            lambda a: model_attr[a].by_param.get(tuple(param_tuple), list()), attributes
        )
        print(
            ", ".join(map(str, param_tuple))
            + ",     "
            + ", ".join(map(data_wrapper, map(np.mean, param_data))),
            file=f,
        )


def main():
    timing = dict()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    dfatool.cli.add_standard_arguments(parser)
    parser.add_argument(
        "--boolean-parameters",
        action="store_true",
        help="Use boolean (not categorical) parameters when building the NFP model",
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
        "--max-std",
        type=str,
        metavar="VALUE_OR_MAP",
        help="Specify desired maximum standard deviation for RMT generation, either as float (global) or <key>/<attribute>=<value>[,<key>/<attribute>=<value>,...]. Has no effect when using CART, LMT or XGBoost.",
    )
    parser.add_argument(
        "--csv-precision",
        type=int,
        metavar="NDIGITS",
        help="Precision (number of decimal digits) for CSV export",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        metavar="FILE",
        help="Export observations aggregated by parameter to FILE",
    )
    parser.add_argument(
        "--export-csv-only",
        action="store_true",
        help="Exit after exporting observations to CSV file",
    )
    parser.add_argument(
        "--export-aggregate",
        type=str,
        metavar="FILE.json.xz",
        help="Export aggregated observations (intermediate and generic benchmark data representation) to FILE. Exported observations are affected by --param-shift and --ignore-param.",
    )
    parser.add_argument(
        "--export-aggregate-only",
        action="store_true",
        help="Exit after exporting aggregated observations",
    )
    parser.add_argument(
        "--export-observations",
        type=str,
        metavar="FILE.json.xz",
        help="Export observations (intermediate and generic benchmark data representation) to FILE",
    )
    parser.add_argument(
        "--export-observations-only",
        action="store_true",
        help="Exit after exporting observations",
    )
    parser.add_argument(
        "--export-webconf",
        type=str,
        help="Export kconfig-webconf NFP model to file",
        metavar="FILE",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Show model results for symbols in .config file",
        metavar="FILE",
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
        help="Path to experiment results directory or observations.json.xz file",
    )

    args = parser.parse_args()
    dfatool.cli.sanity_check(args)

    if args.log_level:
        numeric_level = getattr(logging, args.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            print(f"Invalid log level: {args.log_level}", file=sys.stderr)
            sys.exit(1)
        logging.basicConfig(
            level=numeric_level,
            format="{asctime} {levelname}:{name}:{message}",
            style="{",
        )

    if args.export_dref:
        dref = dict()

    if os.path.isdir(args.model):
        attributes = KConfigAttributes(args.kconfig_path, args.model)
        if args.export_dref:
            dref.update(attributes.to_dref())

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
            shuffled_data_indices = np.random.permutation(np.arange(len(observations)))
            sample_indices = shuffled_data_indices[: args.sample_size]
            new_observations = list()
            for sample_index in sample_indices:
                new_observations.append(observations[sample_index])
            observations = new_observations

        if args.export_observations:
            import lzma

            print(
                f"Exporting {len(observations)} observations to {args.export_observations}"
            )
            with lzma.open(args.export_observations, "wt") as f:
                json.dump(observations, f)
            if args.export_observations_only:
                return
    else:
        # show-failing-symbols, show-nop-symbols, DFATOOL_KCONF_WITH_CHOICE_NODES have no effect
        # in this branch.

        if os.path.exists(args.kconfig_path):
            attributes = KConfigAttributes(args.kconfig_path, None)
            if args.export_dref:
                dref.update(attributes.to_dref())

        if args.model.endswith("xz"):
            import lzma

            with lzma.open(args.model, "rt") as f:
                observations = json.load(f)
        elif args.model.endswith("ubjson"):
            import ubjson

            with open(args.model, "rb") as f:
                observations = ubjson.load(f)
        else:
            with open(args.model, "r") as f:
                observations = json.load(f)

        if bool(int(os.getenv("DFATOOL_KCONF_IGNORE_STRING", 1))) or bool(
            int(os.getenv("DFATOOL_KCONF_IGNORE_NUMERIC", 0))
        ):
            attributes = KConfigAttributes(args.kconfig_path, None)
            if type(observations) is dict:
                ignore_index = dict()
                new_param_names = list()
                for i, param in enumerate(observations["param_names"]):
                    if param in attributes.param_names:
                        new_param_names.append(param)
                    else:
                        ignore_index[i] = True
                observations["param_names"] = new_param_names
                for data in observations["by_name"].values():
                    for i in range(len(data["param"])):
                        for j in sorted(ignore_index.keys(), reverse=True):
                            data["param"][i].pop(j)
            else:
                for observation in observations:
                    to_remove = list()
                    for param in observation["param"].keys():
                        if param not in attributes.param_names:
                            to_remove.append(param)
                    for param in to_remove:
                        observation["param"].pop(param)

    if args.boolean_parameters:
        if type(observations) is list:
            logging.warning("--boolean-parameters is deprecated")
            dfatool.utils.observations_enum_to_bool(observations, kconfig=True)
        else:
            logging.error(
                "--boolean-parameters is only supported with legacy observations data"
            )
            sys.exit(1)

    function_override = dict()
    if args.function_override:
        for function_desc in args.function_override.split(";"):
            state_or_tran, attribute, function_str = function_desc.split(":")
            function_override[(state_or_tran, attribute)] = function_str

    by_name, parameter_names = dfatool.utils.observations_to_by_name(observations)

    if args.ignore_param:
        args.ignore_param = args.ignore_param.split(",")
        dfatool.utils.ignore_param(by_name, parameter_names, args.ignore_param)

    if args.param_shift:
        param_shift = dfatool.cli.parse_param_shift(args.param_shift)
        dfatool.utils.shift_param_in_aggregate(by_name, parameter_names, param_shift)

    if args.normalize_nfp:
        norm = dfatool.cli.parse_nfp_normalization(args.normalize_nfp)
        dfatool.utils.normalize_nfp_in_aggregate(by_name, norm)

    if args.export_aggregate:
        import lzma

        print(f"Exporting aggregate to {args.export_aggregate}")
        with lzma.open(args.export_aggregate, "wt") as f:
            json.dump(
                {"by_name": by_name, "param_names": parameter_names},
                f,
                cls=dfatool.utils.NpEncoder,
            )
        if args.export_aggregate_only:
            return

    # Release memory
    del observations

    if args.filter_param:
        args.filter_param = list(
            map(
                lambda entry: dfatool.cli.parse_filter_string(
                    entry, parameter_names=parameter_names
                ),
                args.filter_param.split(";"),
            )
        )
        dfatool.utils.filter_aggregate_by_param(
            by_name, parameter_names, args.filter_param
        )

    if args.filter_observation:
        args.filter_observation = list(
            map(lambda x: tuple(x.split(":")), args.filter_observation.split(","))
        )
        dfatool.utils.filter_aggregate_by_observation(by_name, args.filter_observation)

    if args.max_std:
        max_std = dict()
        if "=" in args.max_std:
            for kkv in args.max_std.split(","):
                kk, v = kkv.split("=")
                key, attr = kk.split("/")
                if key not in max_std:
                    max_std[key] = dict()
                max_std[key][attr] = float(v)
        else:
            for key in by_name.keys():
                max_std[key] = dict()
                for attr in by_name[key]["attributes"]:
                    max_std[key][attr] = float(args.max_std)
    else:
        max_std = None

    ts = time.time()
    if args.load_json:
        with open(args.load_json, "r") as f:
            model = AnalyticModel.from_json(json.load(f), by_name, parameter_names)
    else:
        model = AnalyticModel(
            by_name,
            parameter_names,
            force_tree=args.force_tree,
            max_std=max_std,
            compute_stats=not args.skip_param_stats,
            function_override=function_override,
        )
    timing["AnalyticModel"] = time.time() - ts

    if not model.names:
        logging.error(
            f"Model contains no names. Is --filter-param={args.filter_param} set too restrictive?"
        )
        sys.exit(1)

    if args.info:
        dfatool.cli.print_info_by_name(model, by_name)

    if args.export_csv_unparam:
        dfatool.cli.export_csv_unparam(
            model, args.export_csv_unparam, dialect=args.export_csv_dialect
        )

    if args.export_pgf_unparam:
        dfatool.cli.export_pgf_unparam(model, args.export_pgf_unparam)

    if args.export_json_unparam:
        dfatool.cli.export_json_unparam(model, args.export_json_unparam)

    if args.plot_unparam:
        import dfatool.plotter as dp

        for kv in args.plot_unparam.split(";"):
            state_or_trans, attribute, ylabel = kv.split(":")
            fname = "param_y_{}_{}.pdf".format(state_or_trans, attribute)
            dp.plot_y(
                model.by_name[state_or_trans][attribute],
                xlabel="measurement #",
                ylabel=ylabel,
                # output=fname,
                show=not args.non_interactive,
            )

    if args.boxplot_unparam:
        import dfatool.plotter as dp

        title = None
        if args.filter_param:
            title = "filter: " + ", ".join(
                map(lambda kv: f"{kv[0]}={kv[1]}", args.filter_param)
            )
        for name in model.names:
            attr_names = sorted(model.attributes(name))
            dp.boxplot(
                attr_names,
                [model.by_name[name][attr] for attr in attr_names],
                xlabel="Attribute",
                output=f"{args.boxplot_unparam}{name}.pdf",
                title=title,
                show=not args.non_interactive,
            )
            for attribute in attr_names:
                dp.boxplot(
                    [attribute],
                    [model.by_name[name][attribute]],
                    output=f"{args.boxplot_unparam}{name}-{attribute}.pdf",
                    title=title,
                    show=not args.non_interactive,
                )

    if args.boxplot_param:
        dfatool.cli.boxplot_param(args, model)

    if args.plot_param:
        import dfatool.plotter as dp

        for kv in args.plot_param.split(";"):
            try:
                state_or_trans, attribute, param_name, *function = kv.split(":")
            except ValueError:
                print(
                    "Usage: --plot-param='state_or_trans attribute param_name [additional function spec]'",
                    file=sys.stderr,
                )
                sys.exit(1)
            if len(function):
                function = gplearn_to_function(" ".join(function))
            else:
                function = None
            dp.plot_param(
                model,
                state_or_trans,
                attribute,
                model.param_index(param_name),
                extra_function=function,
                output=f"{state_or_trans}-{attribute}-{param_name}.pdf",
                show=not args.non_interactive,
            )

    if args.cross_validate:
        if ":" in args.cross_validate:
            xv_method, xv_count = args.cross_validate.split(":")
        else:
            xv_method, xv_count = args.cross_validate, 10
        xv_count = int(xv_count)
        xv = CrossValidator(
            AnalyticModel,
            by_name,
            parameter_names,
            force_tree=args.force_tree,
            max_std=max_std,
            compute_stats=not args.skip_param_stats,
            show_progress=args.progress,
        )
        xv.parameter_aware = args.parameter_aware_cross_validation
    else:
        xv_method = None
        xv_count = None

    static_model = model.get_static()

    ts = time.time()
    if xv_method == "montecarlo":
        static_quality, _ = xv.montecarlo(
            lambda m: m.get_static(), xv_count, static=True
        )
    elif xv_method == "kfold":
        static_quality, _ = xv.kfold(lambda m: m.get_static(), xv_count, static=True)
    else:
        static_quality = model.assess(static_model)
    timing["assess static"] = time.time() - ts

    ts = time.time()
    lut_model = model.get_param_lut()
    timing["get lut"] = time.time() - ts

    if lut_model is None:
        lut_quality = None
    else:
        ts = time.time()
        lut_quality = model.assess(lut_model)
        timing["assess lut"] = time.time() - ts

    if args.export_csv:
        for name in model.names:
            target = f"{args.export_csv}-{name}.csv"
            print(f"Exporting aggregated data to {target}")
            with open(target, "w") as f:
                write_csv(f, model, name, args.csv_precision)
        if args.export_csv_only:
            return

    ts = time.time()
    param_model, param_info = model.get_fitted()
    timing["get model"] = time.time() - ts

    ts = time.time()
    if xv_method == "montecarlo":
        xv.export_filename = args.export_xv
        analytic_quality, xv_analytic_models = xv.montecarlo(
            lambda m: m.get_fitted()[0], xv_count
        )
    elif xv_method == "kfold":
        xv.export_filename = args.export_xv
        analytic_quality, xv_analytic_models = xv.kfold(
            lambda m: m.get_fitted()[0], xv_count
        )
    else:
        if args.export_raw_predictions:
            analytic_quality, raw_results = model.assess(param_model, return_raw=True)
            with open(args.export_raw_predictions, "w") as f:
                json.dump(raw_results, f, cls=dfatool.utils.NpEncoder)
        else:
            analytic_quality = model.assess(param_model)
        xv_analytic_models = None
    timing["assess model"] = time.time() - ts

    if lut_model:
        ts = time.time()
        lut_quality = model.assess(lut_model)
        timing["assess lut"] = time.time() - ts
    else:
        lut_quality = None

    if "static" in args.show_model or "all" in args.show_model:
        print("--- static model ---")
        for name in model.names:
            for attribute in model.attributes(name):
                dfatool.cli.print_static(
                    model,
                    static_model,
                    name,
                    attribute,
                    with_dependence="all" in args.show_model,
                )

    if "param" in args.show_model or "all" in args.show_model:
        print("--- param model ---")
        for name in model.names:
            for attribute in model.attributes(name):
                info = param_info(name, attribute)
                dfatool.cli.print_model(f"{name:20s} {attribute:15s}", info)

    if args.show_model_error:
        dfatool.cli.model_quality_table(
            lut=lut_quality,
            model=analytic_quality,
            static=static_quality,
            model_info=param_info,
            xv_method=xv_method,
            xv_count=xv_count,
            error_metric=args.error_metric,
            load_model=args.load_json,
        )

    if args.show_model_complexity:
        dfatool.cli.print_model_complexity(model)

    if args.export_webconf:
        attributes = KConfigAttributes(args.kconfig_path, None)
        try:
            with open(f"{attributes.kconfig_root}/nfpkeys.json", "r") as f:
                nfpkeys = json.load(f)
        except FileNotFoundError:
            logging.error(
                f"{attributes.kconfig_root}/nfpkeys.json is missing, webconf model will be incomplete"
            )
            nfpkeys = None
        kconfig_hasher = hashlib.sha256()
        with open(args.kconfig_path, "rb") as f:
            kconfig_data = f.read()
            while len(kconfig_data) > 0:
                kconfig_hasher.update(kconfig_data)
                kconfig_data = f.read()
        kconfig_hash = str(kconfig_hasher.hexdigest())
        complete_json_model = model.to_json(
            with_param_name=True, param_names=parameter_names
        )
        json_model = dict()
        for name, attribute_data in complete_json_model["name"].items():
            for attribute, data in attribute_data.items():
                json_model[attribute] = data.copy()
                if nfpkeys:
                    json_model[attribute].update(nfpkeys[name][attribute])
                if "paramValueToIndex" in json_model[attribute]["modelFunction"]:
                    json_model[attribute]["paramValueToIndex"] = json_model[attribute][
                        "modelFunction"
                    ].pop("paramValueToIndex")
        out_model = {
            "model": json_model,
            "modelType": "dfatool-kconfig",
            "project": "tbd",
            "kconfigHash": kconfig_hash,
            "symbols": attributes.symbol_names,
            "choices": attributes.choice_names,
        }
        with open(args.export_webconf, "w") as f:
            json.dump(out_model, f, sort_keys=True, cls=dfatool.utils.NpEncoder)

    if args.export_dot:
        dfatool.cli.export_dot(model, args.export_dot)

    if args.export_dref:
        dref.update(
            model.to_dref(
                static_quality,
                lut_quality,
                analytic_quality,
                xv_models=xv_analytic_models,
            )
        )
        for key, value in timing.items():
            dref[f"timing/{key}"] = (value, r"\second")
        dfatool.cli.export_dataref(
            args.export_dref, dref, precision=args.dref_precision
        )

    if args.export_json:
        with open(args.export_json, "w") as f:
            json.dump(
                model.to_json(),
                f,
                sort_keys=True,
                cls=dfatool.utils.NpEncoder,
                indent=2,
            )

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
