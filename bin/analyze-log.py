#!/usr/bin/env python3

"""analyze-log - Generate a model from performance benchmarks log files

foo
"""

import argparse
import dfatool.cli
import dfatool.plotter
import dfatool.utils
import dfatool.functions as df
from dfatool.loader import Logfile, CSVfile
from dfatool.model import AnalyticModel
from dfatool.validation import CrossValidator
from functools import reduce
import logging
import json
import re
import sys
import time


def parse_logfile(filename):
    if ".csv" in filename:
        loader = CSVfile()
    else:
        loader = Logfile()

    if filename.endswith("xz"):
        import lzma

        with lzma.open(filename, "rt") as f:
            return loader.load(f)
    with open(filename, "r") as f:
        return loader.load(f)


def main():
    timing = dict()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    dfatool.cli.add_standard_arguments(parser)
    parser.add_argument(
        "--export-model", metavar="FILE", type=str, help="Export JSON model to FILE"
    )
    parser.add_argument(
        "logfiles",
        nargs="+",
        type=str,
        help="Path to benchmark output (.txt or .txt.xz)",
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

    if args.filter_observation:
        args.filter_observation = list(
            map(lambda x: tuple(x.split(":")), args.filter_observation.split(","))
        )

    observations = reduce(lambda a, b: a + b, map(parse_logfile, args.logfiles))
    by_name, parameter_names = dfatool.utils.observations_to_by_name(observations)
    del observations

    if args.ignore_param:
        args.ignore_param = args.ignore_param.split(",")
        dfatool.utils.ignore_param(by_name, parameter_names, args.ignore_param)

    if args.filter_param:
        args.filter_param = list(
            map(
                lambda entry: dfatool.cli.parse_filter_string(
                    entry, parameter_names=parameter_names
                ),
                args.filter_param.split(";"),
            )
        )
    else:
        args.filter_param = list()

    dfatool.utils.filter_aggregate_by_param(by_name, parameter_names, args.filter_param)
    dfatool.utils.filter_aggregate_by_observation(by_name, args.filter_observation)

    if args.param_shift:
        param_shift = dfatool.cli.parse_param_shift(args.param_shift)
        dfatool.utils.shift_param_in_aggregate(by_name, parameter_names, param_shift)

    if args.normalize_nfp:
        norm = dfatool.cli.parse_nfp_normalization(args.normalize_nfp)
        dfatool.utils.normalize_nfp_in_aggregate(by_name, norm)

    function_override = dict()
    if args.function_override:
        for function_desc in args.function_override.split(";"):
            state_or_tran, attribute, function_str = function_desc.split(":")
            function_override[(state_or_tran, attribute)] = function_str

    ts = time.time()
    if args.load_json:
        with open(args.load_json, "r") as f:
            model = AnalyticModel.from_json(json.load(f), by_name, parameter_names)
    else:
        model = AnalyticModel(
            by_name,
            parameter_names,
            force_tree=args.force_tree,
            compute_stats=not args.skip_param_stats,
            function_override=function_override,
        )
    timing["AnalyticModel"] = time.time() - ts

    if args.info:
        dfatool.cli.print_info_by_name(model, by_name)

    if args.information_gain:
        dfatool.cli.print_information_gain_by_name(model, by_name)

    if args.export_csv_unparam:
        dfatool.cli.export_csv_unparam(
            model, args.export_csv_unparam, dialect=args.export_csv_dialect
        )

    if args.export_pgf_unparam:
        dfatool.cli.export_pgf_unparam(model, args.export_pgf_unparam)

    if args.export_json_unparam:
        dfatool.cli.export_json_unparam(model, args.export_json_unparam)

    if args.plot_unparam:
        for kv in args.plot_unparam.split(";"):
            state_or_trans, attribute, ylabel = kv.split(":")
            fname = "param_y_{}_{}.pdf".format(state_or_trans, attribute)
            dfatool.plotter.plot_y(
                model.by_name[state_or_trans][attribute],
                xlabel="measurement #",
                ylabel=ylabel,
                # output=fname,
                show=not args.non_interactive,
            )

    if args.boxplot_unparam:
        title = None
        if args.filter_param:
            title = "filter: " + ", ".join(
                map(lambda kv: f"{kv[0]}={kv[1]}", args.filter_param)
            )
        for name in model.names:
            attr_names = sorted(model.attributes(name))
            dfatool.plotter.boxplot(
                attr_names,
                [model.by_name[name][attr] for attr in attr_names],
                xlabel="Attribute",
                output=f"{args.boxplot_unparam}{name}.pdf",
                title=title,
                show=not args.non_interactive,
            )
            for attribute in attr_names:
                dfatool.plotter.boxplot(
                    [attribute],
                    [model.by_name[name][attribute]],
                    output=f"{args.boxplot_unparam}{name}-{attribute}.pdf",
                    title=title,
                    show=not args.non_interactive,
                )

    if args.boxplot_param:
        dfatool.cli.boxplot_param(args, model)

    if args.cross_validate:
        xv_method, xv_count = args.cross_validate.split(":")
        xv_count = int(xv_count)
        xv = CrossValidator(
            AnalyticModel,
            by_name,
            parameter_names,
            force_tree=args.force_tree,
            compute_stats=not args.skip_param_stats,
            show_progress=args.progress,
        )
        xv.parameter_aware = args.parameter_aware_cross_validation
    else:
        xv_method = None
        xv_count = None

    static_model = model.get_static()

    ts = time.time()
    lut_model = model.get_param_lut()
    timing["get lut"] = time.time() - ts

    if lut_model is None:
        lut_quality = None
    else:
        ts = time.time()
        lut_quality = model.assess(lut_model)
        timing["assess lut"] = time.time() - ts

    ts = time.time()
    param_model, param_info = model.get_fitted()
    timing["get model"] = time.time() - ts

    ts = time.time()
    if xv_method == "montecarlo":
        static_quality, _ = xv.montecarlo(
            lambda m: m.get_static(), xv_count, static=True
        )
        xv.export_filename = args.export_xv
        analytic_quality, _ = xv.montecarlo(lambda m: m.get_fitted()[0], xv_count)
    elif xv_method == "kfold":
        static_quality, _ = xv.kfold(lambda m: m.get_static(), xv_count, static=True)
        xv.export_filename = args.export_xv
        analytic_quality, _ = xv.kfold(lambda m: m.get_fitted()[0], xv_count)
    else:
        static_quality = model.assess(static_model)
        if args.export_raw_predictions:
            analytic_quality, raw_results = model.assess(param_model, return_raw=True)
            with open(args.export_raw_predictions, "w") as f:
                json.dump(raw_results, f, cls=dfatool.utils.NpEncoder)
        else:
            analytic_quality = model.assess(param_model)
    timing["assess model"] = time.time() - ts

    if "static" in args.show_model or "all" in args.show_model:
        print("--- static model ---")
        for name in sorted(model.names):
            for attribute in sorted(model.attributes(name)):
                dfatool.cli.print_static(
                    model,
                    static_model,
                    name,
                    attribute,
                    with_dependence="all" in args.show_model,
                    precision=args.show_model_precision,
                )

    if "param" in args.show_model or "all" in args.show_model:
        print("--- param model ---")
        for name in sorted(model.names):
            for attribute in sorted(model.attributes(name)):
                info = param_info(name, attribute)
                dfatool.cli.print_model(
                    f"{name:10s} {attribute:15s}",
                    info,
                    precision=args.show_model_precision,
                )

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

    if args.export_model:
        print(f"Exportding model to {args.export_model}")
        json_model = model.to_json()
        with open(args.export_model, "w") as f:
            json.dump(
                json_model, f, indent=2, sort_keys=True, cls=dfatool.utils.NpEncoder
            )

    if args.export_dot:
        dfatool.cli.export_dot(model, args.export_dot)

    if args.export_dref:
        dref = model.to_dref(static_quality, lut_quality, analytic_quality)
        for key, value in timing.items():
            dref[f"timing/{key}"] = (value, r"\second")

        if args.information_gain:
            for name in model.names:
                for attr in model.attributes(name):
                    mutual_information = model.mutual_information(name, attr)
                    for param in model.parameters:
                        if param in mutual_information:
                            dref[f"mutual information/{name}/{attr}/{param}"] = (
                                mutual_information[param]
                            )

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

    if args.plot_param:
        for kv in args.plot_param.split(";"):
            try:
                state_or_trans, attribute, param_name = kv.split(":")
            except ValueError:
                print(
                    "Usage: --plot-param='state_or_trans:attribute:param_name'",
                    file=sys.stderr,
                )
                sys.exit(1)
            dfatool.plotter.plot_param(
                model,
                state_or_trans,
                attribute,
                model.param_index(param_name),
                title=state_or_trans,
                ylabel=attribute,
                xlabel=param_name,
                output=f"{state_or_trans}-{attribute}-{param_name}.pdf",
                show=not args.non_interactive,
            )


if __name__ == "__main__":
    main()
