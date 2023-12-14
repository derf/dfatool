#!/usr/bin/env python3

"""analyze-log - Generate a model from performance benchmarks log files

foo
"""

import argparse
import dfatool.cli
import dfatool.plotter
import dfatool.utils
import dfatool.functions as df
from dfatool.model import AnalyticModel
from dfatool.validation import CrossValidator
from functools import reduce
import json
import sys
import re


def parse_logfile(filename):
    lf = dfatool.utils.Logfile()

    if filename.endswith("xz"):
        import lzma

        with lzma.open(filename, "rt") as f:
            return lf.load(f)
    with open(filename, "r") as f:
        return lf.load(f)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    dfatool.cli.add_standard_arguments(parser)
    parser.add_argument(
        "--plot-unparam",
        metavar="<name>:<attribute>:<Y axis label>[;<name>:<attribute>:<label>;...]",
        type=str,
        help="Plot all mesurements for <name> <attribute> without regard for parameter values. "
        "X axis is measurement number/id.",
    )
    parser.add_argument(
        "--plot-param",
        metavar="<name>:<attribute>:<parameter>[;<name>:<attribute>:<parameter>;...])",
        type=str,
        help="Plot measurements for <name> <attribute> by <parameter>. "
        "X axis is parameter value. "
        "Plots the model function as one solid line for each combination of non-<parameter> parameters. "
        "Also plots the corresponding measurements. "
        "If gplearn function is set, it is plotted using dashed lines.",
    )
    parser.add_argument(
        "--show-model",
        choices=["static", "paramdetection", "param", "all"],
        action="append",
        default=list(),
        help="static: show static model values as well as parameter detection heuristic.\n"
        "paramdetection: show stddev of static/lut/fitted model\n"
        "param: show parameterized model functions and regression variable values\n"
        "all: all of the above\n",
    )
    parser.add_argument(
        "--show-quality",
        choices=["table", "summary", "all"],
        action="append",
        default=list(),
        help="table: show static/fitted/lut SMAPE and MAE for each name and attribute.\n"
        "summary: show static/fitted/lut SMAPE and MAE for each attribute, averaged over all states/transitions.\n"
        "all: all of the above.\n",
    )
    parser.add_argument(
        "--force-tree",
        action="store_true",
        help="Build decision tree without checking for analytic functions first",
    )
    parser.add_argument(
        "--export-model", metavar="FILE", type=str, help="Export JSON model to FILE"
    )
    parser.add_argument(
        "--non-interactive", action="store_true", help="Do not show interactive plots"
    )
    parser.add_argument(
        "logfiles",
        nargs="+",
        type=str,
        help="Path to benchmark output (.txt or .txt.xz)",
    )
    args = parser.parse_args()

    if args.filter_param:
        args.filter_param = list(
            map(lambda x: x.split("="), args.filter_param.split(","))
        )
    else:
        args.filter_param = list()

    observations = reduce(lambda a, b: a + b, map(parse_logfile, args.logfiles))
    by_name, parameter_names = dfatool.utils.observations_to_by_name(observations)
    del observations

    if args.ignore_param:
        args.ignore_param = args.ignore_param.split(",")
        dfatool.utils.ignore_param(by_name, parameter_names, args.ignore_param)

    dfatool.utils.filter_aggregate_by_param(by_name, parameter_names, args.filter_param)

    if args.param_shift:
        param_shift = dfatool.cli.parse_param_shift(args.param_shift)
        dfatool.utils.shift_param_in_aggregate(by_name, parameter_names, param_shift)

    if args.normalize_nfp:
        norm = dfatool.cli.parse_nfp_normalization(args.normalize_nfp)
        dfatool.utils.normalize_nfp_in_aggregate(by_name, norm)

    function_override = dict()
    if args.function_override:
        for function_desc in args.function_override.split(";"):
            state_or_tran, attribute, *function_str = function_desc.split(" ")
            function_override[(state_or_tran, attribute)] = " ".join(function_str)

    model = AnalyticModel(
        by_name,
        parameter_names,
        force_tree=args.force_tree,
        function_override=function_override,
    )

    if args.info:
        dfatool.cli.print_info_by_name(model, by_name)

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
        title_suffix = None
        if args.filter_param:
            title_suffix = "filter: " + ", ".join(
                map(lambda kv: f"{kv[0]}={kv[1]}", args.filter_param)
            )
        for name in model.names:
            attr_names = sorted(model.attributes(name))
            dfatool.plotter.boxplot(
                attr_names,
                [model.by_name[name][attr] for attr in attr_names],
                xlabel="Attribute",
                output=f"{args.boxplot_unparam}{name}.pdf",
                title_suffix=title_suffix,
                show=not args.non_interactive,
            )
            for attribute in attr_names:
                dfatool.plotter.boxplot(
                    [attribute],
                    [model.by_name[name][attribute]],
                    output=f"{args.boxplot_unparam}{name}-{attribute}.pdf",
                    title_suffix=title_suffix,
                    show=not args.non_interactive,
                )

    if args.boxplot_param:
        title_suffix = None
        if args.filter_param:
            title_suffix = "filter: " + ", ".join(
                map(lambda kv: f"{kv[0]}={kv[1]}", args.filter_param)
            )
        by_param = model.get_by_param()
        for name in model.names:
            attr_names = sorted(model.attributes(name))
            param_keys = list(
                map(lambda kv: kv[1], filter(lambda kv: kv[0] == name, by_param.keys()))
            )
            param_desc = list(
                map(
                    lambda param_key: ", ".join(
                        map(
                            lambda ip: f"{model.param_name(ip[0])}={ip[1]}",
                            enumerate(param_key),
                        )
                    ),
                    param_keys,
                )
            )
            for attribute in attr_names:
                dfatool.plotter.boxplot(
                    param_desc,
                    list(map(lambda k: by_param[(name, k)][attribute], param_keys)),
                    output=f"{args.boxplot_param}{name}-{attribute}.pdf",
                    title_suffix=title_suffix,
                    ylabel=attribute,
                    show=not args.non_interactive,
                )

    if args.cross_validate:
        xv_method, xv_count = args.cross_validate.split(":")
        xv_count = int(xv_count)
        xv = CrossValidator(
            AnalyticModel,
            by_name,
            parameter_names,
            force_tree=args.force_tree,
        )
        xv.parameter_aware = args.parameter_aware_cross_validation
    else:
        xv_method = None

    static_model = model.get_static()
    try:
        lut_model = model.get_param_lut()
    except RuntimeError as e:
        if args.force_tree:
            # this is to be expected
            logging.debug(f"Skipping LUT model: {e}")
        else:
            logging.warning(f"Skipping LUT model: {e}")
        lut_model = None

    param_model, param_info = model.get_fitted()
    static_quality = model.assess(static_model)
    analytic_quality = model.assess(param_model)
    if lut_model:
        lut_quality = model.assess(lut_model)
    else:
        lut_quality = None

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
                )

    if "param" in args.show_model or "all" in args.show_model:
        print("--- param model ---")
        for name in sorted(model.names):
            for attribute in sorted(model.attributes(name)):
                info = param_info(name, attribute)
                if type(info) is df.AnalyticFunction:
                    dfatool.cli.print_analyticinfo(f"{name:10s} {attribute:15s}", info)
                elif type(info) is df.SplitFunction:
                    dfatool.cli.print_splitinfo(
                        model.parameters, info, f"{name:10s} {attribute:15s}"
                    )

    if "table" in args.show_quality or "all" in args.show_quality:
        if xv_method is not None:
            print(f"Model error after cross validation ({xv_method}, {xv_count}):")
        else:
            print("Model error on training data:")
        dfatool.cli.model_quality_table(
            ["static", "parameterized", "LUT"],
            [static_quality, analytic_quality, lut_quality],
            [None, param_info, None],
        )

    if args.export_model:
        print(f"Exportding model to {args.export_model}")
        json_model = model.to_json()
        with open(args.export_model, "w") as f:
            json.dump(
                json_model, f, indent=2, sort_keys=True, cls=dfatool.utils.NpEncoder
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
                output=f"{state_or_trans} {attribute} {param_name}.pdf",
                show=not args.non_interactive,
            )


if __name__ == "__main__":
    main()
