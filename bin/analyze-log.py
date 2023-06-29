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
import re


def kv_to_param(kv_str, cast):
    key, value = kv_str.split("=")
    value = cast(value)
    return key, value


def kv_to_param_f(kv_str):
    return kv_to_param(kv_str, dfatool.utils.soft_cast_float)


def kv_to_param_i(kv_str):
    return kv_to_param(kv_str, dfatool.utils.soft_cast_int)


def parse_logfile(filename):
    observations = list()

    with open(filename, "r") as f:
        for lineno, line in enumerate(f):
            m = re.search(r"\[::\] *([^|]*?) *[|] *([^|]*?) *[|] *(.*)", line)
            if m:
                name_str = m.group(1)
                param_str = m.group(2)
                attr_str = m.group(3)
                try:
                    param = dict(map(kv_to_param_i, param_str.split(" ")))
                    attr = dict(map(kv_to_param_f, attr_str.split(" ")))
                    observations.append(
                        {
                            "name": name_str,
                            "param": param,
                            "attribute": attr,
                        }
                    )
                except ValueError:
                    print(
                        f"Error parsing {filename}: invalid key-value pair in line {lineno+1}"
                    )
                    print(f"Offending entry:\n{line}")
                    raise

    return observations


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
        metavar="<name> <attribute> <parameter> [gplearn function][;<name> <attribute> <parameter> [function];...])",
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
        "logfiles", nargs="+", type=str, help="Path to benchmark output"
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

    model = AnalyticModel(
        by_name,
        parameter_names,
        force_tree=args.force_tree,
    )

    if args.info:
        dfatool.cli.print_info_by_name(model, by_name)

    if args.export_pgf_unparam:
        dfatool.cli.export_pgf_unparam(model, args.export_pgf_unparam)

    if args.plot_unparam:
        for kv in args.plot_unparam.split(";"):
            state_or_trans, attribute, ylabel = kv.split(":")
            fname = "param_y_{}_{}.pdf".format(state_or_trans, attribute)
            dfatool.plotter.plot_y(
                model.by_name[state_or_trans][attribute],
                xlabel="measurement #",
                ylabel=ylabel,
                # output=fname,
            )

    if args.cross_validate:
        xv_method, xv_count = args.cross_validate.split(":")
        xv_count = int(xv_count)
        xv = CrossValidator(
            AnalyticModel,
            by_name,
            parameter_names,
            force_tree=args.force_tree,
            max_std=max_std,
            compute_stats=not args.skip_param_stats,
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
        for name in model.names:
            for attribute in model.attributes(name):
                dfatool.cli.print_static(model, static_model, name, attribute)

    if "param" in args.show_model or "all" in args.show_model:
        for name in model.names:
            for attribute in model.attributes(name):
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
            )


if __name__ == "__main__":
    main()
