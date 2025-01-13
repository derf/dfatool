#!/usr/bin/env python3

"""extract-speedup-from-log - Determine speedup from dfatool log files

foo
"""

import argparse
import dfatool.cli
import dfatool.utils
import logging
import numpy as np
import sys
from dfatool.loader import Logfile, CSVfile
from dfatool.model import AnalyticModel
from functools import reduce


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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "--add-param",
        metavar="<param>=<value>[ <param>=<value> ...]",
        type=str,
        help="Add additional parameter specifications to output lines",
    )
    parser.add_argument(
        "--filter-param",
        metavar="<parameter name><condition>[;<parameter name><condition>...]",
        type=str,
        help="Only consider measurements where <parameter name> satisfies <condition>. "
        "<condition> may be <operator><parameter value> with operator being < / <= / = / >= / >, "
        "or ∈<parameter value>[,<parameter value>...]. "
        "All other measurements (including those where it is None, that is, has not been set yet) are discarded. "
        "Note that this may remove entire function calls from the model.",
    )
    parser.add_argument(
        "--ignore-param",
        metavar="<parameter name>[,<parameter name>,...]",
        type=str,
        help="Ignore listed parameters during model generation",
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        choices=["debug", "info", "warning", "error"],
        default="warning",
        help="Set log level",
    )
    parser.add_argument(
        "numerator",
        type=str,
        help="numerator parameters",
    )
    parser.add_argument(
        "denominator",
        type=str,
        help="denominator parameters",
    )
    parser.add_argument(
        "observation",
        type=str,
        help="observation (key:attribute) used for speedup calculation",
    )
    parser.add_argument(
        "logfiles",
        nargs="+",
        type=str,
        help="Path to benchmark output (.txt or .txt.xz)",
    )
    args = parser.parse_args()

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

    observations = reduce(lambda a, b: a + b, map(parse_logfile, args.logfiles))
    by_name_num, parameter_names_num = dfatool.utils.observations_to_by_name(
        observations
    )
    by_name_denom, parameter_names_denom = dfatool.utils.observations_to_by_name(
        observations
    )
    del observations

    if args.filter_param:
        args.filter_param = list(
            map(
                lambda entry: dfatool.cli.parse_filter_string(
                    entry, parameter_names=parameter_names_num
                ),
                args.filter_param.split(";"),
            )
        )
    else:
        args.filter_param = list()

    filter_num = list(
        map(
            lambda entry: dfatool.cli.parse_filter_string(
                entry, parameter_names=parameter_names_num
            ),
            args.numerator.split(";"),
        )
    )

    filter_denom = list(
        map(
            lambda entry: dfatool.cli.parse_filter_string(
                entry, parameter_names=parameter_names_denom
            ),
            args.denominator.split(";"),
        )
    )

    filter_num += args.filter_param
    filter_denom += args.filter_param

    ignore_num = list(map(lambda x: x[0], filter_num))
    ignore_denom = list(map(lambda x: x[0], filter_denom))
    assert ignore_num == ignore_denom

    if args.ignore_param:
        args.ignore_param = args.ignore_param.split(";")
        ignore_num += args.ignore_param
        ignore_denom += args.ignore_param

    dfatool.utils.filter_aggregate_by_param(
        by_name_num, parameter_names_num, filter_num
    )
    dfatool.utils.filter_aggregate_by_param(
        by_name_denom, parameter_names_denom, filter_denom
    )
    dfatool.utils.ignore_param(by_name_num, parameter_names_num, ignore_num)
    dfatool.utils.ignore_param(by_name_denom, parameter_names_denom, ignore_denom)

    model_num = AnalyticModel(
        by_name_num,
        parameter_names_num,
        compute_stats=False,
    )

    model_denom = AnalyticModel(
        by_name_denom,
        parameter_names_denom,
        compute_stats=False,
    )

    for param_key in model_num.get_by_param().keys():
        name, params = param_key
        num_data = model_num.get_by_param().get(param_key).get(args.observation)
        try:
            denom_data = model_denom.get_by_param().get(param_key).get(args.observation)
        except AttributeError:
            logging.error(f"Cannot find numerator param {param_key}  in denominator")
            logging.error(f"Parameter names == {tuple(parameter_names_num)}")
            logging.error("You may need to adjust --ignore-param")
            sys.exit(1)
        if num_data and denom_data:
            param_str = " ".join(
                map(
                    lambda i: f"{parameter_names_num[i]}={params[i]}",
                    range(len(params)),
                )
            )
            if args.add_param is not None:
                param_str += args.add_param
            for speedup in np.array(num_data) / np.array(denom_data):
                print(f"[::] {name} | {param_str} | speedup={speedup}")


if __name__ == "__main__":
    main()
