#!/usr/bin/env python3

import argparse
import json
import logging
import sys
import dfatool.cli
import dfatool.utils
from dfatool.behaviour import EventSequenceModel
from dfatool.model import AnalyticModel


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    parser.add_argument("--aggregate", choices=["sum"], default="sum")
    parser.add_argument("--aggregate-unit", choices=["s", "B/s"], default="s")
    parser.add_argument(
        "--aggregate-init",
        default=0,
        type=float,
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        choices=["debug", "info", "warning", "error"],
        default="warning",
        help="Set log level",
    )
    parser.add_argument("--normalize-output", type=str)
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show benchmark information (number of measurements, parameter values, ...)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        help="Path to model file (.json or .json.xz)",
    )
    parser.add_argument(
        "--use-lut",
        action="store_true",
        help="Use LUT rather than performance model for prediction",
    )
    parser.add_argument("event", nargs="+", type=str)
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

    models = list()
    for model_file in args.models:
        with open(model_file, "r") as f:
            models.append(AnalyticModel.from_json(json.load(f)))

    if args.info:
        for i in range(len(models)):
            print(f"""{args.models[i]}: {" ".join(models[i].parameters)}""")
            _, param_info = models[i].get_fitted()
            for name in models[i].names:
                for attr in models[i].attributes(name):
                    print(f"    {name}.{attr}  {param_info(name, attr)}")

    workload = EventSequenceModel(models)
    aggregate = workload.eval_strs(
        args.event,
        aggregate=args.aggregate,
        aggregate_init=args.aggregate_init,
        use_lut=args.use_lut,
    )

    if args.normalize_output:
        sf = dfatool.cli.parse_shift_function(
            "--normalize-output", args.normalize_output
        )
        print(dfatool.utils.human_readable(sf(aggregate), args.aggregate_unit))
    else:
        print(dfatool.utils.human_readable(aggregate, args.aggregate_unit))


if __name__ == "__main__":
    main()
