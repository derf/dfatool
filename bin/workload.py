#!/usr/bin/env python3

import argparse
import json
import logging
import sys
import dfatool.cli
import dfatool.utils
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

    aggregate = args.aggregate_init
    for event in args.event:

        event_normalizer = lambda p: p
        if "/" in event:
            v1, v2 = event.split("/")
            if dfatool.utils.is_numeric(v1):
                event = v2.strip()
                event_normalizer = lambda p: dfatool.utils.soft_cast_float(v1) / p
            elif dfatool.utils.is_numeric(v2):
                event = v1.strip()
                event_normalizer = lambda p: p / dfatool.utils.soft_cast_float(v2)
            else:
                raise RuntimeError(f"Cannot parse '{event}'")

        nn, param = event.split("(")
        name, action = nn.split(".")
        param_model = None
        ref_model = None

        for model in models:
            if name in model.names and action in model.attributes(name):
                ref_model = model
                if args.use_lut:
                    param_model = model.get_param_lut(allow_none=True)
                else:
                    param_model, param_info = model.get_fitted()
                break

        if param_model is None:
            raise RuntimeError(f"Did not find a model for {name}.{action}")

        param = param.removesuffix(")")
        if param == "":
            param = dict()
        else:
            param = dfatool.utils.parse_conf_str(param)

        param_list = dfatool.utils.param_dict_to_list(param, ref_model.parameters)

        if not args.use_lut and not param_info(name, action).is_predictable(param_list):
            logging.warning(
                f"Cannot predict {name}.{action}({param}), falling back to static model"
            )

        try:
            event_output = event_normalizer(
                param_model(
                    name,
                    action,
                    param=param_list,
                )
            )
        except KeyError:
            logging.error(f"Cannot predict {name}.{action}({param}) from LUT model")
            raise

        if args.aggregate == "sum":
            aggregate += event_output

    if args.normalize_output:
        sf = dfatool.cli.parse_shift_function(
            "--normalize-output", args.normalize_output
        )
        print(dfatool.utils.human_readable(sf(aggregate), args.aggregate_unit))
    else:
        print(dfatool.utils.human_readable(aggregate, args.aggregate_unit))


if __name__ == "__main__":
    main()
