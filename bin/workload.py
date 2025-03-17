#!/usr/bin/env python3

import argparse
import json
import sys
import dfatool.cli
import dfatool.utils
from dfatool.model import AnalyticModel


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    parser.add_argument("--aggregate", choices=["sum"], default="sum")
    parser.add_argument(
        "--aggregate-init",
        default=0,
        type=float,
    )
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
    parser.add_argument("event", nargs="+", type=str)
    args = parser.parse_args()

    models = list()
    for model_file in args.models:
        with open(model_file, "r") as f:
            models.append(AnalyticModel.from_json(json.load(f)))

    if args.info:
        for i in range(len(models)):
            print(f"""{args.models[i]}: {" ".join(models[i].parameters)}""")
            for name in models[i].names:
                for attr in models[i].attributes(name):
                    print(f"    {name}.{attr}")

    aggregate = args.aggregate_init
    for event in args.event:
        nn, param = event.split("(")
        name, action = nn.split(".")
        param_model = None
        ref_model = None
        for model in models:
            if name in model.names and action in model.attributes(name):
                ref_model = model
                param_model, param_info = model.get_fitted()
                break
        assert param_model is not None
        param = param.removesuffix(")")
        if param == "":
            param = dict()
        else:
            param = dfatool.utils.parse_conf_str(param)
        if args.aggregate == "sum":
            aggregate += param_model(
                name,
                action,
                param=dfatool.utils.param_dict_to_list(param, ref_model.parameters),
            )

    print(aggregate)


if __name__ == "__main__":
    main()
