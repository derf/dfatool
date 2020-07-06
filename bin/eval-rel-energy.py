#!/usr/bin/env python3

import getopt
import re
import sys
from dfatool.loader import RawData, pta_trace_to_aggregate
from dfatool.model import PTAModel

opt = dict()


def get_file_groups(args):
    groups = []
    index_low = 0
    while ":" in args[index_low:]:
        index_high = args[index_low:].index(":") + index_low
        groups.append(args[index_low:index_high])
        index_low = index_high + 1
    groups.append(args[index_low:])
    return groups


if __name__ == "__main__":

    ignored_trace_indexes = []
    discard_outliers = None
    safe_functions_enabled = False
    function_override = {}
    show_models = []
    show_quality = []

    try:
        optspec = (
            "plot-unparam= plot-param= show-models= show-quality= "
            "ignored-trace-indexes= discard-outliers= function-override= "
            "with-safe-functions"
        )
        raw_opts, args = getopt.getopt(sys.argv[1:], "", optspec.split(" "))

        for option, parameter in raw_opts:
            optname = re.sub(r"^--", "", option)
            opt[optname] = parameter

            if "ignored-trace-indexes" in opt:
                ignored_trace_indexes = list(
                    map(int, opt["ignored-trace-indexes"].split(","))
                )
                if 0 in ignored_trace_indexes:
                    print("[E] arguments to --ignored-trace-indexes start from 1")

            if "discard-outliers" in opt:
                discard_outliers = float(opt["discard-outliers"])

            if "function-override" in opt:
                for function_desc in opt["function-override"].split(";"):
                    state_or_tran, attribute, *function_str = function_desc.split(" ")
                    function_override[(state_or_tran, attribute)] = " ".join(
                        function_str
                    )

            if "show-models" in opt:
                show_models = opt["show-models"].split(",")

            if "show-quality" in opt:
                show_quality = opt["show-quality"].split(",")

            if "with-safe-functions" in opt:
                safe_functions_enabled = True

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    score_absolute = 0
    score_relative = 0

    for file_group in get_file_groups(args):
        print("")
        print("{}:".format(" ".join(file_group)))
        raw_data = RawData(file_group)

        preprocessed_data = raw_data.get_preprocessed_data(verbose=False)
        by_name, parameters, arg_count = pta_trace_to_aggregate(
            preprocessed_data, ignored_trace_indexes
        )
        model = PTAModel(
            by_name,
            parameters,
            arg_count,
            traces=preprocessed_data,
            ignore_trace_indexes=ignored_trace_indexes,
            discard_outliers=discard_outliers,
            function_override=function_override,
            verbose=False,
        )

        lut_quality = model.assess(model.get_param_lut())

        for trans in model.transitions():
            absolute_quality = lut_quality["by_name"][trans]["energy"]
            relative_quality = lut_quality["by_name"][trans]["rel_energy_prev"]
            if absolute_quality["mae"] < relative_quality["mae"]:
                best = "absolute"
                score_absolute += 1
            else:
                best = "relative"
                score_relative += 1

            print(
                "{:20s}: {:s}   (diff {:.0f} / {:.2f}%, abs {:.0f} / {:.2f}%, rel {:.0f} / {:.2f}%)".format(
                    trans,
                    best,
                    abs(absolute_quality["mae"] - relative_quality["mae"]),
                    abs(absolute_quality["mae"] - relative_quality["mae"])
                    * 100
                    / max(absolute_quality["mae"], relative_quality["mae"]),
                    absolute_quality["mae"],
                    absolute_quality["smape"],
                    relative_quality["mae"],
                    relative_quality["smape"],
                )
            )

    sys.exit(0)
