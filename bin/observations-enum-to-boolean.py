#!/usr/bin/env python3

import dfatool.utils
import json
import lzma
import sys


def main():
    infile = sys.argv[1]
    outfile = sys.argv[2]

    with lzma.open(infile, "rt") as f:
        observations = json.load(f)

    distinct_param_values = dict()
    replace_map = dict()

    for observation in observations:
        for k, v in observation["param"].items():
            if not k in distinct_param_values:
                distinct_param_values[k] = set()
            if v is not None:
                distinct_param_values[k].add(v)

    for param_name, distinct_values in distinct_param_values.items():
        if len(distinct_values) > 2 and not all(
            map(lambda x: x is None or dfatool.utils.is_numeric(x), distinct_values)
        ):
            replace_map[param_name] = distinct_values

    for observation in observations:
        for k, v in replace_map.items():
            enum_value = observation["param"].pop(k)
            for binary_key in v:
                observation["param"][binary_key] = int(enum_value == binary_key)

    with lzma.open(outfile, "wt") as f:
        json.dump(observations, f)


if __name__ == "__main__":
    main()
