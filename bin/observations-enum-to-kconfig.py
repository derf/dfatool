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

    dfatool.utils.observations_enum_to_bool(observations, kconfig=True)

    with lzma.open(outfile, "wt") as f:
        json.dump(observations, f)


if __name__ == "__main__":
    main()
