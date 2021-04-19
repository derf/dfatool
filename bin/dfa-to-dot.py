#!/usr/bin/env python3

from dfatool.automata import PTA
import sys


def main(filename):
    pta = PTA.from_file(filename)
    print(pta.to_dot())


if __name__ == "__main__":
    main(*sys.argv[1:])
