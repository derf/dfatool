#!/usr/bin/env python3

import json
import sys
from scipy.stats import kstest


def main(pvalue, file1, file2, macro=None):
    with open(file1, "r") as f:
        data1 = json.load(f)
    with open(file2, "r") as f:
        data2 = json.load(f)
    result = kstest(data1, data2)

    pvalue = float(pvalue)

    print(f"% {result}")
    if macro is None:
        print(r"\drefset{ttest/pvalue}{" + str(result.pvalue) + "}")
    if result.pvalue < pvalue:
        if macro:
            print("\\def\\" + macro + "{$p < " + f"{pvalue:0.2f}" + "$}")
        sys.exit(0)
    else:
        if macro:
            print("\\def\\" + macro + "{$p \\ge " + f"{pvalue:0.2f}" + "$}")
        sys.exit(1)


if __name__ == "__main__":
    main(*sys.argv[1:])
