#!/usr/bin/env python3

import argparse
import json
import sys


def main():
    metric = dict()
    for line in sys.stdin:
        line = line.strip()
        data = json.loads(line)

        count = int(float(data["counter-value"]))
        label = data["event"]

        if data["metric-unit"] != "(null)":
            extra = float(data["metric-value"])
            extra_label = data["metric-unit"]
        else:
            extra = None
            extra_label = None

        metric[label] = (count, extra, extra_label)

    buf = ""
    for key in sorted(metric.keys()):
        count, extra, extra_label = metric[key]
        buf += f" {key}={count}"
        if extra_label is not None:
            if extra_label.startswith("of all"):
                label = extra_label.replace(" ", "-")
                buf += f" {key}-percentage-{label}={extra}"
            else:
                buf += f" {key}-metric={extra}"
    print(buf)


if __name__ == "__main__":
    main()
