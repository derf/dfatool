#!/usr/bin/env python3
# vim:tabstop=4 softtabstop=4 shiftwidth=4 textwidth=160 smarttab expandtab colorcolumn=160

import argparse
import dfatool.runner
from dfatool.utils import NpEncoder
import itertools
import json
import logging
import numpy as np
import os
import sklearn.datasets
import subprocess
import sys
import time

if __name__ == "__main__":

    quantization_levels = ("f32", "i8d", "i8")

    parser = argparse.ArgumentParser()

    parser.add_argument("--quantization", type=str, choices=quantization_levels)
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        choices=["debug", "info", "warning", "error"],
        default="warning",
        help="Set log level",
    )
    parser.add_argument(
        "--model-location", type=str, choices=("ram", "rom"), default="rom"
    )
    parser.add_argument("--multipass-base", type=str, default="../multipass")
    parser.add_argument("--multipass-app", type=str, default="tfmicro-test")
    parser.add_argument(
        "--n6705b-logger",
        type=str,
        default="/home/derf/var/projects/n6705b-logger/bin/n6705b.py --channel 3",
    )
    parser.add_argument("model_file")

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

    dfatool_name = args.model_file
    for quantization_level in quantization_levels:
        if dfatool_name.endswith(f".{quantization_level}.tflite"):
            dfatool_name = dfatool_name.removesuffix(f".{quantization_level}.tflite")
            args.quantization = quantization_level

    dfatool_name = dfatool_name.split("/")[-1]

    benchmark = subprocess.run(
        (
            "libexec/multipass-benchmark-tflite",
            args.multipass_base,
            args.model_file,
            args.model_location,
        ),
        capture_output=True,
        universal_newlines=True,
        timeout=60,
        check=True,
    )

    latencies = list()
    for line in benchmark.stdout.split("\n"):
        if line.startswith("interpreter->Invoke() ="):
            try:
                latency, suffix, _ = line.split("= ")[1].split(" ", maxsplit=2)
                if suffix == "ms":
                    latency = float(latency)
                    latencies.append(latency)
            except ValueError:
                # typically caused by garbled UART output after flashing
                pass

    if not latencies:
        print(
            f"Unable to parse multipasso output -- latest line: {line}", file=sys.stderr
        )
        sys.exit(1)

    nfp_file = f"src/app/{args.multipass_app}/main.o"
    nfp_benchmark = dfatool.runner.ShellMonitor(
        (
            "script/nfpvalues.py",
            "arm-none-eabi-size",
            "text*,rodata*,data*",
            "data*,bss*",
            nfp_file,
        ),
        cwd=args.multipass_base,
    )

    stdout, stderr = nfp_benchmark.run()
    try:
        data = json.loads(stdout[0])
    except json.decoder.JSONDecodeError:
        print(f"Unable to parse JSON output: {stdout}", file=sys.stderr)
        raise
    app_rom_B = data[nfp_file]["ROM"]
    app_ram_B = data[nfp_file]["RAM"]

    nfp_file = "build/system.elf"
    nfp_benchmark = dfatool.runner.ShellMonitor(
        ("make", "nfpvalues"),
        cwd=args.multipass_base,
    )

    stdout, stderr = nfp_benchmark.run()
    try:
        data = json.loads(stdout[0])
    except json.decoder.JSONDecodeError:
        print(f"Unable to parse JSON output: {stdout}", file=sys.stderr)
        raise
    all_rom_B = data[nfp_file]["ROM"]
    all_ram_B = data[nfp_file]["RAM"]

    print(
        f"[::] {dfatool_name} | e_quantization={args.quantization} e_location={args.model_location} | "
        + f"all_ram_B={all_ram_B} all_rom_B={all_rom_B} app_ram_B={app_ram_B} app_rom_B={app_rom_B} "
        + f"latency_ms={np.mean(latencies)} main_ram_B={app_ram_B} main_rom_B={app_rom_B}"
    )
