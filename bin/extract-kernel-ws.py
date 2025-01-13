#!/usr/bin/env python3

import argparse
import numpy as np
import sys
import logging


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        choices=["debug", "info", "warning", "error"],
        default="warning",
        help="Set log level",
    )
    parser.add_argument(
        "--output-format",
        metavar="FORMAT",
        choices=["dfatool", "valgrind-ws"],
        default="dfatool",
        help="Set output format",
    )
    parser.add_argument(
        "benchmark_file",
        type=str,
        help="Benchmark file used to run valgrind-ws",
    )
    parser.add_argument(
        "ws_output",
        type=str,
        help="valgrind-ws output file",
    )

    args = parser.parse_args()
    benchmark_filename = args.benchmark_file.split("/")[-1]

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

    with open(args.benchmark_file, "r") as f:
        start_range = [None, None]
        end_range = [None, None]
        in_nop = False
        for lineno, line in enumerate(f):
            line = line.strip()
            if line == "#if NOP_SYNC":
                in_nop = True
                if start_range[0] is None:
                    start_range[0] = lineno
                else:
                    end_range[0] = lineno
            if in_nop and line.startswith("#endif"):
                in_nop = False
                if start_range[1] is None:
                    start_range[1] = lineno
                else:
                    end_range[1] = lineno

    logging.debug(f"start_range = {start_range}, end_range = {end_range}")

    page_size = None
    ws_log = list()
    sample_info = dict()
    with open(args.ws_output, "r") as f:
        in_ws_log = False
        in_sample_info = False
        for line in f:
            line = line.strip()
            if in_ws_log and line == "":
                in_ws_log = False
            if in_sample_info and line == "":
                in_sample_info = False
            if page_size is None and line.startswith("Page size:"):
                page_size = int(line.split()[2])

            if in_ws_log:
                t, wss_i, wss_d, info_ref = line.split()
                ws_log.append((int(t), int(wss_i), int(wss_d), info_ref))
            elif in_sample_info:
                _, info_ref, _, locs = line.split()
                info_ref = info_ref.removesuffix("]")
                locs = locs.removeprefix("loc=")
                sample_info[info_ref] = list()
                for loc in filter(lambda x: len(x), locs.split("|")):
                    filename, lineno = loc.split(":")
                    sample_info[info_ref].append((filename, int(lineno)))

            if line == "t WSS_insn WSS_data info":
                in_ws_log = True
            if line == "Sample info:":
                in_sample_info = True

    if page_size is None:
        raise RuntimeError("Unable to determine page size fom {args.ws_output}")

    logging.debug(f"sample_info = {sample_info}")
    next_in_kernel = False
    in_kernel = False
    insn_working_set_sizes = list()
    data_working_set_sizes = list()
    kernel_range = [None, None]
    for t, wss_i, wss_d, info_ref in ws_log:
        if next_in_kernel:
            next_in_kernel = False
            in_kernel = True
            kernel_range[0] = t

        if info_ref != "-":
            for filename, lineno in sample_info[info_ref]:
                if (
                    filename == benchmark_filename
                    and start_range[0] <= lineno <= start_range[1]
                ):
                    next_in_kernel = True
                elif (
                    filename == benchmark_filename
                    and end_range[0] <= lineno <= end_range[1]
                ):
                    in_kernel = False

        if in_kernel:
            data_working_set_sizes.append(wss_d * page_size)
            insn_working_set_sizes.append(wss_i * page_size)
            kernel_range[1] = t

    if args.output_format == "dfatool":
        print(
            f"wss_data_mean_bytes={np.mean(data_working_set_sizes)}"
            + f" wss_data_median_bytes={np.median(data_working_set_sizes)}"
            + f" wss_data_stddev={np.std(data_working_set_sizes)}"
            + f" wss_insn_mean_bytes={np.mean(insn_working_set_sizes)}"
            + f" wss_insn_median_bytes={np.median(insn_working_set_sizes)}"
            + f" wss_insn_stddev={np.std(insn_working_set_sizes)}"
        )
    elif args.output_format == "valgrind-ws":
        with open(args.ws_output, "r") as f:
            in_ws_log = False
            for line in f:
                if in_ws_log and line.strip() == "":
                    in_ws_log = False

                if in_ws_log:
                    ts = int(line.strip().split()[0])
                    if kernel_range[0] <= ts <= kernel_range[1]:
                        print(line, end="")
                else:
                    print(line, end="")

                if line.strip() == "t WSS_insn WSS_data info":
                    in_ws_log = True


if __name__ == "__main__":
    main()
