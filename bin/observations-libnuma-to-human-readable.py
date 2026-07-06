#!/usr/bin/env python3

import argparse
import dfatool.loader
import dfatool.utils
from functools import reduce


def get_memory_type(host, numa_node):
    if host == "milos":
        if numa_node <= 7:
            return "DDR4"
        if numa_node <= 15:
            return "HBM"
        if numa_node <= 16:
            return "CXL-x16"
        if numa_node <= 17:
            return "CXL-x8"
    raise RuntimeError(f"Unknown configuration: host={host}, numa_node={numa_node}")


def get_memory_locality(host, numa_cpu, numa_ram):
    if host == "milos":
        if numa_cpu == numa_ram or numa_cpu + 8 == numa_ram:
            return "local"
        if numa_cpu in range(0, 4):
            if numa_ram in range(0, 4):
                return "same-socket"
            if numa_ram in range(4, 8):
                return "cross-socket"
            if numa_ram in range(8, 12):
                return "same-socket"
            if numa_ram in range(12, 16):
                return "cross-socket"
            if numa_ram == 16:
                return "same-socket"
            if numa_ram == 17:
                return "cross-socket"
        if numa_cpu in range(4, 8):
            if numa_ram in range(0, 4):
                return "cross-socket"
            if numa_ram in range(4, 8):
                return "same-socket"
            if numa_ram in range(8, 12):
                return "cross-socket"
            if numa_ram in range(12, 16):
                return "same-socket"
            if numa_ram == 16:
                return "cross-socket"
            if numa_ram == 17:
                return "same-socket"
    raise RuntimeError(
        f"Unknown configuration: host={host}, numa_cpu={numa_cpu}, numa_ram={numa_ram}"
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "--host",
        choices=("milos",),
    )
    parser.add_argument(
        "logfiles",
        nargs="+",
        type=str,
        help="Path to benchmark output (.txt or .txt.xz)",
    )
    args = parser.parse_args()

    observations = reduce(
        lambda a, b: a + b, map(dfatool.loader.logfile_to_observations, args.logfiles)
    )

    for observation in observations:
        param = observation["param"]
        assert param.get("numa_node_in", -1) != -1
        assert param.get("numa_node_cpu", -1) != -1

        param["mem_in_type"] = get_memory_type(args.host, param["numa_node_in"])
        param["mem_in_locality"] = get_memory_locality(
            args.host, param["numa_node_cpu"], param["numa_node_in"]
        )

        if "numa_node_out" in param:
            assert param["numa_node_out"] != -1
            param["mem_out_type"] = get_memory_type(args.host, param["numa_node_out"])
            param["mem_out_locality"] = get_memory_locality(
                args.host, param["numa_node_cpu"], param["numa_node_out"]
            )

            param.pop("numa_node_out")
            param.pop("numa_distance_cpu_out")

        param.pop("numa_node_in")
        param.pop("numa_node_cpu")
        param.pop("numa_distance_in_cpu")

    dfatool.utils.observations_to_stdout(observations)


if __name__ == "__main__":
    main()
