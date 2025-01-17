#!/usr/bin/env python3

import sys


def main(perf_line, rapl_names, rapl_start, rapl_stop):
    duration_ns = int(perf_line.split(",")[3])

    rapl_names = rapl_names.split()
    rapl_start = rapl_start.split()
    rapl_stop = rapl_stop.split()

    buf = [f"duration_ns={duration_ns}"]

    for i in range(len(rapl_names)):
        uj_start = int(rapl_start[i])
        uj_stop = int(rapl_stop[i])
        buf.append(f"{rapl_names[i]}_energy_uj={uj_stop - uj_start}")
        buf.append(
            f"{rapl_names[i]}_power_W={(uj_stop - uj_start) * 1000 / duration_ns}"
        )

    print(" ".join(buf))


if __name__ == "__main__":
    main(*sys.argv[1:])
