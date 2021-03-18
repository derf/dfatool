#!/usr/bin/env python3

import argparse
import io
import json
import sys
import tarfile

from dfatool.utils import soft_cast_float


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "--config",
        metavar="<key>=<value>[,<key>=<value>,...]",
        type=str,
        help="Add config to archived ptalog.json file",
    )
    parser.add_argument("ptalog_file", type=str, nargs=1)
    parser.add_argument("energy_files", nargs="+")
    args = parser.parse_args()

    with open(args.ptalog_file[0], "r") as f:
        ptalog = json.load(f)

    ptalog_dir = "/".join(args.ptalog_file[0].split("/")[:-1]) + "/"
    ptalog_prefix = ".".join(args.ptalog_file[0].split(".")[:-1])
    tar_file = f"{ptalog_prefix}.tar"

    if args.config:
        for kv in args.config.split(","):
            k, v = kv.split("=")
            v = soft_cast_float(v)
            ptalog["configs"][0][k] = v

    # TODO optional config (z.B. offset / limit) -> ptalog["configs"].append(...)

    with tarfile.open(tar_file, "w") as tf:
        for energy_file in args.energy_files:
            energy_dir = "/".join(energy_file.split("/")[:-1]) + "/"
            energy_target = energy_file.removeprefix(energy_dir)
            tf.add(energy_file, energy_target)
            ptalog["files"][0].append(energy_target)

        ptalog_content = json.dumps(ptalog).encode("utf-8")
        t = tarfile.TarInfo("ptalog.json")
        t.size = len(ptalog_content)
        t.uid = 1000
        t.gid = 1000
        t.mode = 0o600
        tf.addfile(t, io.BytesIO(ptalog_content))
