#!/usr/bin/env python3

"""explore-kconfig - Obtain build attributes of configuration variants

explore-kconfig obtains build attributes such as ROM or RAM usage of
configuration variants for a given software project. It works on random
configurations (--random) or in the neighbourhood of existing configurations
(--neighbourhood).

Supported projects must be configurable via kconfig and provide a command which
outputs a JSON dict of build attributes on stdout. Use
--{clean,build,attribute}-command to configure explore-kconfig for a project.

explore-kconfig places the experiment results (containing configurations, build
logs, and correspondnig attributes) in the current working directory. Use
analyze-kconfig to build a model once data acquisition is complete.
"""

import argparse
import logging
import os
import sys
import time

from dfatool import kconfig


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "--neighbourhood",
        type=str,
        help="Explore neighbourhood of provided .config file(s)",
    )
    parser.add_argument(
        "--log-level",
        default=logging.INFO,
        type=lambda level: getattr(logging, level.upper()),
        help="Set log level",
    )
    parser.add_argument(
        "--enumerate",
        action="store_true",
        help="Enumerate all valid configurations (DEPRECATED and UNMAINTAINED)",
    )
    parser.add_argument(
        "--random",
        type=int,
        help="Explore a number of random configurations (make randconfig)",
    )
    parser.add_argument(
        "--random-int",
        action="store_true",
        help="Randomize integers after running make randconfig",
    )
    parser.add_argument(
        "--with-neighbourhood",
        action="store_true",
        help="Explore neighbourhood of successful random configurations",
    )
    parser.add_argument(
        "--repeatable",
        action="store_true",
        help="Allow repeated measurements of already benchmarked configurations",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        metavar="N",
        help="Run each benchmark N times. Implies --repeatable",
    )
    parser.add_argument(
        "--clean-command", type=str, help="Clean command", default="make clean"
    )
    parser.add_argument(
        "--build-command", type=str, help="Build command", default="make"
    )
    parser.add_argument(
        "--attribute-command",
        type=str,
        help="Attribute extraction command",
        default="make nfpvalues",
    )
    parser.add_argument(
        "--randconfig-command",
        type=str,
        help="Randconfig command for --random",
        default="make randconfig",
    )
    parser.add_argument(
        "--kconfig-file", type=str, help="Kconfig file", default="Kconfig"
    )
    parser.add_argument("project_root", type=str, help="Project root directory")

    args = parser.parse_args()

    if isinstance(args.log_level, int):
        logging.basicConfig(level=args.log_level)
    else:
        print(f"Invalid log level. Setting log level to INFO.", file=sys.stderr)

    kconf = kconfig.KConfig(args.project_root)

    if args.clean_command:
        kconf.clean_command = args.clean_command
    if args.build_command:
        kconf.build_command = args.build_command
    if args.attribute_command:
        kconf.attribute_command = args.attribute_command
    if args.randconfig_command:
        kconf.randconfig_command = args.randconfig_command
    if args.kconfig_file:
        kconf.kconfig = args.kconfig_file
    if args.repeatable:
        kconf.repeatable = args.repeatable
    if args.repeat:
        kconf.repeat = args.repeat - 1
        kconf.repeatable = True

    kconf.run_nfpkeys()

    if args.enumerate:
        kconf.enumerate()

    if args.random:
        num_successful = 0
        total_randconfig_seconds = 0
        # Assumption: At least 1% of builds are successful
        for i in range(args.random * 100):
            logging.info(f"Running randconfig {num_successful+1} of {args.random}")
            if total_randconfig_seconds and num_successful:
                seconds_per_randconfig = total_randconfig_seconds / num_successful
                remaining_minutes = (
                    int(seconds_per_randconfig * (args.random - num_successful)) // 60
                )
                logging.info(
                    f"Estimated remaining exploration time: {remaining_minutes // (24*60):2d} days {(remaining_minutes % (24*60)) // 60:2d} hours {remaining_minutes % 60:2d} minutes"
                )
            randconfig_start_time = time.time()
            status = kconf.run_randconfig(with_random_int=args.random_int)
            if status["success"]:
                num_successful += 1
            if args.with_neighbourhood and status["success"]:
                config_filename = status["config_path"]
                logging.debug(f"Exploring neighbourhood of {config_filename}")
                kconf.run_exploration_from_file(
                    config_filename, with_initial_config=False
                )
            total_randconfig_seconds += time.time() - randconfig_start_time
            if num_successful == args.random:
                break

    if args.neighbourhood:
        if os.path.isfile(args.neighbourhood):
            kconf.run_exploration_from_file(args.neighbourhood)
        elif os.path.isdir(args.neighbourhood):
            for filename in os.listdir(args.neighbourhood):
                config_filename = f"{args.neighbourhood}/{filename}"
                logging.debug(f"Exploring neighbourhood of {config_filename}")
                kconf.run_exploration_from_file(config_filename)
        else:
            print(
                f"--neighbourhod: Error: {args.neighbourhood} must be a file or directory, but is neither",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
