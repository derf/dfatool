#!/usr/bin/env python3

"""explore-kconfig - Obtain build attributes of configuration variants

explore-kconfig obtains build attributes such as ROM or RAM usage of
configuration variants for a given software project. It works on random
random configurations (--random) or in the neighbourhood
of existing configurations (--neighbourhood).

Supported projects must be configurable via kconfig and provide a command which
outputs a JSON dict of build attributes on stdout. Use
--{clean,build,attribute}-command to configure explore-kconfig for a project.
"""

import argparse
import logging
import os
import sys

from dfatool import kconfig

from versuchung.experiment import Experiment
from versuchung.types import String, Bool, Integer
from versuchung.files import File, Directory


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
        "--random",
        type=int,
        help="Explore a number of random configurations (make randconfig)",
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
        default="make attributes",
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

    if args.random:
        for i in range(args.random):
            logging.info(f"Running randconfig {i+1} of {args.random}")
            kconf.run_randconfig()

    if args.neighbourhood:
        if os.path.isfile(args.neighbourhood):
            kconf.run_exploration_from_file(args.neighbourhood)
        elif os.path.isdir(args.neighbourhood):
            for filename in os.listdir(args.neighbourhood):
                config_filename = f"{args.neighbourhood}/{filename}"
                logging.info(f"Exploring neighbourhood of {config_filename}")
                kconf.run_exploration_from_file(config_filename)
        else:
            print(
                f"--neighbourhod: Error: {args.neighbourhood} must be a file or directory, but is neither",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
