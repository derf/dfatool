#!/usr/bin/env python3

import kconfiglib
import logging
import re
import shutil
import subprocess

from versuchung.experiment import Experiment
from versuchung.types import String, Bool, Integer
from versuchung.files import File, Directory

logger = logging.getLogger(__name__)


class AttributeExperiment(Experiment):
    outputs = {
        "config": File(".config"),
        "attributes": File("attributes.json"),
        "build_out": File("build.out"),
        "build_err": File("build.err"),
    }

    def run(self):
        build_command = self.build_command.value.split()
        attr_command = self.attr_command.value.split()
        shutil.copyfile(f"{self.project_root.path}/.config", self.config.path)
        subprocess.check_call(
            ["make", "clean"],
            cwd=self.project_root.path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            with open(self.build_out.path, "w") as out_fd, open(
                self.build_err.path, "w"
            ) as err_fd:
                subprocess.check_call(
                    build_command,
                    cwd=self.project_root.path,
                    stdout=out_fd,
                    stderr=err_fd,
                )
        except subprocess.CalledProcessError:
            logger.info("build error")
            return
        with open(self.attributes.path, "w") as attr_fd:
            subprocess.check_call(
                attr_command, cwd=self.project_root.path, stdout=attr_fd
            )


class RandomConfig(AttributeExperiment):
    inputs = {
        "randconfig_seed": String("FIXME"),
        "kconfig_hash": String("FIXME"),
        "project_root": Directory("/tmp"),
        "project_version": String("FIXME"),
        "clean_command": String("make clean"),
        "build_command": String("make"),
        "attr_command": String("make attributes"),
    }


class ExploreConfig(AttributeExperiment):
    inputs = {
        "config_hash": String("FIXME"),
        "kconfig_hash": String("FIXME"),
        "project_root": Directory("/tmp"),
        "project_version": String("FIXME"),
        "clean_command": String("make clean"),
        "build_command": String("make"),
        "attr_command": String("make attributes"),
    }


class KConfig:
    def __init__(self, working_directory):
        self.cwd = working_directory
        self.clean_command = "make clean"
        self.build_command = "make"
        self.attribute_command = "make attributes"

    def randconfig(self):
        status = subprocess.run(
            ["make", "randconfig"],
            cwd=self.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        # make randconfig occasionally generates illegal configurations, so a project may run randconfig more than once.
        # Make sure to return the seed of the latest run (don't short-circuit).
        seed = None
        for line in status.stderr.split("\n"):
            match = re.match("KCONFIG_SEED=(.*)", line)
            if match:
                seed = match.group(1)
        if seed:
            return seed
        raise RuntimeError("KCONFIG_SEED not found")

    def git_commit_id(self):
        status = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        revision = status.stdout.strip()
        return revision

    def file_hash(self, config_file):
        status = subprocess.run(
            ["sha256sum", config_file],
            cwd=self.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        sha256sum = status.stdout.split()[0]
        return sha256sum

    def run_randconfig(self):
        """Run a randomconfig experiment in the selected project. Results are written to the current working directory."""
        experiment = RandomConfig()
        experiment(
            [
                "--randconfig_seed",
                self.randconfig(),
                "--kconfig_hash",
                self.file_hash(f"{self.cwd}/Kconfig"),
                "--project_version",
                self.git_commit_id(),
                "--project_root",
                self.cwd,
                "--clean_command",
                self.clean_command,
                "--build_command",
                self.build_command,
                "--attr_command",
                self.attribute_command,
            ]
        )

    def config_is_functional(self, kconf):
        for choice in kconf.choices:
            if (
                not choice.is_optional
                and 2 in choice.assignable
                and choice.selection is None
            ):
                return False
        return True

    def run_exploration_from_file(self, config_file):
        kconfig_file = f"{self.cwd}/Kconfig"
        kconf = kconfiglib.Kconfig(kconfig_file)
        kconf.load_config(config_file)
        symbols = list(kconf.syms.keys())

        experiment = ExploreConfig()
        shutil.copyfile(config_file, f"{self.cwd}/.config")
        experiment(
            [
                "--config_hash",
                self.file_hash(config_file),
                "--kconfig_hash",
                self.file_hash(kconfig_file),
                "--project_version",
                self.git_commit_id(),
                "--project_root",
                self.cwd,
                "--clean_command",
                self.clean_command,
                "--build_command",
                self.build_command,
                "--attr_command",
                self.attribute_command,
            ]
        )

        for symbol in kconf.syms.values():
            if kconfiglib.TYPE_TO_STR[symbol.type] != "bool":
                continue
            if symbol.tri_value == 0 and 2 in symbol.assignable:
                logger.debug(f"Set {symbol.name} to y")
                symbol.set_value(2)
            elif symbol.tri_value == 2 and 0 in symbol.assignable:
                logger.debug(f"Set {symbol.name} to n")
                symbol.set_value(0)
            else:
                continue

            if not self.config_is_functional(kconf):
                logger.debug("Configuration is non-functional")
                kconf.load_config(config_file)
                continue

            kconf.write_config(f"{self.cwd}/.config")
            experiment = ExploreConfig()
            experiment(
                [
                    "--config_hash",
                    self.file_hash(f"{self.cwd}/.config"),
                    "--kconfig_hash",
                    self.file_hash(kconfig_file),
                    "--project_version",
                    self.git_commit_id(),
                    "--project_root",
                    self.cwd,
                    "--clean_command",
                    self.clean_command,
                    "--build_command",
                    self.build_command,
                    "--attr_command",
                    self.attribute_command,
                ]
            )
            kconf.load_config(config_file)
