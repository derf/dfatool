#!/usr/bin/env python3

import kconfiglib
import logging
import itertools
import os
import re
import shutil
import subprocess

from .utils import cd
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
        "config_hash": String("FIXME"),
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
        self.cwd = os.path.abspath(working_directory)
        self.clean_command = "make clean"
        self.build_command = "make"
        self.attribute_command = "make attributes"
        self.randconfig_command = "make randconfig"
        self.kconfig = "Kconfig"

    def randconfig(self):
        status = subprocess.run(
            self.randconfig_command.split(),
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
        return "unknown"

    def git_commit_id(self):
        try:
            status = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            revision = status.stdout.strip()
            return revision
        except FileNotFoundError:
            return "unknown"

    def file_hash(self, config_file):
        status = subprocess.run(
            ["sha256sum", config_file],
            cwd=self.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        try:
            sha256sum = status.stdout.split()[0]
        except IndexError:
            raise RuntimeError(
                f"Unable to extract hash from  'sha256sum {config_file}' output '{status.stdout}'"
            )
        return sha256sum

    def run_nfpkeys(self):
        nfpkeys = File("nfpkeys.json")
        with open(nfpkeys.path, "w") as out_fd:
            subprocess.check_call(["make", "nfpkeys"], cwd=self.cwd, stdout=out_fd)

    def run_randconfig(self):
        """Run a randomconfig experiment in the selected project. Results are written to the current working directory."""
        experiment = RandomConfig()
        experiment(
            [
                "--randconfig_seed",
                self.randconfig(),
                "--config_hash",
                self.file_hash(f"{self.cwd}/.config"),
                "--kconfig_hash",
                self.file_hash(f"{self.cwd}/{self.kconfig}"),
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
        success = os.path.exists(experiment.attributes.path)
        return {"success": success, "config_path": experiment.config.path}

    def config_is_functional(self, kconf):
        for choice in kconf.choices:
            if (
                not choice.is_optional
                and 2 in choice.assignable
                and choice.selection is None
            ):
                return False
        return True

    def _run_explore_experiment(self, kconf, kconf_hash, config_file):
        if not self.config_is_functional(kconf):
            logger.debug("Configuration is non-functional")
            kconf.load_config(config_file)
            return
        kconf.write_config(f"{self.cwd}/.config")
        experiment = ExploreConfig()
        experiment(
            [
                "--config_hash",
                self.file_hash(f"{self.cwd}/.config"),
                "--kconfig_hash",
                kconf_hash,
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

    def _can_be_handled_by_bdd(self, symbol):
        sym_type = kconfiglib.TYPE_TO_STR[symbol.type]
        return sym_type == "bool"

    def _dependencies_to_bdd_expr(self, depends_on):
        depends_on = kconfiglib.expr_str(depends_on)

        choices = list()

        for match in re.finditer(r"<choice ([^>]+)>", depends_on):
            choices.append(match.group(1))

        for choice in choices:
            depends_on = re.sub(f"<choice {choice}>", f"_choice_{choice}", depends_on)

        depends_on = re.sub("&& <([^>]+)>", "", depends_on)

        if depends_on == "n" or depends_on == "m" or depends_on == "y":
            return None

        depends_on = re.sub("&&", "&", depends_on)
        depends_on = re.sub("\|\|", "|", depends_on)

        return depends_on

    def enumerate(
        self, cudd=True, export_pdf=None, return_count=False, return_solutions=False
    ):
        if cudd:
            from dd.cudd import BDD
        else:
            from dd.autoref import BDD
        kconfig_file = f"{self.cwd}/{self.kconfig}"
        kconfig_hash = self.file_hash(kconfig_file)
        with cd(self.cwd):
            kconf = kconfiglib.Kconfig(kconfig_file)
        pre_variables = list()
        pre_expressions = list()
        for choice in kconf.choices:
            var_name = f"_choice_{choice.name}"
            pre_variables.append(var_name)

            # Build "exactly_one", expressing that exactly one of the symbols managed by this choice must be selected
            symbols = list(map(lambda sym: sym.name, choice.syms))
            exactly_one = list()
            for sym1 in symbols:
                subexpr = list()
                for sym2 in symbols:
                    if sym1 == sym2:
                        subexpr.append(sym2)
                    else:
                        subexpr.append(f"!{sym2}")
                exactly_one.append("(" + " & ".join(subexpr) + ")")
            exactly_one = " | ".join(exactly_one)

            # If the choice is selected, exactly one choice element must be selected
            pre_expressions.append(f"{var_name} -> ({exactly_one})")
            # Each choice symbol in exactly_once depends on the choice itself, which will lead to "{symbol} -> {var_name}" rules being generated later on. This
            # ensures that if the choice is false, each symbol is false too. We do not need to handle that case here.

            # The choice may depend on other variables
            depends_on = self._dependencies_to_bdd_expr(choice.direct_dep)

            if depends_on:
                if choice.is_optional:
                    pre_expressions.append(f"{var_name} -> {depends_on}")
                else:
                    pre_expressions.append(f"{var_name} <-> {depends_on}")
            elif not choice.is_optional:
                # Always active
                pre_expressions.append(var_name)

        for symbol in kconf.syms.values():
            if not self._can_be_handled_by_bdd(symbol):
                continue
            pre_variables.append(symbol.name)
            depends_on = self._dependencies_to_bdd_expr(symbol.direct_dep)
            if depends_on:
                pre_expressions.append(f"{symbol.name} -> {depends_on}")
            for selected_symbol, depends_on in symbol.selects:
                depends_on = self._dependencies_to_bdd_expr(depends_on)
                if depends_on:
                    pre_expressions.append(
                        f"({symbol.name} & ({depends_on})) -> {selected_symbol.name}"
                    )
                else:
                    pre_expressions.append(f"{symbol.name} -> {selected_symbol.name}")

        logger.debug("Variables:")
        logger.debug("\n".join(pre_variables))
        logger.debug("Expressions:")
        logger.debug("\n".join(pre_expressions))

        variables = list()
        expressions = list()

        bdd = BDD()
        variable_count = 0
        for variable in pre_variables:
            if variable[0] != "#":
                variables.append(variable)
                variable_count += 1
                bdd.declare(variable)
        logger.debug(f"Got {variable_count} variables")

        constraint = "True"
        expression_count = 0
        for expression in pre_expressions:
            if expression[0] != "#":
                expressions.append(expression)
                expression_count += 1
                constraint += f" & ({expression})"
        logger.debug(f"Got {expression_count} rules")
        logger.debug(constraint)

        constraint = bdd.add_expr(constraint)

        if cudd:
            # Egal?
            logger.debug("Reordering ...")
            BDD.reorder(bdd)

        else:
            # Wichtig! Lesbarkeit++ falls gedumpt wird, Performance vermutlich auch.
            logger.debug("Collecting Garbage ...")
            bdd.collect_garbage()

            # See <http://www.ecs.umass.edu/ece/labs/vlsicad/ece667/reading/somenzi99bdd.pdf> for how to read the graphical representation.
            # A solid line is followed if the origin node is 1
            # A dashed line is followed if the origin node is 0
            # A path from a top node to 1 satisfies the function iff the number of negations ("-1" annotations) is even
            if export_pdf is not None:
                logger.info(f"Dumping to {export_pdf} ...")
                bdd.dump(export_pdf)

        logger.debug("Solving ...")

        # still need to be set, otherwise autoref and cudd complain and set them anyways.
        # care_vars = list(filter(lambda x: "meta_" not in x and "_choice_" not in x, variables))

        if return_solutions:
            return bdd.pick_iter(constraint, care_vars=variables)

        if return_count:
            return len(bdd.pick_iter(constraint, care_vars=variables))

        config_file = f"{self.cwd}/.config"
        for solution in bdd.pick_iter(constraint, care_vars=variables):
            logger.debug(f"Set {solution}")
            with open(config_file, "w") as f:
                for k, v in solution.items():
                    if v:
                        print(f"CONFIG_{k}=y", file=f)
                    else:
                        print(f"# CONFIG_{k} is not set", file=f)
            with cd(self.cwd):
                kconf = kconfiglib.Kconfig(kconfig_file)
            kconf.load_config(config_file)

            int_values = list()
            int_names = list()
            for symbol in kconf.syms.values():
                if (
                    kconfiglib.TYPE_TO_STR[symbol.type] == "int"
                    and symbol.visibility
                    and symbol.ranges
                ):
                    for min_val, max_val, condition in symbol.ranges:
                        if condition.tri_value:
                            int_names.append(symbol.name)
                            min_val = int(min_val.str_value, 0)
                            max_val = int(max_val.str_value, 0)
                            step_size = (max_val - min_val) // 8
                            if step_size == 0:
                                step_size = 1
                            int_values.append(
                                list(range(min_val, max_val + 1, step_size))
                            )
                            continue

            for int_config in itertools.product(*int_values):
                for i, int_name in enumerate(int_names):
                    val = int_config[i]
                    symbol = kconf.syms[int_name]
                    logger.debug(f"Set {symbol.name} to {val}")
                    symbol.set_value(str(val))
                self._run_explore_experiment(kconf, kconfig_hash, config_file)

    def run_exploration_from_file(self, config_file, with_initial_config=True):
        kconfig_file = f"{self.cwd}/{self.kconfig}"
        kconfig_hash = self.file_hash(kconfig_file)
        with cd(self.cwd):
            kconf = kconfiglib.Kconfig(kconfig_file)
        kconf.load_config(config_file)

        if with_initial_config:
            experiment = ExploreConfig()
            shutil.copyfile(config_file, f"{self.cwd}/.config")
            experiment(
                [
                    "--config_hash",
                    self.file_hash(config_file),
                    "--kconfig_hash",
                    kconfig_hash,
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
            if kconfiglib.TYPE_TO_STR[symbol.type] == "bool":
                if symbol.tri_value == 0 and 2 in symbol.assignable:
                    logger.debug(f"Set {symbol.name} to y")
                    symbol.set_value(2)
                    self._run_explore_experiment(kconf, kconfig_hash, config_file)
                elif symbol.tri_value == 2 and 0 in symbol.assignable:
                    logger.debug(f"Set {symbol.name} to n")
                    symbol.set_value(0)
                    self._run_explore_experiment(kconf, kconfig_hash, config_file)
            elif (
                kconfiglib.TYPE_TO_STR[symbol.type] == "int"
                and symbol.visibility
                and symbol.ranges
            ):
                for min_val, max_val, condition in symbol.ranges:
                    if kconfiglib.expr_value(condition):
                        min_val = int(min_val.str_value, 0)
                        max_val = int(max_val.str_value, 0)
                        step_size = (max_val - min_val) // 5
                        if step_size == 0:
                            step_size = 1
                        for val in range(min_val, max_val + 1, step_size):
                            print(f"Set {symbol.name} to {val}")
                            symbol.set_value(str(val))
                            self._run_explore_experiment(
                                kconf, kconfig_hash, config_file
                            )
                        break
            else:
                continue
