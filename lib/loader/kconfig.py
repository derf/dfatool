#!/usr/bin/env python3

from .. import kconfiglib
from dfatool.utils import cd
import json
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


class KConfigAttributes:
    def __init__(self, kconfig_path, datadir):
        experiments = list()
        failed_experiments = list()
        if not datadir is None:
            for direntry in os.listdir(datadir):
                config_path = f"{datadir}/{direntry}/.config"
                attr_path = f"{datadir}/{direntry}/attributes.json"
                metadata_path = f"{datadir}/{direntry}/metadata"
                if os.path.exists(attr_path):
                    experiments.append((config_path, attr_path))
                elif os.path.exists(config_path):
                    failed_experiments.append(config_path)

        self.kconfig_root = "/".join(kconfig_path.split("/")[:-1])
        kconfig_file = kconfig_path.split("/")[-1]

        if self.kconfig_root == "":
            self.kconfig_root = "."

        with cd(self.kconfig_root):
            kconf = kconfiglib.Kconfig(kconfig_file)
        self.kconf = kconf

        self.kconfig_hash = self.file_hash(kconfig_path)
        self.kconfig_dir = "unknown"
        if "/" in kconfig_path:
            self.kconfig_dir = kconfig_path.split("/")[-2]

        accepted_symbol_types = "bool tristate int string hex".split()

        if bool(int(os.getenv("DFATOOL_KCONF_IGNORE_NUMERIC", 0))):
            accepted_symbol_types.remove("int")
            accepted_symbol_types.remove("hex")

        if bool(int(os.getenv("DFATOOL_KCONF_IGNORE_STRING", 0))):
            accepted_symbol_types.remove("string")

        self.symbol_names = sorted(
            map(
                lambda sym: sym.name,
                filter(
                    lambda sym: kconfiglib.TYPE_TO_STR[sym.type]
                    in accepted_symbol_types,
                    kconf.syms.values(),
                ),
            )
        )

        unnamed_choice_index = 0
        for choice in kconf.choices:
            if choice.name is None:
                choice.name = f"UNNAMED_CHOICE_{unnamed_choice_index}"
                unnamed_choice_index += 1

        self.choice_names = sorted(map(lambda choice: choice.name, kconf.choices))

        self.choice = dict()
        self.choice_symbol_names = list()
        for choice in kconf.choices:
            self.choice[choice.name] = choice
            self.choice_symbol_names.extend(map(lambda sym: sym.name, choice.syms))

        self.symbol = dict()
        for symbol_name in self.symbol_names:
            self.symbol[symbol_name] = kconf.syms[symbol_name]

        if int(os.getenv("DFATOOL_KCONF_WITH_CHOICE_NODES", 1)):
            for sym_name in self.choice_symbol_names:
                try:
                    self.symbol_names.remove(sym_name)
                except ValueError:
                    logger.error(
                        f"Trying to remove choice {sym_name}, but it is not present in the symbol list"
                    )
                    raise
            self.param_names = self.symbol_names + self.choice_names
        else:
            self.param_names = self.symbol_names
            self.choice_names = list()

        self.data = list()
        self.configs = list()
        self.failures = list()

        for config_path, attr_path in experiments:
            self.configs.append(config_path)
            kconf.load_config(config_path)
            with open(attr_path, "r") as f:
                attr = json.load(f)

            param = self._conf_to_param()
            self.data.append((param, attr))

        for config_path in failed_experiments:
            kconf.load_config(config_path)
            param = self._conf_to_param()
            self.failures.append(param)

    def _conf_to_param(self):
        param = dict()
        for sym_name in self.symbol_names:
            sym = self.kconf.syms[sym_name]
            if not sym.visibility and sym.str_value == "":
                param[sym_name] = None
            elif kconfiglib.TYPE_TO_STR[sym.type] in ("int", "hex"):
                try:
                    param[sym_name] = int(sym.str_value, base=0)
                except ValueError:
                    print(
                        f"Warning: Illegal value for {sym.__repr__()}, defaulting to None"
                    )
                    param[sym_name] = None
            else:
                param[sym_name] = sym.str_value
        for choice in self.choice_names:
            if self.choice[choice].selection is None:
                param[choice] = None
            else:
                param[choice] = self.choice[choice].selection.name
        return param

    def file_hash(self, config_file):
        status = subprocess.run(
            ["sha256sum", config_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        sha256sum = status.stdout.split()[0]
        return sha256sum

    def to_dref(self) -> dict:
        type_count = {
            "bool": 0,
            "tristate": 0,
            "numeric": 0,
            "string": 0,
            "enum": 0,
        }
        for param_name in self.param_names:
            if param_name in self.symbol:
                str_type = kconfiglib.TYPE_TO_STR[self.symbol[param_name].type]
                if str_type in type_count:
                    type_count[str_type] += 1
                elif str_type in ("int", "hex"):
                    type_count["numeric"] += 1
                else:
                    raise RuntimeError(
                        f"param {param_name} has unknown type: {str_type}"
                    )
            else:
                type_count["enum"] += 1
        return {
            "raw measurements/valid": len(self.configs),
            "raw measurements/total": len(self.configs) + len(self.failures),
            "kconfig/features/total": len(self.param_names),
            "kconfig/features/bool": type_count["bool"],
            "kconfig/features/tristate": type_count["tristate"],
            "kconfig/features/string": type_count["string"],
            "kconfig/features/numeric": type_count["numeric"],
            "kconfig/features/enum": type_count["enum"],
        }
