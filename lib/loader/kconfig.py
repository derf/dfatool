#!/usr/bin/env python3

import kconfiglib
from frozendict import frozendict
import json
import os
import subprocess


class KConfigAttributes:
    def __init__(self, kconfig_path, datadir):
        experiments = list()
        failed_experiments = list()
        for direntry in os.listdir(datadir):
            config_path = f"{datadir}/{direntry}/.config"
            attr_path = f"{datadir}/{direntry}/attributes.json"
            metadata_path = f"{datadir}/{direntry}/metadata"
            if os.path.exists(attr_path):
                experiments.append((config_path, attr_path))
            elif os.path.exists(config_path):
                failed_experiments.append(config_path)

        kconf = kconfiglib.Kconfig(kconfig_path)
        self.kconf = kconf

        self.kconfig_hash = self.file_hash(kconfig_path)
        self.kconfig_dir = "unknown"
        if "/" in kconfig_path:
            self.kconfig_dir = kconfig_path.split("/")[-2]

        self.symbol_names = sorted(
            map(
                lambda sym: sym.name,
                filter(
                    lambda sym: kconfiglib.TYPE_TO_STR[sym.type]
                    in ("bool", "tristate", "int", "string", "hex"),
                    kconf.syms.values(),
                ),
            )
        )

        self.choice_names = sorted(
            map(lambda choice: choice.name or choice.name_and_loc, kconf.choices)
        )

        self.choice = dict()
        self.choice_symbol_names = list()
        for choice in kconf.choices:
            self.choice[choice.name or choice.name_and_loc] = choice
            self.choice_symbol_names.extend(map(lambda sym: sym.name, choice.syms))

        self.symbol = dict()
        for symbol_name in self.symbol_names:
            self.symbol[symbol_name] = kconf.syms[symbol_name]

        if int(os.getenv("DFATOOL_KCONF_WITH_CHOICE_NODES", 1)):
            for sym_name in self.choice_symbol_names:
                self.symbol_names.remove(sym_name)
            self.param_names = self.symbol_names + self.choice_names
        else:
            self.param_names = self.symbol_names

        self.data = list()
        self.configs = list()
        self.failures = list()

        for config_path, attr_path in experiments:
            self.configs.append(config_path)
            kconf.load_config(config_path)
            with open(attr_path, "r") as f:
                attr = json.load(f)

            param = self._conf_to_param()
            self.data.append((frozendict(param), attr))

        for config_path in failed_experiments:
            kconf.load_config(config_path)
            param = self._conf_to_param()
            self.failures.append(frozendict(param))

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
