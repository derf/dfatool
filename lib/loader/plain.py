#!/usr/bin/env python3

from ..utils import soft_cast_int_or_float, soft_cast_float
import os
import re


class CSVfile:
    def __init__(self):
        self.ignore_names = os.environ.get("DFATOOL_CSV_IGNORE", "").split(",")
        self.observation_names = os.environ.get("DFATOOL_CSV_OBSERVATIONS", "").split(
            ","
        )
        pass

    def load(self, f):
        self.column_type = dict()
        observations = list()
        param_names = list()
        attr_names = list()
        for lineno, line in enumerate(f):
            line = line.removesuffix("\n")
            if lineno == 0:
                for i, col_name in enumerate(line.split(",")):
                    if col_name in self.ignore_names:
                        self.column_type[i] = 0
                    elif col_name in self.observation_names:
                        self.column_type[i] = 2
                        attr_names.append(col_name)
                    else:
                        self.column_type[i] = 1
                        param_names.append(col_name)
            else:
                param_values = list(
                    map(
                        soft_cast_int_or_float,
                        map(
                            lambda iv: iv[1],
                            filter(
                                lambda iv: self.column_type[iv[0]] == 1,
                                enumerate(line.split(",")),
                            ),
                        ),
                    )
                )
                attr_values = list(
                    map(
                        soft_cast_float,
                        map(
                            lambda iv: iv[1],
                            filter(
                                lambda iv: self.column_type[iv[0]] == 2,
                                enumerate(line.split(",")),
                            ),
                        ),
                    )
                )
                observations.append(
                    {
                        "name": "CSVFile",
                        "param": dict(zip(param_names, param_values)),
                        "attribute": dict(zip(attr_names, attr_values)),
                    }
                )
        return observations


class Logfile:
    def __init__(self):
        pass

    def kv_to_param(self, kv_str, cast):
        try:
            key, value = kv_str.split("=")
            value = cast(value)
            return key, value
        except ValueError:
            logger.warning(f"Invalid key-value pair: {kv_str}")
            raise

    def kv_to_param_f(self, kv_str):
        return self.kv_to_param(kv_str, soft_cast_float)

    def kv_to_param_i(self, kv_str):
        return self.kv_to_param(kv_str, soft_cast_int_or_float)

    def load(self, f):
        observations = list()
        for lineno, line in enumerate(f):
            m = re.search(r"\[::\] *([^|]*?) *[|] *([^|]*?) *[|] *(.*)", line)
            if m:
                name_str = m.group(1)
                param_str = m.group(2)
                attr_str = m.group(3)
                try:
                    param = dict(map(self.kv_to_param_i, param_str.split()))
                    attr = dict(map(self.kv_to_param_f, attr_str.split()))
                    observations.append(
                        {
                            "name": name_str,
                            "param": param,
                            "attribute": attr,
                        }
                    )
                except ValueError:
                    logger.warning(
                        f"Error parsing {f}: invalid key-value pair in line {lineno+1}"
                    )
                    logger.warning(f"Offending entry:\n{line}")
                    raise

        return observations

    def dump(self, observations, f):
        for observation in observations:
            name = observation["name"]
            param = observation["param"]
            attr = observation["attribute"]

            param_str = " ".join(
                map(
                    lambda kv: f"{kv[0]}={kv[1]}",
                    sorted(param.items(), key=lambda kv: kv[0]),
                )
            )
            attr_str = " ".join(
                map(
                    lambda kv: f"{kv[0]}={kv[1]}",
                    sorted(attr.items(), key=lambda kv: kv[0]),
                )
            )

            print(f"[::] {name} | {param_str} | {attr_str}", file=f)
