#!/usr/bin/env python3

from ..utils import soft_cast_int_or_float, soft_cast_float
import os
import re

import logging

logger = logging.getLogger(__name__)


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


class TraceAnnotation:
    offset = None
    name = None
    param = dict()

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def apply_offset(self, offset):
        self.offset += offset
        return self

    def __repr__(self):
        param_desc = " ".join(map(lambda kv: f"{kv[0]}={kv[1]}", self.param.items()))
        return f"{self.name}<{param_desc} @ {self.offset}>"


class RunAnnotation:
    name = None
    start = None
    kernels = list()
    end = None

    # start: offset points to first run entry
    # kernel: offset points to first kernel run entry
    # end: offset points to first non-run entry (i.e., for all run entries: offset < end.offset)

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def apply_offset(self, offset):
        self.start.apply_offset(offset)
        for kernel in self.kernels:
            kernel.apply_offset(offset)
        self.end.apply_offset(offset)
        return self

    def __repr__(self):
        return f"RunAnnotation<{self.name}, start={self.start}, kernels={self.kernels}, end={self.end}>"


class Logfile:
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

    def load(self, f, is_trace=False):
        observations = list()
        if is_trace:
            trace_status = None
            trace_start = None
            trace_kernels = list()
            trace_end = None
            annotations = list()

        for lineno, line in enumerate(f):
            if m := re.search(r"\[::\] *([^|]*?) *[|] *([^|]*?) *[|] *(.*)", line):
                name_str = m.group(1)
                param_str = m.group(2)
                attr_str = m.group(3)
                if is_trace:
                    name_str, name_annot = name_str.split("@")
                    name_str = name_str.strip()
                    name_annot = name_annot.strip()
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
                    if is_trace:
                        observations[-1]["place"] = name_annot
                except ValueError:
                    logger.warning(
                        f"Error parsing {f}: invalid key-value pair in line {lineno+1}"
                    )
                    logger.warning(f"Offending entry:\n{line}")
                    raise

            # only relevant for is_trace == True
            if m := re.fullmatch(r"\[>>\] *([^|]*?) *[|] *([^|]*?) *", line):
                trace_status = 1
                trace_kernels = list()
                name_str = m.group(1)
                param_str = m.group(2)
                try:
                    param = dict(map(self.kv_to_param_i, param_str.split()))
                except ValueError:
                    logger.warning(
                        f"Error parsing {f}: invalid key-value pair in line {lineno+1}"
                    )
                    logger.warning(f"Offending entry:\n{line}")
                    raise
                trace_start = TraceAnnotation(
                    offset=len(observations), name=name_str, param=param
                )

            if m := re.fullmatch(r"\[--\] *([^|]*?) *[|] *([^|]*?) *", line):
                trace_status = 2
                name_str = m.group(1)
                param_str = m.group(2)
                try:
                    param = dict(map(self.kv_to_param_i, param_str.split()))
                except ValueError:
                    logger.warning(
                        f"Error parsing {f}: invalid key-value pair in line {lineno+1}"
                    )
                    logger.warning(f"Offending entry:\n{line}")
                    raise
                trace_kernels.append(
                    TraceAnnotation(
                        offset=len(observations), name=name_str, param=param
                    )
                )

            if m := re.fullmatch(r"\[<<\] *([^|]*?) *[|] *([^|]*?) *", line):
                trace_status = None
                name_str = m.group(1)
                param_str = m.group(2)
                try:
                    param = dict(map(self.kv_to_param_i, param_str.split()))
                except ValueError:
                    logger.warning(
                        f"Error parsing {f}: invalid key-value pair in line {lineno+1}"
                    )
                    logger.warning(f"Offending entry:\n{line}")
                    raise
                trace_end = TraceAnnotation(
                    offset=len(observations), name=name_str, param=param
                )
                if trace_start is not None:
                    assert trace_start.name == trace_end.name
                    for kernel in trace_kernels:
                        assert trace_start.name == kernel.name
                    annotations.append(
                        RunAnnotation(
                            name=trace_start.name,
                            start=trace_start,
                            kernels=trace_kernels,
                            end=trace_end,
                        )
                    )

                trace_status = None
                trace_start = None
                trace_kernels = list()
                trace_end = None

        if is_trace:
            return observations, annotations
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
