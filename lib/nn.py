#!/usr/bin/env python3

from .utils import flatten
import re


class LayerInfo:
    def __init__(self, line):
        node_type, start, first, avg, _, _, _, _, name = line.split(", ")
        self.node_type = node_type
        self.avg_ms = float(avg)

        name = name.rstrip("0123456789")

        name = name.removeprefix("[")
        name = name.removesuffix("]:")

        self.name = name
        self.ops = name.split(";")
        # xnnpack separates via "\t "
        self.ops = flatten(map(lambda op: op.split("\t "), self.ops))
        self._matched_layers = list()
        self._match_complete = False

        self.blocks = set()

        for op in self.ops:
            subs = op.split("/")
            if len(subs) > 1:
                self.blocks.add(subs[1])

        # print(f"{self.node_type:30s} {self.avg_ms:.2f} {self.ops}")

    def __repr__(self):
        return f"<{self.node_type} {self.ops}>"

    def match_tf_layer(self, tf_layer):
        for op in self.ops:
            if f"/{tf_layer.name}/" in op or f"/{tf_layer.name}_" in op:
                self._matched_layers.append(tf_layer)
                if len(self.ops) == len(self._matched_layers):
                    self._match_complete = True
                return True
        return False

    def match_complete(self, seen_ops):
        if self._match_complete:
            return True
        for need_op in self.ops:
            found = False
            for have_op in seen_ops:
                if f"/{have_op}/" in need_op or f"/{have_op}_" in need_op:
                    found = True
                    break
            if not found:
                return False
        return True


def load_tflite_profiling_csv(filename):
    layers = list()
    state = "intro"
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if (
                state == "intro"
                and line == "Operator-wise Profiling Info for Regular Benchmark Runs:"
            ):
                state = "opheader"
            elif (
                state == "opheader"
                and line
                == "node type, start, first, avg_ms, %, cdf%, mem KB, times called, name"
            ):
                state = "ops"
            elif state == "ops" and line == "":
                state = "intro2"
            elif state == "ops":
                layers.append(LayerInfo(line))
    return layers


def load_tflite(filename):
    num_threads = None
    with_xnn = False
    model_size = None
    memory_footprint = None
    inference_time = None
    with open(filename, "r") as f:
        for line in f:
            match = re.match(r"Num threads: \[(\d+)\]", line)
            if match:
                num_threads = int(match.group(1))
            match = re.match(r"The input model file size \(MB\): ([0-9.]+)", line)
            if match:
                model_size = float(match.group(1))
            match = re.match(
                r"Peak memory footprint \(MB\): init=[0-9.e+-]+ overall=([0-9.e+-]+)",
                line,
            )
            if match:
                memory_footprint = float(match.group(1))
            match = re.match(
                r"Inference timings in us: Init: [0-9.e+-]+, First inference: [0-9.e+-]+, Warmup \(avg\): [0-9.e+-]+, Inference \(avg\): ([0-9.e+-]+)",
                line,
            )
            if match:
                inference_time = float(match.group(1))
            if line == "Use xnnpack: [1]":
                with_xnn = True
    return {
        "num_threads": num_threads,
        "with_xnn": with_xnn,
        "model_size_mb": model_size,
        "memory_footprint_mb": memory_footprint,
        "inference_time_us": inference_time,
    }
