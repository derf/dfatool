#!/usr/bin/env python3

import io
import json
import logging
import numpy as np
import os
import re
import struct
import tarfile
import hashlib
from multiprocessing import Pool

from dfatool.utils import NpEncoder, running_mean, soft_cast_int

from .energytrace import (
    EnergyTrace,
    EnergyTraceWithBarcode,
    EnergyTraceWithLogicAnalyzer,
    EnergyTraceWithTimer,
)
from .keysight import DLog, KeysightCSV
from .mimosa import MIMOSA

logger = logging.getLogger(__name__)


def _preprocess_mimosa(measurement):
    setup = measurement["setup"]
    mim = MIMOSA(
        float(setup["mimosa_voltage"]),
        int(setup["mimosa_shunt"]),
        with_traces=measurement["with_traces"],
    )
    try:
        charges, triggers = mim.load_data(measurement["content"])
        trigidx = mim.trigger_edges(triggers)
    except EOFError as e:
        mim.errors.append("MIMOSA logfile error: {}".format(e))
        trigidx = list()

    if len(trigidx) == 0:
        mim.errors.append("MIMOSA log has no triggers")
        return {
            "fileno": measurement["fileno"],
            "info": measurement["info"],
            "errors": mim.errors,
            "repeat_id": measurement["repeat_id"],
            "valid": False,
        }

    cal_edges = mim.calibration_edges(
        running_mean(mim.currents_nocal(charges[0 : trigidx[0]]), 10)
    )
    calfunc, caldata = mim.calibration_function(charges, cal_edges)
    vcalfunc = np.vectorize(calfunc, otypes=[np.float64])
    traces = mim.analyze_states(charges, trigidx, vcalfunc)

    # the last (v0) / first (v1) state is not part of the benchmark
    traces.pop(measurement["pop"])

    mim.validate(
        len(trigidx), traces, measurement["expected_trace"], setup["state_duration"]
    )

    processed_data = {
        "triggers": len(trigidx),
        "first_trig": trigidx[0] * 10,
        "calibration": caldata,
        "energy_trace": traces,
        "errors": mim.errors,
        "valid": len(mim.errors) == 0,
    }

    for key in ["fileno", "info", "repeat_id"]:
        processed_data[key] = measurement[key]

    return processed_data


def _preprocess_etlog(measurement):
    setup = measurement["setup"]

    energytrace_class = EnergyTraceWithBarcode
    if measurement["sync_mode"] == "la":
        energytrace_class = EnergyTraceWithLogicAnalyzer
    elif measurement["sync_mode"] == "timer":
        energytrace_class = EnergyTraceWithTimer

    etlog = energytrace_class(
        float(setup["voltage"]),
        int(setup["state_duration"]),
        measurement["transition_names"],
        with_traces=measurement["with_traces"],
    )
    states_and_transitions = list()
    try:
        etlog.load_data(measurement["content"])
        states_and_transitions = etlog.analyze_states(
            measurement["expected_trace"], measurement["repeat_id"]
        )
    except EOFError as e:
        etlog.errors.append("EnergyTrace logfile error: {}".format(e))
    except RuntimeError as e:
        etlog.errors.append("EnergyTrace loader error: {}".format(e))

    processed_data = {
        "fileno": measurement["fileno"],
        "repeat_id": measurement["repeat_id"],
        "info": measurement["info"],
        "energy_trace": states_and_transitions,
        "valid": len(etlog.errors) == 0,
        "errors": etlog.errors,
    }

    return processed_data


def _preprocess_dlog(measurement):
    setup = measurement["setup"]
    dlog = DLog(
        float(setup["voltage"]),
        int(setup["state_duration"]),
        with_traces=measurement["with_traces"],
    )

    states_and_transitions = list()
    try:
        dlog.load_data(measurement["content"])
        states_and_transitions = dlog.analyze_states(
            measurement["expected_trace"], measurement["repeat_id"]
        )
    except EOFError as e:
        dlog.errors.append("DLog file error: {}".format(e))
    except RuntimeError as e:
        dlog.errors.append("DLog loader error: {}".format(e))

    processed_data = {
        "fileno": measurement["fileno"],
        "repeat_id": measurement["repeat_id"],
        "info": measurement["info"],
        "energy_trace": states_and_transitions,
        "valid": len(dlog.errors) == 0,
        "errors": dlog.errors,
    }

    return processed_data


class TimingData:
    """
    Loader for timing model traces measured with on-board timers using `harness.OnboardTimerHarness`.

    Excpets a specific trace format and UART log output (as produced by
    generate-dfa-benchmark.py). Prunes states from output. (TODO)
    """

    def __init__(self, filenames):
        """
        Create a new TimingData object.

        Each filenames element corresponds to a measurement run.
        """
        self.filenames = filenames.copy()
        # holds the benchmark plan (dfa traces) for each series of benchmark runs.
        # Note that a single entry typically has more than one corresponding mimosa/energytrace benchmark files,
        # as benchmarks are run repeatedly to distinguish between random and parameter-dependent measurement effects.
        self.traces_by_fileno = []
        self.setup_by_fileno = []
        self.preprocessed = False
        self.version = 0

    def _concatenate_analyzed_traces(self):
        self.traces = []
        for trace_group in self.traces_by_fileno:
            for trace in trace_group:
                # TimingHarness logs states, but does not aggregate any data for them at the moment -> throw all states away
                transitions = list(
                    filter(lambda x: x["isa"] == "transition", trace["trace"])
                )
                self.traces.append({"id": trace["id"], "trace": transitions})
        for i, trace in enumerate(self.traces):
            trace["orig_id"] = trace["id"]
            trace["id"] = i
            for log_entry in trace["trace"]:
                paramkeys = sorted(log_entry["parameter"].keys())
                if "param" not in log_entry["offline_aggregates"]:
                    log_entry["offline_aggregates"]["param"] = list()
                if "duration" in log_entry["offline_aggregates"]:
                    for i in range(len(log_entry["offline_aggregates"]["duration"])):
                        paramvalues = list()
                        for paramkey in paramkeys:
                            if type(log_entry["parameter"][paramkey]) is list:
                                paramvalues.append(
                                    soft_cast_int(log_entry["parameter"][paramkey][i])
                                )
                            else:
                                paramvalues.append(
                                    soft_cast_int(log_entry["parameter"][paramkey])
                                )
                        if "args" in log_entry:
                            paramvalues.extend(map(soft_cast_int, log_entry["args"]))
                        log_entry["offline_aggregates"]["param"].append(paramvalues)

    def _preprocess_0(self):
        for filename in self.filenames:
            with open(filename, "r") as f:
                log_data = json.load(f)
                self.traces_by_fileno.extend(log_data["traces"])
        self._concatenate_analyzed_traces()

    def get_preprocessed_data(self):
        """
        Return a list of DFA traces annotated with timing and parameter data.

        Suitable for the PTAModel constructor.
        See PTAModel(...) docstring for format details.
        """
        if self.preprocessed:
            return self.traces
        if self.version == 0:
            self._preprocess_0()
        self.preprocessed = True
        return self.traces


def sanity_check_aggregate(aggregate):
    for key in aggregate:
        if "param" not in aggregate[key]:
            raise RuntimeError("aggregate[{}][param] does not exist".format(key))
        if "attributes" not in aggregate[key]:
            raise RuntimeError("aggregate[{}][attributes] does not exist".format(key))
        for attribute in aggregate[key]["attributes"]:
            if attribute not in aggregate[key]:
                raise RuntimeError(
                    "aggregate[{}][{}] does not exist, even though it is contained in aggregate[{}][attributes]".format(
                        key, attribute, key
                    )
                )
            param_len = len(aggregate[key]["param"])
            attr_len = len(aggregate[key][attribute])
            if param_len != attr_len:
                raise RuntimeError(
                    "parameter mismatch: len(aggregate[{}][param]) == {} != len(aggregate[{}][{}]) == {}".format(
                        key, param_len, key, attribute, attr_len
                    )
                )


def assert_legacy_compatibility(f1, t1, f2, t2):
    expected_param_names = sorted(
        t1["expected_trace"][0]["trace"][0]["parameter"].keys()
    )
    for run in t2["expected_trace"]:
        for state_or_trans in run["trace"]:
            actual_param_names = sorted(state_or_trans["parameter"].keys())
            if actual_param_names != expected_param_names:
                err = f"parameters in {f1} and {f2} are incompatible: {expected_param_names} ≠ {actual_param_names}"
                logger.error(err)
                raise ValueError(err)


def assert_ptalog_compatibility(f1, pl1, f2, pl2):
    param1 = pl1["pta"]["parameters"]
    param2 = pl2["pta"]["parameters"]
    if param1 != param2:
        err = f"parameters in {f1} and {f2} are incompatible: {param1} ≠ {param2}"
        logger.error(err)
        raise ValueError(err)

    states1 = list(sorted(pl1["pta"]["state"].keys()))
    states2 = list(sorted(pl2["pta"]["state"].keys()))
    if states1 != states2:
        err = f"states in {f1} and {f2} differ: {states1} ≠ {states2}"
        logger.warning(err)

    transitions1 = list(sorted(map(lambda t: t["name"], pl1["pta"]["transitions"])))
    transitions2 = list(sorted(map(lambda t: t["name"], pl1["pta"]["transitions"])))
    if transitions1 != transitions2:
        err = f"transitions in {f1} and {f2} differ: {transitions1} ≠ {transitions2}"
        logger.warning(err)


class RawData:
    """
    Loader for hardware model traces measured with MIMOSA.

    Expects a specific trace format and UART log output (as produced by the
    dfatool benchmark generator). Loads data, prunes bogus measurements, and
    provides preprocessed data suitable for PTAModel. Results are cached on the
    file system, making subsequent loads near-instant.
    """

    def __init__(self, filenames, with_traces=False, skip_cache=False):
        """
        Create a new RawData object.

        Each filename element corresponds to a measurement run.
        It must be a tar archive with the following contents:

        Version 0:

        * `setup.json`: measurement setup. Must contain the keys `state_duration` (how long each state is active, in ms),
          `mimosa_voltage` (voltage applied to dut, in V), and `mimosa_shunt` (shunt value, in Ohm)
        * `src/apps/DriverEval/DriverLog.json`: PTA traces and parameters for this benchmark.
          Layout: List of traces, each trace has an 'id' (numeric, starting with 1) and 'trace' (list of states and transitions) element.
          Each trace has an even number of elements, starting with the first state (usually `UNINITIALIZED`) and ending with a transition.
          Each state/transition must have the members `.parameter` (parameter values, empty string or None if unknown), `.isa` ("state" or "transition") and `.name`.
          Each transition must additionally contain `.plan.level` ("user" or "epilogue").
          Example: `[ {"id": 1, "trace": [ {"parameter": {...}, "isa": "state", "name": "UNINITIALIZED"}, ...] }, ... ]
        * At least one `*.mim` file. Each file corresponds to a single execution of the entire benchmark (i.e., all runs described in DriverLog.json) and starts with a MIMOSA Autocal calibration sequence.
          MIMOSA files are parsed by the `MIMOSA` class.

        Version 1:

        * `ptalog.json`: measurement setup and traces. Contents:
          `.opt.sleep`: state duration
          `.opt.pta`: PTA
          `.opt.traces`: list of sub-benchmark traces (the benchmark may have been split due to code size limitations). Each item is a list of traces as returned by `harness.traces`:
            `.opt.traces[]`: List of traces. Each trace has an 'id' (numeric, starting with 1) and 'trace' (list of states and transitions) element.
              Each state/transition must have the members '`parameter` (dict with normalized parameter values), `.isa` ("state" or "transition") and `.name`
              Each transition must additionally contain `.args`
          `.opt.files`: list of coresponding MIMOSA measurements.
            `.opt.files[]` = ['abc123.mim', ...]
          `.opt.configs`: ....
        * MIMOSA log files (`*.mim`) as specified in `.opt.files`

        Version 2:

        * `ptalog.json`: measurement setup and traces. Contents:
          `.opt.sleep`: state duration
          `.opt.pta`: PTA
          `.opt.traces`: list of sub-benchmark traces (the benchmark may have been split due to code size limitations). Each item is a list of traces as returned by `harness.traces`:
            `.opt.traces[]`: List of traces. Each trace has an 'id' (numeric, starting with 1) and 'trace' (list of states and transitions) element.
              Each state/transition must have the members '`parameter` (dict with normalized parameter values), `.isa` ("state" or "transition") and `.name`
              Each transition must additionally contain `.args` and `.duration`
              * `.duration`: list of durations, one per repetition
          `.opt.files`: list of coresponding EnergyTrace measurements.
            `.opt.files[]` = ['abc123.etlog', ...]
          `.opt.configs`: ....
        * EnergyTrace log files (`*.etlog`) as specified in `.opt.files`

        If a cached result for a file is available, it is loaded and the file
        is not preprocessed, unless `with_traces` is set.

        tbd
        """
        self.with_traces = with_traces
        self.input_filenames = filenames.copy()
        self.filenames = list()
        self.traces_by_fileno = list()
        self.setup_by_fileno = list()
        self.version = 0
        self.preprocessed = False
        self._parameter_names = None
        self.ignore_clipping = False
        self.pta = None
        self.ptalog = None

        with tarfile.open(filenames[0]) as tf:
            for member in tf.getmembers():
                if member.name == "ptalog.json" and self.version == 0:
                    self.version = 1
                    # or greater, if *.etlog / *.dlog exist
                elif ".etlog" in member.name:
                    self.version = 2
                    break
                elif ".dlog" in member.name:
                    self.version = 3
                    break
            if self.version >= 1:
                self.ptalog = json.load(tf.extractfile(tf.getmember("ptalog.json")))
                self.pta = self.ptalog["pta"]

        if self.ptalog and len(filenames) > 1:
            for filename in filenames[1:]:
                with tarfile.open(filename) as tf:
                    new_ptalog = json.load(tf.extractfile(tf.getmember("ptalog.json")))
                    assert_ptalog_compatibility(
                        filenames[0], self.ptalog, filename, new_ptalog
                    )
                    self.ptalog["files"].extend(new_ptalog["files"])

        self.set_cache_file()
        if not with_traces and not skip_cache:
            self.load_cache()

    def set_cache_file(self):
        cache_key = hashlib.sha256("!".join(self.input_filenames).encode()).hexdigest()
        self.cache_dir = os.path.dirname(self.input_filenames[0]) + "/cache"
        self.cache_file = "{}/{}.json".format(self.cache_dir, cache_key)

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                try:
                    cache_data = json.load(f)
                    self.filenames = cache_data["filenames"]
                    self.traces = cache_data["traces"]
                    self.preprocessing_stats = cache_data["preprocessing_stats"]
                    if "pta" in cache_data:
                        self.pta = cache_data["pta"]
                    if "ptalog" in cache_data:
                        self.ptalog = cache_data["ptalog"]
                    self.setup_by_fileno = cache_data["setup_by_fileno"]
                    self.preprocessed = True
                except json.decoder.JSONDecodeError as e:
                    logger.info(f"Skipping cache entry {self.cache_file}: {e}")

    def save_cache(self):
        if self.with_traces:
            return
        try:
            os.mkdir(self.cache_dir)
        except FileExistsError:
            pass
        except PermissionError:
            logger.info(
                f"Cannot write cache entry {self.cache_file}: permission denied"
            )
            # no cache for you
            return
        with open(self.cache_file, "w") as f:
            cache_data = {
                "filenames": self.filenames,
                "traces": self.traces,
                "preprocessing_stats": self.preprocessing_stats,
                "pta": self.pta,
                "ptalog": self.ptalog,
                "setup_by_fileno": self.setup_by_fileno,
            }
            json.dump(cache_data, f, cls=NpEncoder)

    def to_dref(self) -> dict:
        return {
            "raw measurements/valid": self.preprocessing_stats["num_valid"],
            "raw measurements/total": self.preprocessing_stats["num_runs"],
            "static state duration/mean": (
                np.mean(list(map(lambda x: x["state_duration"], self.setup_by_fileno))),
                r"\milli\second",
            ),
        }

    def _concatenate_traces(self, list_of_traces):
        """
        Concatenate `list_of_traces` (list of lists) into a single trace while adjusting trace IDs.

        :param list_of_traces: List of list of traces.
        :returns: List of traces with ['id'] in ascending order and ['orig_id'] as previous ['id']
        """

        trace_output = list()
        for trace in list_of_traces:
            trace_output.extend(trace.copy())
        for i, trace in enumerate(trace_output):
            trace["orig_id"] = trace["id"]
            trace["id"] = i
        return trace_output

    def get_preprocessed_data(self):
        """
        Return a list of DFA traces annotated with energy, timing, and parameter data.
        The list is cached on disk, unless the constructor was called with `with_traces` set.

        Each DFA trace contains the following elements:
         * `id`: Numeric ID, starting with 1
         * `total_energy`: Total amount of energy (as measured by MIMOSA) in the entire trace
         * `orig_id`: Original trace ID. May differ when concatenating multiple (different) benchmarks into one analysis, i.e., when calling RawData() with more than one file argument.
         * `trace`: List of the individual states and transitions in this trace. Always contains an even number of elements, staring with the first state (typically "UNINITIALIZED") and ending with a transition.

        Each trace element (that is, an entry of the `trace` list mentioned above) contains the following elements:
         * `isa`: "state" or "transition"
         * `name`: name
         * `offline`: List of offline measumerents for this state/transition. Each entry contains a result for this state/transition during one benchmark execution.
           Entry contents:
            - `clip_rate`: rate of clipped energy measurements, 0 .. 1
            - `raw_mean`: mean raw MIMOSA value
            - `raw_std`: standard deviation of raw MIMOSA value
            - `uW_mean`: mean power draw, uW
            - `uw_std`: standard deviation of power draw, uW
            - `us`: state/transition duration, us
            - `uW_mean_delta_prev`: (only for transitions) difference between uW_mean of this transition and uW_mean of previous state
            - `uW_mean_elta_next`: (only for transitions) difference between uW_mean of this transition and uW_mean of next state
            - `timeout`: (only for transitions) duration of previous state, us
         * `offline_aggregates`: Aggregate of `offline` entries. dict of lists, each list entry has the same length
            - `duration`: state/transition durations ("us"), us
            - `energy`: state/transition energy ("us * uW_mean"), us
            - `power`: mean power draw ("uW_mean"), uW
            - `power_std`: standard deviations of power draw ("uW_std"), uW^2
            - `paramkeys`: List of lists, each sub-list contains the parameter names corresponding to the `param` entries
            - `param`: List of lists, each sub-list contains the parameter values for this measurement. Typically, all sub-lists are the same.
            - `rel_energy_prev`: (only for transitions) transition energy relative to previous state mean power, pJ
            - `rel_energy_next`: (only for transitions) transition energy relative to next state mean power, pJ
            - `rel_power_prev`: (only for transitions) powerrelative to previous state mean power, µW
            - `rel_power_next`: (only for transitions) power relative to next state mean power, µW
            - `timeout`: (only for transitions) duration of previous state, us
         * `offline_attributes`: List containing the keys of `offline_aggregates` which are meant to be part of the model.
           This list ultimately decides which hardware/software attributes the model describes.
           If isa == state, it contains power, duration, energy
           If isa == transition, it contains power, rel_power_prev, rel_power_next, duration, timeout
         * `online`: List of online estimations for this state/transition. Each entry contains a result for this state/transition during one benchmark execution.
          Entry contents for isa == state:
            - `time`: state/transition
          Entry contents for isa == transition:
            - `timeout`: Duration of previous state, measured using on-board timers
         * `parameter`: dictionary describing parameter values for this state/transition. Parameter values refer to the begin of the state/transition and do not account for changes made by the transition.
         * `plan`: Dictionary describing expected behaviour according to schedule / offline model.
           Contents for isa == state: `energy`, `power`, `time`
           Contents for isa == transition: `energy`, `timeout`, `level`.
           If level is "user", the transition is part of the regular driver API. If level is "epilogue", it is an interrupt service routine and not called explicitly.
        Each transition also contains:
         * `args`: List of arguments the corresponding function call was called with. args entries are strings which are not necessarily numeric
         * `code`: List of function name (first entry) and arguments (remaining entries) of the corresponding function call
        """
        if self.preprocessed:
            return self.traces
        if self.version <= 3:
            self._preprocess(self.version)
        else:
            raise ValueError(f"Unsupported raw data version: {self.version}")
        self.preprocessed = True
        self.save_cache()
        return self.traces

    def _preprocess(self, version):
        """Load raw MIMOSA data and turn it into measurements which are ready to be analyzed."""
        offline_data = []
        for i, filename in enumerate(self.input_filenames):

            if version == 0:

                self.filenames = self.input_filenames
                with tarfile.open(filename) as tf:
                    self.setup_by_fileno.append(json.load(tf.extractfile("setup.json")))
                    traces = json.load(
                        tf.extractfile("src/apps/DriverEval/DriverLog.json")
                    )
                    self.traces_by_fileno.append(traces)
                    for member in tf.getmembers():
                        _, extension = os.path.splitext(member.name)
                        if extension == ".mim":
                            offline_data.append(
                                {
                                    "content": tf.extractfile(member).read(),
                                    # only for validation
                                    "expected_trace": traces,
                                    "fileno": i,
                                    # For debug output and warnings
                                    "info": member,
                                    # Strip the last state (it is not part of the scheduled measurement)
                                    "pop": -1,
                                    "repeat_id": 0,  # needed to add runtime "return_value.apply_from" parameters to offline_aggregates. Irrelevant in v0.
                                    "setup": self.setup_by_fileno[i],
                                    "with_traces": self.with_traces,
                                }
                            )

            elif version == 1:

                with tarfile.open(filename) as tf:
                    ptalog = json.load(tf.extractfile(tf.getmember("ptalog.json")))

                    # Benchmark code may be too large to be executed in a single
                    # run, so benchmarks (a benchmark is basically a list of DFA runs)
                    # may be split up. To accomodate this, ptalog['traces'] is
                    # a list of lists: ptalog['traces'][0] corresponds to the
                    # first benchmark part, ptalog['traces'][1] to the
                    # second, and so on. ptalog['traces'][0][0] is the first
                    # trace (a sequence of states and transitions) in the
                    # first benchmark part, ptalog['traces'][0][1] the second, etc.
                    #
                    # As traces are typically repeated to minimize the effect
                    # of random noise, observations for each benchmark part
                    # are also lists. In this case, this applies in two
                    # cases: traces[i][j]['parameter'][some_param] is either
                    # a value (if the parameter is controlld by software)
                    # or a list (if the parameter is known a posteriori, e.g.
                    # "how many retransmissions did this packet take?").
                    #
                    # The second case is the MIMOSA energy measurements, which
                    # are listed in ptalog['files']. ptalog['files'][0]
                    # contains a list of files for the first benchmark part,
                    # ptalog['files'][0][0] is its first iteration/repetition,
                    # ptalog['files'][0][1] the second, etc.

                    for j, traces in enumerate(ptalog["traces"]):
                        self.filenames.append("{}#{}".format(filename, j))
                        self.traces_by_fileno.append(traces)
                        self.setup_by_fileno.append(
                            {
                                "mimosa_voltage": ptalog["configs"][j]["voltage"],
                                "mimosa_shunt": ptalog["configs"][j]["shunt"],
                                "state_duration": ptalog["opt"]["sleep"],
                            }
                        )
                        for repeat_id, mim_file in enumerate(ptalog["files"][j]):
                            # MIMOSA benchmarks always use a single .mim file per benchmark run.
                            # However, depending on the dfatool version used to run the
                            # benchmark, ptalog["files"][j] is either "foo.mim" (before Oct 2020)
                            # or ["foo.mim"] (from Oct 2020 onwards).
                            if type(mim_file) is list:
                                mim_file = mim_file[0]
                            member = tf.getmember(mim_file)
                            offline_data.append(
                                {
                                    "content": tf.extractfile(member).read(),
                                    # only for validation
                                    "expected_trace": traces,
                                    "fileno": len(self.traces_by_fileno) - 1,
                                    # For debug output and warnings
                                    "info": member,
                                    # The first online measurement is the UNINITIALIZED state. In v1,
                                    # it is not part of the expected PTA trace -> remove it.
                                    "pop": 0,
                                    "setup": self.setup_by_fileno[-1],
                                    "repeat_id": repeat_id,  # needed to add runtime "return_value.apply_from" parameters to offline_aggregates.
                                    "with_traces": self.with_traces,
                                }
                            )

            elif version == 2:

                with tarfile.open(filename) as tf:
                    ptalog = json.load(tf.extractfile(tf.getmember("ptalog.json")))
                    if "sync" in ptalog["opt"]["energytrace"]:
                        sync_mode = ptalog["opt"]["energytrace"]["sync"]
                    else:
                        sync_mode = "bar"

                    # Benchmark code may be too large to be executed in a single
                    # run, so benchmarks (a benchmark is basically a list of DFA runs)
                    # may be split up. To accomodate this, ptalog['traces'] is
                    # a list of lists: ptalog['traces'][0] corresponds to the
                    # first benchmark part, ptalog['traces'][1] to the
                    # second, and so on. ptalog['traces'][0][0] is the first
                    # trace (a sequence of states and transitions) in the
                    # first benchmark part, ptalog['traces'][0][1] the second, etc.
                    #
                    # As traces are typically repeated to minimize the effect
                    # of random noise, observations for each benchmark part
                    # are also lists. In this case, this applies in two
                    # cases: traces[i][j]['parameter'][some_param] is either
                    # a value (if the parameter is controlld by software)
                    # or a list (if the parameter is known a posteriori, e.g.
                    # "how many retransmissions did this packet take?").
                    #
                    # The second case is the MIMOSA energy measurements, which
                    # are listed in ptalog['files']. ptalog['files'][0]
                    # contains a list of files for the first benchmark part,
                    # ptalog['files'][0][0] is its first iteration/repetition,
                    # ptalog['files'][0][1] the second, etc.

                    # generate-dfa-benchmark uses TimingHarness to obtain timing data.
                    # Data is placed in 'offline_aggregates', which is also
                    # where we are going to store power/energy data.
                    # In case of invalid measurements, this can lead to a
                    # mismatch between duration and power/energy data, e.g.
                    # where duration = [A, B, C], power = [a, b], B belonging
                    # to an invalid measurement and thus power[b] corresponding
                    # to duration[C]. At the moment, this is harmless, but in the
                    # future it might not be.
                    if "offline_aggregates" in ptalog["traces"][0][0]["trace"][0]:
                        for trace_group in ptalog["traces"]:
                            for trace in trace_group:
                                for state_or_transition in trace["trace"]:
                                    offline_aggregates = state_or_transition.pop(
                                        "offline_aggregates", None
                                    )
                                    if offline_aggregates:
                                        state_or_transition[
                                            "online_aggregates"
                                        ] = offline_aggregates

                    for j, traces in enumerate(ptalog["traces"]):
                        self.filenames.append("{}#{}".format(filename, j))
                        self.traces_by_fileno.append(traces)
                        self.setup_by_fileno.append(
                            {
                                "voltage": ptalog["configs"][j]["voltage"],
                                "state_duration": ptalog["opt"]["sleep"],
                            }
                        )
                        for repeat_id, etlog_files in enumerate(ptalog["files"][j]):
                            # legacy measurements supported only one file per run
                            if type(etlog_files) is not list:
                                etlog_files = [etlog_files]
                            members = list(map(tf.getmember, etlog_files))
                            offline_data.append(
                                {
                                    "content": list(
                                        map(lambda f: tf.extractfile(f).read(), members)
                                    ),
                                    # used to determine EnergyTrace class for analysis
                                    "sync_mode": sync_mode,
                                    "fileno": len(self.traces_by_fileno) - 1,
                                    # For debug output and warnings
                                    "info": members[0],
                                    "setup": self.setup_by_fileno[-1],
                                    # needed to add runtime "return_value.apply_from" parameters to offline_aggregates, also for EnergyTraceWithBarcode
                                    "repeat_id": repeat_id,
                                    # only for validation
                                    "expected_trace": traces,
                                    "with_traces": self.with_traces,
                                    # only for EnergyTraceWithBarcode
                                    "transition_names": list(
                                        map(
                                            lambda x: x["name"],
                                            ptalog["pta"]["transitions"],
                                        )
                                    ),
                                }
                            )
                # TODO remove 'offline_aggregates' from pre-parse data and place
                # it under 'online_aggregates' or similar instead. This way, if
                # a .etlog file fails to parse, its corresponding duration data
                # will not linger in 'offline_aggregates' and confuse the hell
                # out of other code paths

            elif self.version == 3:
                with tarfile.open(filename) as tf:
                    ptalog = json.load(tf.extractfile(tf.getmember("ptalog.json")))
                    # generate-dfa-benchmark uses TimingHarness to obtain timing data.
                    # Data is placed in 'offline_aggregates', which is also
                    # where we are going to store power/energy data.
                    # In case of invalid measurements, this can lead to a
                    # mismatch between duration and power/energy data, e.g.
                    # where duration = [A, B, C], power = [a, b], B belonging
                    # to an invalid measurement and thus power[b] corresponding
                    # to duration[C]. At the moment, this is harmless, but in the
                    # future it might not be.
                    if "offline_aggregates" in ptalog["traces"][0][0]["trace"][0]:
                        for trace_group in ptalog["traces"]:
                            for trace in trace_group:
                                for state_or_transition in trace["trace"]:
                                    offline_aggregates = state_or_transition.pop(
                                        "offline_aggregates", None
                                    )
                                    if offline_aggregates:
                                        state_or_transition[
                                            "online_aggregates"
                                        ] = offline_aggregates
                    for j, traces in enumerate(ptalog["traces"]):
                        self.filenames.append("{}#{}".format(filename, j))
                        self.traces_by_fileno.append(traces)
                        self.setup_by_fileno.append(
                            {
                                "voltage": ptalog["configs"][j]["voltage"],
                                "state_duration": ptalog["opt"]["sleep"],
                            }
                        )
                        for repeat_id, dlog_file in enumerate(ptalog["files"][j]):
                            member = tf.getmember(dlog_file)
                            offline_data.append(
                                {
                                    "content": tf.extractfile(member).read(),
                                    "fileno": len(self.traces_by_fileno) - 1,
                                    # For debug output and warnings
                                    "info": member,
                                    "setup": self.setup_by_fileno[-1],
                                    # needed to add runtime "return_value.apply_from" parameters to offline_aggregates
                                    "repeat_id": repeat_id,
                                    # only for validation
                                    "expected_trace": traces,
                                    "with_traces": self.with_traces,
                                }
                            )

        if self.version == 0 and len(self.input_filenames) > 1:
            for entry in offline_data:
                assert_legacy_compatibility(
                    self.input_filenames[0],
                    offline_data[0],
                    self.input_filenames[entry["fileno"]],
                    entry,
                )

        with Pool() as pool:
            if self.version <= 1:
                measurements = pool.map(_preprocess_mimosa, offline_data)
            elif self.version == 2:
                measurements = pool.map(_preprocess_etlog, offline_data)
            elif self.version == 3:
                measurements = pool.map(_preprocess_dlog, offline_data)

        num_valid = 0
        for measurement in measurements:

            if "energy_trace" not in measurement:
                logger.warning(
                    "Skipping {ar:s}/{m:s}: {e:s}".format(
                        ar=self.filenames[measurement["fileno"]],
                        m=measurement["info"].name,
                        e="; ".join(measurement["errors"]),
                    )
                )
                continue

            if version == 0 or version == 1:
                if measurement["valid"]:
                    MIMOSA.add_offline_aggregates(
                        self.traces_by_fileno[measurement["fileno"]],
                        measurement["energy_trace"],
                        measurement["repeat_id"],
                    )
                    num_valid += 1
                else:
                    logger.warning(
                        "Skipping {ar:s}/{m:s}: {e:s}".format(
                            ar=self.filenames[measurement["fileno"]],
                            m=measurement["info"].name,
                            e="; ".join(measurement["errors"]),
                        )
                    )
            elif version == 2 or version == 3:
                if measurement["valid"]:
                    try:
                        EnergyTrace.add_offline_aggregates(
                            self.traces_by_fileno[measurement["fileno"]],
                            measurement["energy_trace"],
                            measurement["repeat_id"],
                        )
                        num_valid += 1
                    except Exception as e:
                        logger.warning(
                            f"Skipping #{measurement['fileno']} {measurement['info']}:\n{e}"
                        )
                else:
                    logger.warning(
                        "Skipping {ar:s}/{m:s}: {e:s}".format(
                            ar=self.filenames[measurement["fileno"]],
                            m=measurement["info"].name,
                            e="; ".join(measurement["errors"]),
                        )
                    )
        logger.info(
            "{num_valid:d}/{num_total:d} measurements are valid".format(
                num_valid=num_valid, num_total=len(measurements)
            )
        )
        self.traces = self._concatenate_traces(self.traces_by_fileno)
        self.preprocessing_stats = {
            "num_runs": len(measurements),
            "num_valid": num_valid,
        }


def _add_trace_data_to_aggregate(aggregate, key, element):
    # Only cares about element['isa'], element['offline_aggregates'], and
    # element['plan']['level']
    if key not in aggregate:
        aggregate[key] = {"isa": element["isa"]}
        for datakey in element["offline_aggregates"].keys():
            aggregate[key][datakey] = []
        if element["isa"] == "state":
            aggregate[key]["attributes"] = ["power"]
        else:
            # TODO do not hardcode values
            aggregate[key]["attributes"] = [
                "duration",
                "power",
                "rel_power_prev",
                "rel_power_next",
                "energy",
                "rel_energy_prev",
                "rel_energy_next",
            ]
            if "plan" in element and element["plan"]["level"] == "epilogue":
                aggregate[key]["attributes"].insert(0, "timeout")
        attributes = aggregate[key]["attributes"].copy()
        for attribute in attributes:
            if attribute not in element["offline_aggregates"]:
                aggregate[key]["attributes"].remove(attribute)
        if "offline_support" in element:
            aggregate[key]["supports"] = element["offline_support"]
        else:
            aggregate[key]["supports"] = list()
    for datakey, dataval in element["offline_aggregates"].items():
        aggregate[key][datakey].extend(dataval)


def pta_trace_to_aggregate(traces, ignore_trace_indexes=[]):
    """
    Convert preprocessed DFA traces from peripherals/drivers to by_name aggregate for PTAModel.

    arguments:
    traces -- [ ... Liste von einzelnen Läufen (d.h. eine Zustands- und Transitionsfolge UNINITIALIZED -> foo -> FOO -> bar -> BAR -> ...)
        Jeder Lauf:
        - id: int Nummer des Laufs, beginnend bei 1
        - trace: [ ... Liste von Zuständen und Transitionen
            Jeweils:
            - name: str Name
            - isa: str state // transition
            - parameter: { ... globaler Parameter: aktueller wert. null falls noch nicht eingestellt }
            - args: [ Funktionsargumente, falls isa == 'transition' ]
            - offline_aggregates:
                - power: [float(uW)] Mittlere Leistung während Zustand/Transitions
                - power_std: [float(uW^2)] Standardabweichung der Leistung
                - duration: [int(us)] Dauer
                - energy: [float(pJ)] Energieaufnahme des Zustands / der Transition
                - clip_rate: [float(0..1)] Clipping
                - paramkeys: [[str]] Name der berücksichtigten Parameter
                - param: [int // str] Parameterwerte. Quasi-Duplikat von 'parameter' oben
                Falls isa == 'transition':
                - timeout: [int(us)] Dauer des vorherigen Zustands
                - rel_energy_prev: [int(pJ)]
                - rel_energy_next: [int(pJ)]
                - rel_power_prev: [int(µW)]
                - rel_power_next: [int(µW)]
        ]
    ]
    ignore_trace_indexes -- list of trace indexes. The corresponding taces will be ignored.

    returns a tuple of three elements:
    by_name -- measurements aggregated by state/transition name, annotated with parameter values
    parameter_names -- list of parameter names
    arg_count -- dict mapping transition names to the number of arguments of their corresponding driver function

    by_name layout:
    Dictionary with one key per state/transition ('send', 'TX', ...).
    Each element is in turn a dict with the following elements:
    - isa: 'state' or 'transition'
    - power: list of mean power measurements in µW
    - duration: list of durations in µs
    - power_std: list of stddev of power per state/transition
    - energy: consumed energy (power*duration) in pJ
    - paramkeys: list of parameter names in each measurement (-> list of lists)
    - param: list of parameter values in each measurement (-> list of lists)
    - attributes: list of keys that should be analyzed,
        e.g. ['power', 'duration']
    additionally, only if isa == 'transition':
    - timeout: list of duration of previous state in µs
    - rel_energy_prev: transition energy relative to previous state mean power in pJ
    - rel_energy_next: transition energy relative to next state mean power in pJ
    """
    arg_count = dict()
    by_name = dict()
    parameter_names = sorted(traces[0]["trace"][0]["parameter"].keys())
    for run in traces:
        if run["id"] not in ignore_trace_indexes:
            for elem in run["trace"]:
                if (
                    elem["isa"] == "transition"
                    and not elem["name"] in arg_count
                    and "args" in elem
                ):
                    arg_count[elem["name"]] = len(elem["args"])
                if elem["name"] != "UNINITIALIZED":
                    _add_trace_data_to_aggregate(by_name, elem["name"], elem)
    for elem in by_name.values():
        for key in elem["attributes"]:
            elem[key] = np.array(elem[key])
    return by_name, parameter_names, arg_count
