#!/usr/bin/env python3

import csv
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

from .utils import NpEncoder, running_mean, soft_cast_int

logger = logging.getLogger(__name__)

try:
    from .pubcode import Code128
    import zbar

    zbar_available = True
except ImportError:
    zbar_available = False


arg_support_enabled = True


class KeysightCSV:
    """Simple loader for Keysight CSV data, as exported by the windows software."""

    def __init__(self):
        """Create a new KeysightCSV object."""
        pass

    def load_data(self, filename: str):
        """
        Load log data from filename, return timestamps and currents.

        Returns two one-dimensional NumPy arrays: timestamps and corresponding currents.
        """
        with open(filename) as f:
            for i, _ in enumerate(f):
                pass
            timestamps = np.ndarray((i - 3), dtype=float)
            currents = np.ndarray((i - 3), dtype=float)
        # basically seek back to start
        with open(filename) as f:
            for _ in range(4):
                next(f)
            reader = csv.reader(f, delimiter=",")
            for i, row in enumerate(reader):
                timestamps[i] = float(row[0])
                currents[i] = float(row[2]) * -1
        return timestamps, currents


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
            "has_datasource_error": len(mim.errors) > 0,
            "datasource_errors": mim.errors,
            "expected_trace": measurement["expected_trace"],
            "repeat_id": measurement["repeat_id"],
        }

    cal_edges = mim.calibration_edges(
        running_mean(mim.currents_nocal(charges[0 : trigidx[0]]), 10)
    )
    calfunc, caldata = mim.calibration_function(charges, cal_edges)
    vcalfunc = np.vectorize(calfunc, otypes=[np.float64])

    processed_data = {
        "fileno": measurement["fileno"],
        "info": measurement["info"],
        "triggers": len(trigidx),
        "first_trig": trigidx[0] * 10,
        "calibration": caldata,
        "energy_trace": mim.analyze_states(charges, trigidx, vcalfunc),
        "has_datasource_error": len(mim.errors) > 0,
        "datasource_errors": mim.errors,
    }

    for key in ["expected_trace", "repeat_id"]:
        if key in measurement:
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
        "expected_trace": measurement["expected_trace"],
        "energy_trace": states_and_transitions,
        "has_datasource_error": len(etlog.errors) > 0,
        "datasource_errors": etlog.errors,
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
        self.traces_by_fileno = []
        self.setup_by_fileno = []
        self.preprocessed = False
        self._parameter_names = None
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
                        if arg_support_enabled and "args" in log_entry:
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
        self.filenames = filenames.copy()
        self.traces_by_fileno = []
        self.setup_by_fileno = []
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
                    # might also be version 2
                    # depends on whether *.etlog exists or not
                elif ".etlog" in member.name:
                    self.version = 2
                    break
            if self.version >= 1:
                self.ptalog = json.load(tf.extractfile(tf.getmember("ptalog.json")))
                self.pta = self.ptalog["pta"]

        self.set_cache_file()
        if not with_traces and not skip_cache:
            self.load_cache()

    def set_cache_file(self):
        cache_key = hashlib.sha256("!".join(self.filenames).encode()).hexdigest()
        self.cache_dir = os.path.dirname(self.filenames[0]) + "/cache"
        self.cache_file = "{}/{}.json".format(self.cache_dir, cache_key)

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
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

    def save_cache(self):
        if self.with_traces:
            return
        try:
            os.mkdir(self.cache_dir)
        except FileExistsError:
            pass
        with open(self.cache_file, "w") as f:
            cache_data = {
                "filenames": self.filenames,
                "traces": self.traces,
                "preprocessing_stats": self.preprocessing_stats,
                "pta": self.pta,
                "ptalog": self.ptalog,
                "setup_by_fileno": self.setup_by_fileno,
            }
            json.dump(cache_data, f)

    def _state_is_too_short(self, online, offline, state_duration, next_transition):
        # We cannot control when an interrupt causes a state to be left
        if next_transition["plan"]["level"] == "epilogue":
            return False

        # Note: state_duration is stored as ms, not us
        return offline["us"] < state_duration * 500

    def _state_is_too_long(self, online, offline, state_duration, prev_transition):
        # If the previous state was left by an interrupt, we may have some
        # waiting time left over. So it's okay if the current state is longer
        # than expected.
        if prev_transition["plan"]["level"] == "epilogue":
            return False
        # state_duration is stored as ms, not us
        return offline["us"] > state_duration * 1500

    def _measurement_is_valid_2(self, processed_data):
        """
        Check if a dfatool v2 measurement is valid.

        processed_data layout:
        'fileno' : measurement['fileno'],
        'info' : measurement['info'],
        'energy_trace' : etlog.analyze_states()
            A sequence of unnamed, unparameterized states and transitions with
            power and timing data
        'expected_trace' : trace from PTA DFS (with parameter data)
        etlog.analyze_states returns a list of (alternating) states and transitions.
        Each element is a dict containing:
            - isa: 'state' oder 'transition'
            - W_mean: Mittelwert der (kalibrierten) Leistungsaufnahme
            - W_std: Standardabweichung der (kalibrierten) Leistungsaufnahme
            - s: duration

            if isa == 'transition':
            - W_mean_delta_prev: Differenz zwischen W_mean und W_mean des vorherigen Zustands
            - W_mean_delta_next: Differenz zwischen W_mean und W_mean des Folgezustands
        """

        # Check for low-level parser errors
        if processed_data["has_datasource_error"]:
            processed_data["error"] = "; ".join(processed_data["datasource_errors"])
            return False

        # Note that the low-level parser (EnergyTraceWithBarcode) already checks
        # whether the transition count is correct

        return True

    def _measurement_is_valid_01(self, processed_data):
        """
        Check if a dfatool v0 or v1 measurement is valid.

        processed_data layout:
        'fileno' : measurement['fileno'],
        'info' : measurement['info'],
        'triggers' : len(trigidx),
        'first_trig' : trigidx[0] * 10,
        'calibration' : caldata,
        'energy_trace' : mim.analyze_states(charges, trigidx, vcalfunc)
            A sequence of unnamed, unparameterized states and transitions with
            power and timing data
        'expected_trace' : trace from PTA DFS (with parameter data)
        mim.analyze_states returns a list of (alternating) states and transitions.
        Each element is a dict containing:
            - isa: 'state' oder 'transition'
            - clip_rate: range(0..1) Anteil an Clipping im Energieverbrauch
            - raw_mean: Mittelwert der Rohwerte
            - raw_std: Standardabweichung der Rohwerte
            - uW_mean: Mittelwert der (kalibrierten) Leistungsaufnahme
            - uW_std: Standardabweichung der (kalibrierten) Leistungsaufnahme
            - us: Dauer

            Nur falls isa == 'transition':
            - timeout: Dauer des vorherigen Zustands
            - uW_mean_delta_prev: Differenz zwischen uW_mean und uW_mean des vorherigen Zustands
            - uW_mean_delta_next: Differenz zwischen uW_mean und uW_mean des Folgezustands
        """
        setup = self.setup_by_fileno[processed_data["fileno"]]
        if "expected_trace" in processed_data:
            traces = processed_data["expected_trace"]
        else:
            traces = self.traces_by_fileno[processed_data["fileno"]]
        state_duration = setup["state_duration"]

        # Check MIMOSA error
        if processed_data["has_datasource_error"]:
            processed_data["error"] = "; ".join(processed_data["datasource_errors"])
            return False

        # Check trigger count
        sched_trigger_count = 0
        for run in traces:
            sched_trigger_count += len(run["trace"])
        if sched_trigger_count != processed_data["triggers"]:
            processed_data[
                "error"
            ] = "got {got:d} trigger edges, expected {exp:d}".format(
                got=processed_data["triggers"], exp=sched_trigger_count
            )
            return False
        # Check state durations. Very short or long states can indicate a
        # missed trigger signal which wasn't detected due to duplicate
        # triggers elsewhere
        online_datapoints = []
        for run_idx, run in enumerate(traces):
            for trace_part_idx in range(len(run["trace"])):
                online_datapoints.append((run_idx, trace_part_idx))
        for offline_idx, online_ref in enumerate(online_datapoints):
            online_run_idx, online_trace_part_idx = online_ref
            offline_trace_part = processed_data["energy_trace"][offline_idx]
            online_trace_part = traces[online_run_idx]["trace"][online_trace_part_idx]

            if self._parameter_names is None:
                self._parameter_names = sorted(online_trace_part["parameter"].keys())

            if sorted(online_trace_part["parameter"].keys()) != self._parameter_names:
                processed_data[
                    "error"
                ] = "Offline #{off_idx:d} (online {on_name:s} @ {on_idx:d}/{on_sub:d}) has inconsistent parameter set: should be {param_want}, is {param_is}".format(
                    off_idx=offline_idx,
                    on_idx=online_run_idx,
                    on_sub=online_trace_part_idx,
                    on_name=online_trace_part["name"],
                    param_want=self._parameter_names,
                    param_is=sorted(online_trace_part["parameter"].keys()),
                )

            if online_trace_part["isa"] != offline_trace_part["isa"]:
                processed_data[
                    "error"
                ] = "Offline #{off_idx:d} (online {on_name:s} @ {on_idx:d}/{on_sub:d}) claims to be {off_isa:s}, but should be {on_isa:s}".format(
                    off_idx=offline_idx,
                    on_idx=online_run_idx,
                    on_sub=online_trace_part_idx,
                    on_name=online_trace_part["name"],
                    off_isa=offline_trace_part["isa"],
                    on_isa=online_trace_part["isa"],
                )
                return False

            # Clipping in UNINITIALIZED (offline_idx == 0) can happen during
            # calibration and is handled by MIMOSA
            if (
                offline_idx != 0
                and offline_trace_part["clip_rate"] != 0
                and not self.ignore_clipping
            ):
                processed_data[
                    "error"
                ] = "Offline #{off_idx:d} (online {on_name:s} @ {on_idx:d}/{on_sub:d}) was clipping {clip:f}% of the time".format(
                    off_idx=offline_idx,
                    on_idx=online_run_idx,
                    on_sub=online_trace_part_idx,
                    on_name=online_trace_part["name"],
                    clip=offline_trace_part["clip_rate"] * 100,
                )
                return False

            if (
                online_trace_part["isa"] == "state"
                and online_trace_part["name"] != "UNINITIALIZED"
                and len(traces[online_run_idx]["trace"]) > online_trace_part_idx + 1
            ):
                online_prev_transition = traces[online_run_idx]["trace"][
                    online_trace_part_idx - 1
                ]
                online_next_transition = traces[online_run_idx]["trace"][
                    online_trace_part_idx + 1
                ]
                try:
                    if self._state_is_too_short(
                        online_trace_part,
                        offline_trace_part,
                        state_duration,
                        online_next_transition,
                    ):
                        processed_data[
                            "error"
                        ] = "Offline #{off_idx:d} (online {on_name:s} @ {on_idx:d}/{on_sub:d}) is too short (duration = {dur:d} us)".format(
                            off_idx=offline_idx,
                            on_idx=online_run_idx,
                            on_sub=online_trace_part_idx,
                            on_name=online_trace_part["name"],
                            dur=offline_trace_part["us"],
                        )
                        return False
                    if self._state_is_too_long(
                        online_trace_part,
                        offline_trace_part,
                        state_duration,
                        online_prev_transition,
                    ):
                        processed_data[
                            "error"
                        ] = "Offline #{off_idx:d} (online {on_name:s} @ {on_idx:d}/{on_sub:d}) is too long (duration = {dur:d} us)".format(
                            off_idx=offline_idx,
                            on_idx=online_run_idx,
                            on_sub=online_trace_part_idx,
                            on_name=online_trace_part["name"],
                            dur=offline_trace_part["us"],
                        )
                        return False
                except KeyError:
                    pass
                    # TODO es gibt next_transitions ohne 'plan'
        return True

    def _merge_online_and_mimosa(self, measurement):
        # Edits self.traces_by_fileno[measurement['fileno']][*]['trace'][*]['offline']
        # and self.traces_by_fileno[measurement['fileno']][*]['trace'][*]['offline_aggregates'] in place
        # (appends data from measurement['energy_trace'])
        # If measurement['expected_trace'] exists, it is edited in place instead
        # "offline_aggregates" is the only data used later on by model.py's by_name / by_param dicts
        online_datapoints = []
        if "expected_trace" in measurement:
            traces = measurement["expected_trace"]
            traces = self.traces_by_fileno[measurement["fileno"]]
        else:
            traces = self.traces_by_fileno[measurement["fileno"]]
        for run_idx, run in enumerate(traces):
            for trace_part_idx in range(len(run["trace"])):
                online_datapoints.append((run_idx, trace_part_idx))
        for offline_idx, online_ref in enumerate(online_datapoints):
            online_run_idx, online_trace_part_idx = online_ref
            offline_trace_part = measurement["energy_trace"][offline_idx]
            online_trace_part = traces[online_run_idx]["trace"][online_trace_part_idx]

            if "offline" not in online_trace_part:
                online_trace_part["offline"] = [offline_trace_part]
            else:
                online_trace_part["offline"].append(offline_trace_part)

            paramkeys = sorted(online_trace_part["parameter"].keys())

            paramvalues = list()

            for paramkey in paramkeys:
                if type(online_trace_part["parameter"][paramkey]) is list:
                    paramvalues.append(
                        soft_cast_int(
                            online_trace_part["parameter"][paramkey][
                                measurement["repeat_id"]
                            ]
                        )
                    )
                else:
                    paramvalues.append(
                        soft_cast_int(online_trace_part["parameter"][paramkey])
                    )

            # NB: Unscheduled transitions do not have an 'args' field set.
            # However, they should only be caused by interrupts, and
            # interrupts don't have args anyways.
            if arg_support_enabled and "args" in online_trace_part:
                paramvalues.extend(map(soft_cast_int, online_trace_part["args"]))

            # TODO rename offline_aggregates to make it clear that this is what ends up in by_name / by_param and model.py
            if "offline_aggregates" not in online_trace_part:
                online_trace_part["offline_attributes"] = [
                    "power",
                    "duration",
                    "energy",
                ]
                # this is what ends up in by_name / by_param and is used by model.py
                online_trace_part["offline_aggregates"] = {
                    "power": [],
                    "duration": [],
                    "power_std": [],
                    "energy": [],
                    "paramkeys": [],
                    "param": [],
                }
                if online_trace_part["isa"] == "transition":
                    online_trace_part["offline_attributes"].extend(
                        [
                            "rel_energy_prev",
                            "rel_energy_next",
                            "rel_power_prev",
                            "rel_power_next",
                            "timeout",
                        ]
                    )
                    online_trace_part["offline_aggregates"]["rel_energy_prev"] = []
                    online_trace_part["offline_aggregates"]["rel_energy_next"] = []
                    online_trace_part["offline_aggregates"]["rel_power_prev"] = []
                    online_trace_part["offline_aggregates"]["rel_power_next"] = []
                    online_trace_part["offline_aggregates"]["timeout"] = []
                if "plot" in offline_trace_part:
                    online_trace_part["offline_support"] = [
                        "power_traces",
                        "timestamps",
                    ]
                    online_trace_part["offline_aggregates"]["power_traces"] = list()
                    online_trace_part["offline_aggregates"]["timestamps"] = list()

            # Note: All state/transitions are 20us "too long" due to injected
            # active wait states. These are needed to work around MIMOSA's
            # relatively low sample rate of 100 kHz (10us) and removed here.
            online_trace_part["offline_aggregates"]["power"].append(
                offline_trace_part["uW_mean"]
            )
            online_trace_part["offline_aggregates"]["duration"].append(
                offline_trace_part["us"] - 20
            )
            online_trace_part["offline_aggregates"]["power_std"].append(
                offline_trace_part["uW_std"]
            )
            online_trace_part["offline_aggregates"]["energy"].append(
                offline_trace_part["uW_mean"] * (offline_trace_part["us"] - 20)
            )
            online_trace_part["offline_aggregates"]["paramkeys"].append(paramkeys)
            online_trace_part["offline_aggregates"]["param"].append(paramvalues)
            if online_trace_part["isa"] == "transition":
                online_trace_part["offline_aggregates"]["rel_energy_prev"].append(
                    offline_trace_part["uW_mean_delta_prev"]
                    * (offline_trace_part["us"] - 20)
                )
                online_trace_part["offline_aggregates"]["rel_energy_next"].append(
                    offline_trace_part["uW_mean_delta_next"]
                    * (offline_trace_part["us"] - 20)
                )
                online_trace_part["offline_aggregates"]["rel_power_prev"].append(
                    offline_trace_part["uW_mean_delta_prev"]
                )
                online_trace_part["offline_aggregates"]["rel_power_next"].append(
                    offline_trace_part["uW_mean_delta_next"]
                )
                online_trace_part["offline_aggregates"]["timeout"].append(
                    offline_trace_part["timeout"]
                )

            if online_trace_part["isa"] == "state" and "plot" in offline_trace_part:
                online_trace_part["offline_aggregates"]["power_traces"].append(
                    offline_trace_part["plot"][1]
                )
                online_trace_part["offline_aggregates"]["timestamps"].append(
                    offline_trace_part["plot"][0]
                )

    def _merge_online_and_etlog(self, measurement):
        # Edits self.traces_by_fileno[measurement['fileno']][*]['trace'][*]['offline']
        # and self.traces_by_fileno[measurement['fileno']][*]['trace'][*]['offline_aggregates'] in place
        # (appends data from measurement['energy_trace'])
        online_datapoints = []
        traces = self.traces_by_fileno[measurement["fileno"]]
        for run_idx, run in enumerate(traces):
            for trace_part_idx in range(len(run["trace"])):
                online_datapoints.append((run_idx, trace_part_idx))
        for offline_idx, online_ref in enumerate(online_datapoints):
            online_run_idx, online_trace_part_idx = online_ref
            try:
                offline_trace_part = measurement["energy_trace"][offline_idx]
            except IndexError:
                logger.error(
                    f"While handling file #{measurement['fileno']} {measurement['info']}:"
                )
                logger.error(f"  offline energy_trace data is shorter than online data")
                logger.error(f"  len(online_datapoints) == {len(online_datapoints)}")
                logger.error(
                    f"  len(energy_trace) == {len(measurement['energy_trace'])}"
                )
                raise
            online_trace_part = traces[online_run_idx]["trace"][online_trace_part_idx]

            if "offline" not in online_trace_part:
                online_trace_part["offline"] = [offline_trace_part]
            else:
                online_trace_part["offline"].append(offline_trace_part)

            paramkeys = sorted(online_trace_part["parameter"].keys())

            paramvalues = list()

            for paramkey in paramkeys:
                if type(online_trace_part["parameter"][paramkey]) is list:
                    paramvalues.append(
                        soft_cast_int(
                            online_trace_part["parameter"][paramkey][
                                measurement["repeat_id"]
                            ]
                        )
                    )
                else:
                    paramvalues.append(
                        soft_cast_int(online_trace_part["parameter"][paramkey])
                    )

            # NB: Unscheduled transitions do not have an 'args' field set.
            # However, they should only be caused by interrupts, and
            # interrupts don't have args anyways.
            if arg_support_enabled and "args" in online_trace_part:
                paramvalues.extend(map(soft_cast_int, online_trace_part["args"]))

            if "offline_aggregates" not in online_trace_part:
                online_trace_part["offline_aggregates"] = {
                    "offline_attributes": ["power", "duration", "energy"],
                    "duration": list(),
                    "power": list(),
                    "power_std": list(),
                    "energy": list(),
                    "paramkeys": list(),
                    "param": list(),
                }
                if "plot" in offline_trace_part:
                    online_trace_part["offline_support"] = ["power_traces"]
                    online_trace_part["offline_aggregates"]["power_traces"] = list()
                if online_trace_part["isa"] == "transition":
                    online_trace_part["offline_aggregates"][
                        "offline_attributes"
                    ].extend(["rel_power_prev", "rel_power_next"])
                    online_trace_part["offline_aggregates"]["rel_energy_prev"] = list()
                    online_trace_part["offline_aggregates"]["rel_energy_next"] = list()
                    online_trace_part["offline_aggregates"]["rel_power_prev"] = list()
                    online_trace_part["offline_aggregates"]["rel_power_next"] = list()

            offline_aggregates = online_trace_part["offline_aggregates"]

            # if online_trace_part['isa'] == 'transitions':
            #    online_trace_part['offline_attributes'].extend(['rel_energy_prev', 'rel_energy_next'])
            #    offline_aggregates['rel_energy_prev'] = list()
            #    offline_aggregates['rel_energy_next'] = list()

            offline_aggregates["duration"].append(offline_trace_part["s"] * 1e6)
            offline_aggregates["power"].append(offline_trace_part["W_mean"] * 1e6)
            offline_aggregates["power_std"].append(offline_trace_part["W_std"] * 1e6)
            offline_aggregates["energy"].append(
                offline_trace_part["W_mean"] * offline_trace_part["s"] * 1e12
            )
            offline_aggregates["paramkeys"].append(paramkeys)
            offline_aggregates["param"].append(paramvalues)

            if "plot" in offline_trace_part:
                offline_aggregates["power_traces"].append(offline_trace_part["plot"][1])

            if online_trace_part["isa"] == "transition":
                offline_aggregates["rel_energy_prev"].append(
                    offline_trace_part["W_mean_delta_prev"]
                    * offline_trace_part["s"]
                    * 1e12
                )
                offline_aggregates["rel_energy_next"].append(
                    offline_trace_part["W_mean_delta_next"]
                    * offline_trace_part["s"]
                    * 1e12
                )
                offline_aggregates["rel_power_prev"].append(
                    offline_trace_part["W_mean_delta_prev"] * 1e6
                )
                offline_aggregates["rel_power_next"].append(
                    offline_trace_part["W_mean_delta_next"] * 1e6
                )

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
        if self.version == 0:
            self._preprocess_012(0)
        elif self.version == 1:
            self._preprocess_012(1)
        elif self.version == 2:
            self._preprocess_012(2)
        self.preprocessed = True
        self.save_cache()
        return self.traces

    def _preprocess_012(self, version):
        """Load raw MIMOSA data and turn it into measurements which are ready to be analyzed."""
        offline_data = []
        for i, filename in enumerate(self.filenames):

            if version == 0:

                with tarfile.open(filename) as tf:
                    self.setup_by_fileno.append(json.load(tf.extractfile("setup.json")))
                    self.traces_by_fileno.append(
                        json.load(tf.extractfile("src/apps/DriverEval/DriverLog.json"))
                    )
                    for member in tf.getmembers():
                        _, extension = os.path.splitext(member.name)
                        if extension == ".mim":
                            offline_data.append(
                                {
                                    "content": tf.extractfile(member).read(),
                                    "fileno": i,
                                    "info": member,
                                    "setup": self.setup_by_fileno[i],
                                    "with_traces": self.with_traces,
                                }
                            )

            elif version == 1:

                new_filenames = list()
                with tarfile.open(filename) as tf:
                    ptalog = self.ptalog

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
                        new_filenames.append("{}#{}".format(filename, j))
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
                                    "fileno": j,
                                    "info": member,
                                    "setup": self.setup_by_fileno[j],
                                    "repeat_id": repeat_id,
                                    "expected_trace": ptalog["traces"][j],
                                    "with_traces": self.with_traces,
                                }
                            )
                self.filenames = new_filenames

            elif version == 2:

                new_filenames = list()
                with tarfile.open(filename) as tf:
                    ptalog = self.ptalog
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
                        new_filenames.append("{}#{}".format(filename, j))
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
                                    "sync_mode": sync_mode,
                                    "fileno": j,
                                    "info": members[0],
                                    "setup": self.setup_by_fileno[j],
                                    "repeat_id": repeat_id,
                                    "expected_trace": traces,
                                    "with_traces": self.with_traces,
                                    "transition_names": list(
                                        map(
                                            lambda x: x["name"],
                                            ptalog["pta"]["transitions"],
                                        )
                                    ),
                                }
                            )
                self.filenames = new_filenames
                # TODO remove 'offline_aggregates' from pre-parse data and place
                # it under 'online_aggregates' or similar instead. This way, if
                # a .etlog file fails to parse, its corresponding duration data
                # will not linger in 'offline_aggregates' and confuse the hell
                # out of other code paths

        with Pool() as pool:
            if self.version <= 1:
                measurements = pool.map(_preprocess_mimosa, offline_data)
            elif self.version == 2:
                measurements = pool.map(_preprocess_etlog, offline_data)

        num_valid = 0
        for measurement in measurements:

            if "energy_trace" not in measurement:
                logger.warning(
                    "Skipping {ar:s}/{m:s}: {e:s}".format(
                        ar=self.filenames[measurement["fileno"]],
                        m=measurement["info"].name,
                        e="; ".join(measurement["datasource_errors"]),
                    )
                )
                continue

            if version == 0:
                # Strip the last state (it is not part of the scheduled measurement)
                measurement["energy_trace"].pop()
            elif version == 1:
                # The first online measurement is the UNINITIALIZED state. In v1,
                # it is not part of the expected PTA trace -> remove it.
                measurement["energy_trace"].pop(0)

            if version == 0 or version == 1:
                if self._measurement_is_valid_01(measurement):
                    self._merge_online_and_mimosa(measurement)
                    num_valid += 1
                else:
                    logger.warning(
                        "Skipping {ar:s}/{m:s}: {e:s}".format(
                            ar=self.filenames[measurement["fileno"]],
                            m=measurement["info"].name,
                            e=measurement["error"],
                        )
                    )
            elif version == 2:
                if self._measurement_is_valid_2(measurement):
                    try:
                        self._merge_online_and_etlog(measurement)
                        num_valid += 1
                    except Exception as e:
                        logger.warning(
                            f"Skipping #{measurement['fileno']} {measurement['info']}: {e}"
                        )
                else:
                    logger.warning(
                        "Skipping {ar:s}/{m:s}: {e:s}".format(
                            ar=self.filenames[measurement["fileno"]],
                            m=measurement["info"].name,
                            e=measurement["error"],
                        )
                    )
        logger.info(
            "{num_valid:d}/{num_total:d} measurements are valid".format(
                num_valid=num_valid, num_total=len(measurements)
            )
        )
        if version == 0:
            self.traces = self._concatenate_traces(self.traces_by_fileno)
        elif version == 1:
            self.traces = self._concatenate_traces(
                map(lambda x: x["expected_trace"], measurements)
            )
            self.traces = self._concatenate_traces(self.traces_by_fileno)
        elif version == 2:
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
            ]
            # Uncomment this line if you also want to analyze mean transition power
            # aggregate[key]['attributes'].append('power')
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


def _load_energytrace(data_string):
    """
    Load log data (raw energytrace .txt file, one line per event).

    :param log_data: raw energytrace log file in 4-column .txt format
    """

    lines = data_string.decode("ascii").split("\n")
    data_count = sum(map(lambda x: len(x) > 0 and x[0] != "#", lines))
    data_lines = filter(lambda x: len(x) > 0 and x[0] != "#", lines)

    data = np.empty((data_count, 4))
    hardware_states = [None for i in range(data_count)]

    for i, line in enumerate(data_lines):
        fields = line.split(" ")
        if len(fields) == 4:
            timestamp, current, voltage, total_energy = map(int, fields)
        elif len(fields) == 5:
            hardware_states[i] = fields[0]
            timestamp, current, voltage, total_energy = map(int, fields[1:])
        else:
            raise RuntimeError('cannot parse line "{}"'.format(line))
        data[i] = [timestamp, current, voltage, total_energy]

    interval_start_timestamp = data[1:, 0] * 1e-6
    interval_duration = (data[1:, 0] - data[:-1, 0]) * 1e-6
    interval_power = (data[1:, 3] - data[:-1, 3]) / (data[1:, 0] - data[:-1, 0]) * 1e-3

    m_duration_us = data[-1, 0] - data[0, 0]

    sample_rate = data_count / (m_duration_us * 1e-6)

    hardware_state_changes = list()
    if hardware_states[0]:
        prev_state = hardware_states[0]
        # timestamps start at data[1], so hardware state change indexes must start at 1, too
        for i, state in enumerate(hardware_states[1:]):
            if (
                state != prev_state
                and state != "0000000000000000"
                and prev_state != "0000000000000000"
            ):
                hardware_state_changes.append(i)
            if state != "0000000000000000":
                prev_state = state

    logger.debug(
        "got {} samples with {} seconds of log data ({} Hz)".format(
            data_count, m_duration_us * 1e-6, sample_rate
        )
    )

    return (
        interval_start_timestamp,
        interval_duration,
        interval_power,
        sample_rate,
        hardware_state_changes,
    )


class EnergyTraceWithBarcode:
    """
    EnergyTrace log loader for DFA traces.

    Expects an EnergyTrace log file generated via msp430-etv / energytrace-util
    and a dfatool-generated benchmark. An EnergyTrace log consits of a series
    of measurements. Each measurement has a timestamp, mean current, voltage,
    and cumulative energy since start of measurement. Each transition is
    preceded by a Code128 barcode embedded into the energy consumption by
    toggling a LED.

    Note that the baseline power draw of board and peripherals is not subtracted
    at the moment.
    """

    def __init__(
        self,
        voltage: float,
        state_duration: int,
        transition_names: list,
        with_traces=False,
    ):
        """
        Create a new EnergyTraceWithBarcode object.

        :param voltage: supply voltage [V], usually 3.3 V
        :param state_duration: state duration [ms]
        :param transition_names: list of transition names in PTA transition order.
            Needed to map barcode synchronization numbers to transitions.
        """
        self.voltage = voltage
        self.state_duration = state_duration * 1e-3
        self.transition_names = transition_names
        self.with_traces = with_traces
        self.errors = list()

        # TODO auto-detect
        self.led_power = 10e-3

        # multipass/include/object/ptalog.h#startTransition
        self.module_duration = 5e-3

        # multipass/include/object/ptalog.h#startTransition
        self.quiet_zone_duration = 60e-3

        # TODO auto-detect?
        # Note that we consider barcode duration after start, so only the
        # quiet zone -after- the code is relevant
        self.min_barcode_duration = 57 * self.module_duration + self.quiet_zone_duration
        self.max_barcode_duration = 68 * self.module_duration + self.quiet_zone_duration

    def load_data(self, log_data):
        """
        Load log data (raw energytrace .txt file, one line per event).

        :param log_data: raw energytrace log file in 4-column .txt format
        """

        if not zbar_available:
            logger.error("zbar module is not available")
            self.errors.append(
                'zbar module is not available. Try "apt install python3-zbar"'
            )
            self.interval_power = None
            return list()

        (
            self.interval_start_timestamp,
            self.interval_duration,
            self.interval_power,
            self.sample_rate,
            self.hw_statechange_indexes,
        ) = _load_energytrace(log_data[0])

    def ts_to_index(self, timestamp):
        """
        Convert timestamp in seconds to interval_start_timestamp / interval_duration / interval_power index.

        Returns the index of the interval which timestamp is part of.
        """
        return self._ts_to_index(timestamp, 0, len(self.interval_start_timestamp))

    def _ts_to_index(self, timestamp, left_index, right_index):
        if left_index == right_index:
            return left_index
        if left_index + 1 == right_index:
            return left_index

        mid_index = left_index + (right_index - left_index) // 2

        # I'm feeling lucky
        if (
            timestamp > self.interval_start_timestamp[mid_index]
            and timestamp
            <= self.interval_start_timestamp[mid_index]
            + self.interval_duration[mid_index]
        ):
            return mid_index

        if timestamp <= self.interval_start_timestamp[mid_index]:
            return self._ts_to_index(timestamp, left_index, mid_index)

        return self._ts_to_index(timestamp, mid_index, right_index)

    def analyze_states(self, traces, offline_index: int):
        """
        Split log data into states and transitions and return duration, energy, and mean power for each element.

        :param traces: expected traces, needed to synchronize with the measurement.
            traces is a list of runs, traces[*]['trace'] is a single run
            (i.e. a list of states and transitions, starting with a transition
            and ending with a state).
        :param offline_index: This function uses traces[*]['trace'][*]['online_aggregates']['duration'][offline_index] to find sync codes

        :param charges: raw charges (each element describes the charge in pJ transferred during 10 µs)
        :param trigidx: "charges" indexes corresponding to a trigger edge, see `trigger_edges`
        :param ua_func: charge(pJ) -> current(µA) function as returned by `calibration_function`

        :returns: maybe returns list of states and transitions, both starting andending with a state.
            Each element is a dict containing:
            * `isa`: 'state' or 'transition'
            * `clip_rate`: range(0..1) Anteil an Clipping im Energieverbrauch
            * `raw_mean`: Mittelwert der Rohwerte
            * `raw_std`: Standardabweichung der Rohwerte
            * `uW_mean`: Mittelwert der (kalibrierten) Leistungsaufnahme
            * `uW_std`: Standardabweichung der (kalibrierten) Leistungsaufnahme
            * `us`: Dauer
            if isa == 'transition, it also contains:
            * `timeout`: Dauer des vorherigen Zustands
            * `uW_mean_delta_prev`: Differenz zwischen uW_mean und uW_mean des vorherigen Zustands
            * `uW_mean_delta_next`: Differenz zwischen uW_mean und uW_mean des Folgezustands
        """

        energy_trace = list()
        first_sync = self.find_first_sync()

        if first_sync is None:
            logger.error("did not find initial synchronization pulse")
            return energy_trace

        expected_transitions = list()
        for trace_number, trace in enumerate(traces):
            for state_or_transition_number, state_or_transition in enumerate(
                trace["trace"]
            ):
                if state_or_transition["isa"] == "transition":
                    try:
                        expected_transitions.append(
                            (
                                state_or_transition["name"],
                                state_or_transition["online_aggregates"]["duration"][
                                    offline_index
                                ]
                                * 1e-6,
                            )
                        )
                    except IndexError:
                        self.errors.append(
                            'Entry #{} ("{}") in trace #{} has no duration entry for offline_index/repeat_id {}'.format(
                                state_or_transition_number,
                                state_or_transition["name"],
                                trace_number,
                                offline_index,
                            )
                        )
                        return energy_trace

        next_barcode = first_sync

        for name, duration in expected_transitions:
            bc, start, stop, end = self.find_barcode(next_barcode)
            if bc is None:
                logger.error('did not find transition "{}"'.format(name))
                break
            next_barcode = end + self.state_duration + duration
            logger.debug(
                '{} barcode "{}" area: {:0.2f} .. {:0.2f} / {:0.2f} seconds'.format(
                    offline_index, bc, start, stop, end
                )
            )
            if bc != name:
                logger.error('mismatch: expected "{}", got "{}"'.format(name, bc))
            logger.debug(
                "{} estimated transition area: {:0.3f} .. {:0.3f} seconds".format(
                    offline_index, end, end + duration
                )
            )

            transition_start_index = self.ts_to_index(end)
            transition_done_index = self.ts_to_index(end + duration) + 1
            state_start_index = transition_done_index
            state_done_index = (
                self.ts_to_index(end + duration + self.state_duration) + 1
            )

            logger.debug(
                "{} estimated transitionindex: {:0.3f} .. {:0.3f} seconds".format(
                    offline_index,
                    transition_start_index / self.sample_rate,
                    transition_done_index / self.sample_rate,
                )
            )

            transition_power_W = self.interval_power[
                transition_start_index:transition_done_index
            ]

            transition = {
                "isa": "transition",
                "W_mean": np.mean(transition_power_W),
                "W_std": np.std(transition_power_W),
                "s": duration,
                "s_coarse": self.interval_start_timestamp[transition_done_index]
                - self.interval_start_timestamp[transition_start_index],
            }

            if self.with_traces:
                timestamps = (
                    self.interval_start_timestamp[
                        transition_start_index:transition_done_index
                    ]
                    - self.interval_start_timestamp[transition_start_index]
                )
                transition["plot"] = (timestamps, transition_power_W)

            energy_trace.append(transition)

            if len(energy_trace) > 1:
                energy_trace[-1]["W_mean_delta_prev"] = (
                    energy_trace[-1]["W_mean"] - energy_trace[-2]["W_mean"]
                )

            state_power_W = self.interval_power[state_start_index:state_done_index]
            state = {
                "isa": "state",
                "W_mean": np.mean(state_power_W),
                "W_std": np.std(state_power_W),
                "s": self.state_duration,
                "s_coarse": self.interval_start_timestamp[state_done_index]
                - self.interval_start_timestamp[state_start_index],
            }

            if self.with_traces:
                timestamps = (
                    self.interval_start_timestamp[state_start_index:state_done_index]
                    - self.interval_start_timestamp[state_start_index]
                )
                state["plot"] = (timestamps, state_power_W)

            energy_trace.append(state)

            energy_trace[-2]["W_mean_delta_next"] = (
                energy_trace[-2]["W_mean"] - energy_trace[-1]["W_mean"]
            )

        expected_transition_count = len(expected_transitions)
        recovered_transition_ount = len(energy_trace) // 2

        if expected_transition_count != recovered_transition_ount:
            self.errors.append(
                "Expected {:d} transitions, got {:d}".format(
                    expected_transition_count, recovered_transition_ount
                )
            )

        return energy_trace

    def find_first_sync(self):
        # zbar unavailable
        if self.interval_power is None:
            return None
        # LED Power is approx. self.led_power W, use self.led_power/2 W above surrounding median as threshold
        sync_threshold_power = (
            np.median(self.interval_power[: int(3 * self.sample_rate)])
            + self.led_power / 3
        )
        for i, ts in enumerate(self.interval_start_timestamp):
            if ts > 2 and self.interval_power[i] > sync_threshold_power:
                return self.interval_start_timestamp[i - 300]
        return None

    def find_barcode(self, start_ts):
        """
        Return absolute position and content of the next barcode following `start_ts`.

        :param interval_ts: list of start timestamps (one per measurement interval) [s]
        :param interval_power: mean power per measurement interval [W]
        :param start_ts: timestamp at which to start looking for a barcode [s]
        """

        for i, ts in enumerate(self.interval_start_timestamp):
            if ts >= start_ts:
                start_position = i
                break

        # Lookaround: 100 ms in both directions
        lookaround = int(0.1 * self.sample_rate)

        # LED Power is approx. self.led_power W, use self.led_power/2 W above surrounding median as threshold
        sync_threshold_power = (
            np.median(
                self.interval_power[
                    start_position - lookaround : start_position + lookaround
                ]
            )
            + self.led_power / 3
        )

        logger.debug(
            "looking for barcode starting at {:0.2f} s, threshold is {:0.1f} mW".format(
                start_ts, sync_threshold_power * 1e3
            )
        )

        sync_area_start = None
        sync_start_ts = None
        sync_area_end = None
        sync_end_ts = None
        for i, ts in enumerate(self.interval_start_timestamp):
            if (
                sync_area_start is None
                and ts >= start_ts
                and self.interval_power[i] > sync_threshold_power
            ):
                sync_area_start = i - 300
                sync_start_ts = ts
            if (
                sync_area_start is not None
                and sync_area_end is None
                and ts > sync_start_ts + self.min_barcode_duration
                and (
                    ts > sync_start_ts + self.max_barcode_duration
                    or abs(sync_threshold_power - self.interval_power[i])
                    > self.led_power
                )
            ):
                sync_area_end = i
                sync_end_ts = ts
                break

        barcode_data = self.interval_power[sync_area_start:sync_area_end]

        logger.debug(
            "barcode search area: {:0.2f} .. {:0.2f} seconds ({} samples)".format(
                sync_start_ts, sync_end_ts, len(barcode_data)
            )
        )

        bc, start, stop, padding_bits = self.find_barcode_in_power_data(barcode_data)

        if bc is None:
            return None, None, None, None

        start_ts = self.interval_start_timestamp[sync_area_start + start]
        stop_ts = self.interval_start_timestamp[sync_area_start + stop]

        end_ts = (
            stop_ts + self.module_duration * padding_bits + self.quiet_zone_duration
        )

        # barcode content, barcode start timestamp, barcode stop timestamp, barcode end (stop + padding) timestamp
        return bc, start_ts, stop_ts, end_ts

    def find_barcode_in_power_data(self, barcode_data):

        min_power = np.min(barcode_data)
        max_power = np.max(barcode_data)

        # zbar seems to be confused by measurement (and thus image) noise
        # inside of barcodes. As our barcodes are only 1px high, this is
        # likely not trivial to fix.
        # -> Create a black and white (not grayscale) image to avoid this.
        # Unfortunately, this decreases resilience against background noise
        # (e.g. a not-exactly-idle peripheral device or CPU interrupts).
        image_data = np.around(
            1 - ((barcode_data - min_power) / (max_power - min_power))
        )
        image_data *= 255

        # zbar only returns the complete barcode position if it is at least
        # two pixels high. For a 1px barcode, it only returns its right border.

        width = len(image_data)
        height = 2

        image_data = bytes(map(int, image_data)) * height

        # img = Image.frombytes('L', (width, height), image_data).resize((width, 100))
        # img.save('/tmp/test-{}.png'.format(os.getpid()))

        zbimg = zbar.Image(width, height, "Y800", image_data)
        scanner = zbar.ImageScanner()
        scanner.parse_config("enable")

        if scanner.scan(zbimg):
            (sym,) = zbimg.symbols
            content = sym.data
            try:
                sym_start = sym.location[1][0]
            except IndexError:
                sym_start = 0
            sym_end = sym.location[0][0]

            match = re.fullmatch(r"T(\d+)", content)
            if match:
                content = self.transition_names[int(match.group(1))]

            # PTALog barcode generation operates on bytes, so there may be
            # additional non-barcode padding (encoded as LED off / image white).
            # Calculate the amount of extra bits to determine the offset until
            # the transition starts.
            padding_bits = len(Code128(sym.data, charset="B").modules) % 8

            # sym_start leaves out the first two bars, but we don't do anything about that here
            # sym_end leaves out the last three bars, each of which is one padding bit long.
            # as a workaround, we unconditionally increment padding_bits by three.
            padding_bits += 3

            return content, sym_start, sym_end, padding_bits
        else:
            logger.warning("unable to find barcode")
            return None, None, None, None


class EnergyTraceWithLogicAnalyzer:
    def __init__(
        self,
        voltage: float,
        state_duration: int,
        transition_names: list,
        with_traces=False,
    ):

        """
        Create a new EnergyTraceWithLogicAnalyzer object.

        :param voltage: supply voltage [V], usually 3.3 V
        :param state_duration: state duration [ms]
        :param transition_names: list of transition names in PTA transition order.
            Needed to map barcode synchronization numbers to transitions.
        """
        self.voltage = voltage
        self.state_duration = state_duration * 1e-3
        self.transition_names = transition_names
        self.with_traces = with_traces
        self.errors = list()

    def load_data(self, log_data):
        from dfatool.lennart.SigrokInterface import SigrokResult
        from dfatool.lennart.EnergyInterface import EnergyInterface

        # Daten laden
        self.sync_data = SigrokResult.fromString(log_data[0])
        (
            self.interval_start_timestamp,
            self.interval_duration,
            self.interval_power,
            self.sample_rate,
            self.hw_statechange_indexes,
        ) = _load_energytrace(log_data[1])

    def analyze_states(self, traces, offline_index: int):
        """
        Split log data into states and transitions and return duration, energy, and mean power for each element.

        :param traces: expected traces, needed to synchronize with the measurement.
            traces is a list of runs, traces[*]['trace'] is a single run
            (i.e. a list of states and transitions, starting with a transition
            and ending with a state).
        :param offline_index: This function uses traces[*]['trace'][*]['online_aggregates']['duration'][offline_index] to find sync codes

        :param charges: raw charges (each element describes the charge in pJ transferred during 10 µs)
        :param trigidx: "charges" indexes corresponding to a trigger edge, see `trigger_edges`
        :param ua_func: charge(pJ) -> current(µA) function as returned by `calibration_function`

        :returns: returns list of states and transitions, starting with a transition and ending with astate
            Each element is a dict containing:
            * `isa`: 'state' or 'transition'
            * `W_mean`: Mittelwert der Leistungsaufnahme
            * `W_std`: Standardabweichung der Leistungsaufnahme
            * `s`: Dauer
            if isa == 'transition, it also contains:
            * `W_mean_delta_prev`: Differenz zwischen W_mean und W_mean des vorherigen Zustands
            * `W_mean_delta_next`: Differenz zwischen W_mean und W_mean des Folgezustands
        """

        names = []
        for trace_number, trace in enumerate(traces):
            for state_or_transition in trace["trace"]:
                names.append(state_or_transition["name"])
        # print(names[:15])
        from dfatool.lennart.DataProcessor import DataProcessor

        dp = DataProcessor(
            sync_data=self.sync_data,
            et_timestamps=self.interval_start_timestamp,
            et_power=self.interval_power,
            hw_statechange_indexes=self.hw_statechange_indexes,
        )
        dp.run()
        energy_trace_new = dp.getStatesdfatool(
            state_sleep=self.state_duration, with_traces=self.with_traces
        )
        # Uncomment to plot traces
        if os.getenv("DFATOOL_PLOT_LASYNC") is not None and offline_index == int(
            os.getenv("DFATOOL_PLOT_LASYNC")
        ):
            dp.plot()  # <- plot traces with sync annotatons
            # dp.plot(names) # <- plot annotated traces (with state/transition names)
        if os.getenv("DFATOOL_EXPORT_LASYNC") is not None:
            filename = os.getenv("DFATOOL_EXPORT_LASYNC") + f"_{offline_index}.json"
            with open(filename, "w") as f:
                json.dump(dp.export_sync(), f, cls=NpEncoder)
            logger.info("Exported data and LA sync timestamps to {filename}")

        energy_trace = list()
        expected_transitions = list()

        # Print for debug purposes
        # for number, name in enumerate(names):
        #    if "P15_8MW" in name:
        #        print(name, energy_trace_new[number]["W_mean"])

        # st = ""
        # for i, x in enumerate(energy_trace_new[-10:]):
        #    #st += "(%s|%s|%s)" % (energy_trace[i-10]["name"],x['W_mean'],x['s'])
        #    st += "(%s|%s|%s)\n" % (energy_trace[i-10]["s"], x['s'], x['W_mean'])

        # print(st, "\n_______________________")
        # print(len(self.sync_data.timestamps), " - ", len(energy_trace_new), " - ", len(energy_trace), " - ", ",".join([str(x["s"]) for x in energy_trace_new[-6:]]), " - ", ",".join([str(x["s"]) for x in energy_trace[-6:]]))
        # if len(energy_trace_new) < len(energy_trace):
        #    return None

        return energy_trace_new


class EnergyTraceWithTimer(EnergyTraceWithLogicAnalyzer):
    def __init__(
        self,
        voltage: float,
        state_duration: int,
        transition_names: list,
        with_traces=False,
    ):

        """
        Create a new EnergyTraceWithLogicAnalyzer object.

        :param voltage: supply voltage [V], usually 3.3 V
        :param state_duration: state duration [ms]
        :param transition_names: list of transition names in PTA transition order.
            Needed to map barcode synchronization numbers to transitions.
        """

        self.voltage = voltage
        self.state_duration = state_duration * 1e-3
        self.transition_names = transition_names
        self.with_traces = with_traces
        self.errors = list()

        super().__init__(voltage, state_duration, transition_names, with_traces)

    def load_data(self, log_data):
        self.sync_data = None
        (
            self.interval_start_timestamp,
            self.interval_duration,
            self.interval_power,
            self.sample_rate,
            self.hw_statechange_indexes,
        ) = _load_energytrace(log_data[0])

    def analyze_states(self, traces, offline_index: int):

        # Start "Synchronization pulse"
        timestamps = [0, 10, 1e6, 1e6 + 10]

        # The first trace doesn't start immediately, append offset saved by OnboarTimerHarness
        timestamps.append(timestamps[-1] + traces[0]["start_offset"][offline_index])
        for tr in traces:
            for t in tr["trace"]:
                # print(t["online_aggregates"]["duration"][offline_index])
                try:
                    timestamps.append(
                        timestamps[-1]
                        + t["online_aggregates"]["duration"][offline_index]
                    )
                except IndexError:
                    self.errors.append(
                        f"""offline_index {offline_index} missing in trace {tr["id"]}"""
                    )
                    return list()

        # print(timestamps)

        # Stop "Synchronization pulses". The first one has already started.
        timestamps.extend(np.array([10, 1e6, 1e6 + 10]) + timestamps[-1])
        timestamps.extend(np.array([0, 10, 1e6, 1e6 + 10]) + 250e3 + timestamps[-1])

        timestamps = list(np.array(timestamps) * 1e-6)

        from dfatool.lennart.SigrokInterface import SigrokResult

        self.sync_data = SigrokResult(timestamps, False)
        return super().analyze_states(traces, offline_index)


class MIMOSA:
    """
    MIMOSA log loader for DFA traces with auto-calibration.

    Expects a MIMOSA log file generated via dfatool and a dfatool-generated
    benchmark. A MIMOSA log consists of a series of measurements. Each measurement
    gives the total charge (in pJ) and binary buzzer/trigger value during a 10µs interval.

    There must be a calibration run consisting of at least two seconds with disconnected DUT,
    two seconds with 1 kOhm (984 Ohm), and two seconds with 100 kOhm (99013 Ohm) resistor at
    the start. The first ten seconds of data are reserved for calbiration and must not contain
    measurements, as trigger/buzzer signals are ignored in this time range.

    Resulting data is a list of state/transition/state/transition/... measurements.
    """

    def __init__(self, voltage: float, shunt: int, with_traces=False):
        """
        Initialize MIMOSA loader for a specific voltage and shunt setting.

        :param voltage: MIMOSA DUT supply voltage (V)
        :para mshunt: MIMOSA Shunt (Ohms)
        """
        self.voltage = voltage
        self.shunt = shunt
        self.with_traces = with_traces
        self.r1 = 984  # "1k"
        self.r2 = 99013  # "100k"
        self.errors = list()

    def charge_to_current_nocal(self, charge):
        """
        Convert charge per 10µs (in pJ) to mean currents (in µA) without accounting for calibration.

        :param charge: numpy array of charges (pJ per 10µs) as returned by `load_data` or `load_file`

        :returns: numpy array of mean currents (µA per 10µs)
        """
        ua_max = 1.836 / self.shunt * 1_000_000
        ua_step = ua_max / 65535
        return charge * ua_step

    def _load_tf(self, tf):
        """
        Load MIMOSA log data from an open `tarfile` instance.

        :param tf: `tarfile` instance

        :returns: (numpy array of charges (pJ per 10µs), numpy array of triggers (0/1 int, per 10µs))
        """
        num_bytes = tf.getmember("/tmp/mimosa//mimosa_scale_1.tmp").size
        charges = np.ndarray(shape=(int(num_bytes / 4)), dtype=np.int32)
        triggers = np.ndarray(shape=(int(num_bytes / 4)), dtype=np.int8)
        with tf.extractfile("/tmp/mimosa//mimosa_scale_1.tmp") as f:
            content = f.read()
            iterator = struct.iter_unpack("<I", content)
            i = 0
            for word in iterator:
                charges[i] = word[0] >> 4
                triggers[i] = (word[0] & 0x08) >> 3
                i += 1
        return charges, triggers

    def load_data(self, raw_data):
        """
        Load MIMOSA log data from a MIMOSA log file passed as raw byte string

        :param raw_data: MIMOSA log file, passed as raw byte string

        :returns: (numpy array of charges (pJ per 10µs), numpy array of triggers (0/1 int, per 10µs))
        """
        with io.BytesIO(raw_data) as data_object:
            with tarfile.open(fileobj=data_object) as tf:
                return self._load_tf(tf)

    def load_file(self, filename):
        """
        Load MIMOSA log data from a MIMOSA log file

        :param filename: MIMOSA log file

        :returns: (numpy array of charges (pJ per 10µs), numpy array of triggers (0/1 int, per 10µs))
        """
        with tarfile.open(filename) as tf:
            return self._load_tf(tf)

    def currents_nocal(self, charges):
        """
        Convert charges (pJ per 10µs) to mean currents without accounting for calibration.

        :param charges: numpy array of charges (pJ per 10µs)

        :returns: numpy array of currents (mean µA per 10µs)"""
        ua_max = 1.836 / self.shunt * 1_000_000
        ua_step = ua_max / 65535
        return charges.astype(np.double) * ua_step

    def trigger_edges(self, triggers):
        """
        Return indexes of trigger edges (both 0->1 and 1->0) in log data.

        Ignores the first 10 seconds, which are used for calibration and may
        contain bogus triggers due to DUT resets.

        :param triggers: trigger array (int, 0/1) as returned by load_data

        :returns: list of int (trigger indices, e.g. [2000000, ...] means the first trigger appears in charges/currents interval 2000000 -> 20s after start of measurements. Keep in mind that each interval is 10µs long, not 1µs, so index values are not µs timestamps)
        """
        trigidx = []

        if len(triggers) < 1_000_000:
            self.errors.append("MIMOSA log is too short")
            return trigidx

        prevtrig = triggers[999_999]

        # if the first trigger is high (i.e., trigger/buzzer pin is active before the benchmark starts),
        # something went wrong and are unable to determine when the first
        # transition starts.
        if prevtrig != 0:
            self.errors.append(
                "Unable to find start of first transition (log starts with trigger == {} != 0)".format(
                    prevtrig
                )
            )

        # if the last trigger is high (i.e., trigger/buzzer pin is active when the benchmark ends),
        # it terminated in the middle of a transition -- meaning that it was not
        # measured in its entirety.
        if triggers[-1] != 0:
            self.errors.append("Log ends during a transition".format(prevtrig))

        # the device is reset for MIMOSA calibration in the first 10s and may
        # send bogus interrupts -> bogus triggers
        for i in range(1_000_000, triggers.shape[0]):
            trig = triggers[i]
            if trig != prevtrig:
                # Due to MIMOSA's integrate-read-reset cycle, the charge/current
                # interval belonging to this trigger comes two intervals (20µs) later
                trigidx.append(i + 2)
            prevtrig = trig
        return trigidx

    def calibration_edges(self, currents):
        """
        Return start/stop indexes of calibration measurements.

        :param currents: uncalibrated currents as reported by MIMOSA. For best results,
            it may help to use a running mean, like so:
            `currents = running_mean(currents_nocal(..., 10))`

        :returns: indices of calibration events in MIMOSA data:
            (disconnect start, disconnect stop, R1 (1k) start, R1 (1k) stop, R2 (100k) start, R2 (100k) stop)
            indices refer to charges/currents arrays, so 0 refers to the first 10µs interval, 1 to the second, and so on.
        """
        r1idx = 0
        r2idx = 0
        ua_r1 = self.voltage / self.r1 * 1_000_000
        # first second may be bogus
        for i in range(100_000, len(currents)):
            if r1idx == 0 and currents[i] > ua_r1 * 0.6:
                r1idx = i
            elif (
                r1idx != 0
                and r2idx == 0
                and i > (r1idx + 180_000)
                and currents[i] < ua_r1 * 0.4
            ):
                r2idx = i
        # 2s disconnected, 2s r1, 2s r2  with r1 < r2  ->  ua_r1 > ua_r2
        # allow 5ms buffer in both directions to account for bouncing relais contacts
        return (
            r1idx - 180_500,
            r1idx - 500,
            r1idx + 500,
            r2idx - 500,
            r2idx + 500,
            r2idx + 180_500,
        )

    def calibration_function(self, charges, cal_edges):
        """
        Calculate calibration function from previously determined calibration edges.

        :param charges: raw charges from MIMOSA
        :param cal_edges: calibration edges as returned by calibration_edges

        :returns: (calibration_function, calibration_data):
            calibration_function -- charge in pJ (float) -> current in uA (float).
                Converts the amount of charge in a 10 µs interval to the
                mean current during the same interval.
            calibration_data -- dict containing the following keys:
                edges -- calibration points in the log file, in µs
                offset -- ...
                offset2 --  ...
                slope_low -- ...
                slope_high -- ...
                add_low -- ...
                add_high -- ..
                r0_err_uW -- mean error of uncalibrated data at "∞ Ohm" in µW
                r0_std_uW -- standard deviation of uncalibrated data at "∞ Ohm" in µW
                r1_err_uW -- mean error of uncalibrated data at 1 kOhm
                r1_std_uW -- stddev at 1 kOhm
                r2_err_uW -- mean error at 100 kOhm
                r2_std_uW -- stddev at 100 kOhm
        """
        dis_start, dis_end, r1_start, r1_end, r2_start, r2_end = cal_edges
        if dis_start < 0:
            dis_start = 0
        chg_r0 = charges[dis_start:dis_end]
        chg_r1 = charges[r1_start:r1_end]
        chg_r2 = charges[r2_start:r2_end]
        cal_0_mean = np.mean(chg_r0)
        cal_r1_mean = np.mean(chg_r1)
        cal_r2_mean = np.mean(chg_r2)

        ua_r1 = self.voltage / self.r1 * 1_000_000
        ua_r2 = self.voltage / self.r2 * 1_000_000

        if cal_r2_mean > cal_0_mean:
            b_lower = (ua_r2 - 0) / (cal_r2_mean - cal_0_mean)
        else:
            logger.warning("0 uA == %.f uA during calibration" % (ua_r2))
            b_lower = 0

        b_upper = (ua_r1 - ua_r2) / (cal_r1_mean - cal_r2_mean)

        a_lower = -b_lower * cal_0_mean
        a_upper = -b_upper * cal_r2_mean

        if self.shunt == 680:
            # R1 current is higher than shunt range -> only use R2 for calibration
            def calfunc(charge):
                if charge < cal_0_mean:
                    return 0
                else:
                    return charge * b_lower + a_lower

        else:

            def calfunc(charge):
                if charge < cal_0_mean:
                    return 0
                if charge <= cal_r2_mean:
                    return charge * b_lower + a_lower
                else:
                    return charge * b_upper + a_upper + ua_r2

        caldata = {
            "edges": [x * 10 for x in cal_edges],
            "offset": cal_0_mean,
            "offset2": cal_r2_mean,
            "slope_low": b_lower,
            "slope_high": b_upper,
            "add_low": a_lower,
            "add_high": a_upper,
            "r0_err_uW": np.mean(self.currents_nocal(chg_r0)) * self.voltage,
            "r0_std_uW": np.std(self.currents_nocal(chg_r0)) * self.voltage,
            "r1_err_uW": (np.mean(self.currents_nocal(chg_r1)) - ua_r1) * self.voltage,
            "r1_std_uW": np.std(self.currents_nocal(chg_r1)) * self.voltage,
            "r2_err_uW": (np.mean(self.currents_nocal(chg_r2)) - ua_r2) * self.voltage,
            "r2_std_uW": np.std(self.currents_nocal(chg_r2)) * self.voltage,
        }

        # print("if charge < %f : return 0" % cal_0_mean)
        # print("if charge <= %f : return charge * %f + %f" % (cal_r2_mean, b_lower, a_lower))
        # print("else : return charge * %f + %f + %f" % (b_upper, a_upper, ua_r2))

        return calfunc, caldata

    def analyze_states(self, charges, trigidx, ua_func):
        """
        Split log data into states and transitions and return duration, energy, and mean power for each element.

        :param charges: raw charges (each element describes the charge in pJ transferred during 10 µs)
        :param trigidx: "charges" indexes corresponding to a trigger edge, see `trigger_edges`
        :param ua_func: charge(pJ) -> current(µA) function as returned by `calibration_function`

        :returns: list of states and transitions, both starting andending with a state.
            Each element is a dict containing:
            * `isa`: 'state' or 'transition'
            * `clip_rate`: range(0..1) Anteil an Clipping im Energieverbrauch
            * `raw_mean`: Mittelwert der Rohwerte
            * `raw_std`: Standardabweichung der Rohwerte
            * `uW_mean`: Mittelwert der (kalibrierten) Leistungsaufnahme
            * `uW_std`: Standardabweichung der (kalibrierten) Leistungsaufnahme
            * `us`: Dauer
            if isa == 'transition, it also contains:
            * `timeout`: Dauer des vorherigen Zustands
            * `uW_mean_delta_prev`: Differenz zwischen uW_mean und uW_mean des vorherigen Zustands
            * `uW_mean_delta_next`: Differenz zwischen uW_mean und uW_mean des Folgezustands
        """
        previdx = 0
        is_state = True
        iterdata = []

        # The last state (between the last transition and end of file) may also
        # be important. Pretend it ends when the log ends.
        trigger_indices = trigidx.copy()
        trigger_indices.append(len(charges))

        for idx in trigger_indices:
            range_raw = charges[previdx:idx]
            range_ua = ua_func(range_raw)

            isa = "state"
            if not is_state:
                isa = "transition"

            data = {
                "isa": isa,
                "clip_rate": np.mean(range_raw == 65535),
                "raw_mean": np.mean(range_raw),
                "raw_std": np.std(range_raw),
                "uW_mean": np.mean(range_ua * self.voltage),
                "uW_std": np.std(range_ua * self.voltage),
                "us": (idx - previdx) * 10,
            }

            if self.with_traces:
                data["plot"] = (
                    np.arange(len(range_ua)) * 1e-5,
                    range_ua * self.voltage * 1e-6,
                )

            if isa == "transition":
                # subtract average power of previous state
                # (that is, the state from which this transition originates)
                data["uW_mean_delta_prev"] = data["uW_mean"] - iterdata[-1]["uW_mean"]
                # placeholder to avoid extra cases in the analysis
                data["uW_mean_delta_next"] = data["uW_mean"]
                data["timeout"] = iterdata[-1]["us"]
            elif len(iterdata) > 0:
                # subtract average power of next state
                # (the state into which this transition leads)
                iterdata[-1]["uW_mean_delta_next"] = (
                    iterdata[-1]["uW_mean"] - data["uW_mean"]
                )

            iterdata.append(data)

            previdx = idx
            is_state = not is_state
        return iterdata
