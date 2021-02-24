#!/usr/bin/env python3

import hashlib
import json
import logging
import numpy as np
import os
from multiprocessing import Pool
from .utils import NpEncoder

logger = logging.getLogger(__name__)


def PELT_get_changepoints(index, penalty, algo):
    res = (index, penalty, algo.predict(pen=penalty))
    return res


# calculates the raw_states for a measurement. num_measurement is used to identify the return value
# penalty, model and jump are directly passed to pelt
def PELT_get_raw_states(num_measurement, algo, penalty):
    changepoints = algo.predict(pen=penalty)
    substates = list()
    start_index = 0
    end_index = 0
    # calc metrics for all states
    for changepoint in changepoints:
        # start_index of state is end_index of previous one
        # (Transitions are instantaneous)
        start_index = end_index
        end_index = changepoint - 1
        substate = (start_index, end_index)
        substates.append(substate)

    return num_measurement, substates


class PELT:
    def __init__(self, **kwargs):
        self.algo = "pelt"
        self.model = "l1"
        self.jump = 1
        self.min_dist = 10
        self.name_filter = None
        self.refinement_threshold = 200e-6  # 200 µW
        self.range_min = 1
        self.range_max = 89
        self.stretch = 1
        self.with_multiprocessing = True
        self.cache_dir = "cache"
        self.__dict__.update(kwargs)

        self.jump = int(self.jump)
        self.min_dist = int(self.min_dist)
        self.stretch = int(self.stretch)

        if os.getenv("DFATOOL_PELT_MODEL"):
            # https://centre-borelli.github.io/ruptures-docs/user-guide/costs/costl1/
            self.model = os.getenv("DFATOOL_PELT_MODEL")
        if os.getenv("DFATOOL_PELT_JUMP"):
            self.jump = int(os.getenv("DFATOOL_PELT_JUMP"))
        if os.getenv("DFATOOL_PELT_MIN_DIST"):
            self.min_dist = int(os.getenv("DFATOOL_PELT_MIN_DIST"))

    # signals: a set of uW measurements belonging to a single parameter configuration (i.e., a single by_param entry)
    def needs_refinement(self, signals):
        count = 0
        for signal in signals:
            if len(signal) < 30:
                continue

            p1, median, p99 = np.percentile(signal[5:-5], (1, 50, 99))

            if median - p1 > self.refinement_threshold:
                count += 1
            elif p99 - median > self.refinement_threshold:
                count += 1
        refinement_ratio = count / len(signals)
        return refinement_ratio > 0.3

    def norm_signal(self, signal, scaler=25):
        max_val = max(np.abs(signal))
        normed_signal = np.zeros(shape=len(signal))
        for i, signal_i in enumerate(signal):
            normed_signal[i] = signal_i / max_val
            normed_signal[i] = normed_signal[i] * scaler
        return normed_signal

    def cache_key(self, traces, penalty, num_changepoints):
        config = [
            traces,
            penalty,
            num_changepoints,
            self.algo,
            self.model,
            self.jump,
            self.min_dist,
            self.range_min,
            self.range_max,
            self.stretch,
        ]
        cache_key = hashlib.sha256(
            json.dumps(config, cls=NpEncoder).encode()
        ).hexdigest()
        return cache_key

    def save_cache(self, traces, penalty, num_changepoints, data):
        if self.cache_dir is None:
            return
        cache_key = self.cache_key(traces, penalty, num_changepoints)

        try:
            os.mkdir(self.cache_dir)
        except FileExistsError:
            pass

        try:
            os.mkdir(f"{self.cache_dir}/{cache_key[:2]}")
        except FileExistsError:
            pass

        with open(f"{self.cache_dir}/{cache_key[:2]}/pelt-{cache_key}.json", "w") as f:
            json.dump(data, f, cls=NpEncoder)

    def load_cache(self, traces, penalty, num_changepoints):
        cache_key = self.cache_key(traces, penalty, num_changepoints)
        try:
            with open(
                f"{self.cache_dir}/{cache_key[:2]}/pelt-{cache_key}.json", "r"
            ) as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except json.decoder.JSONDecodeError:
            logger.warning(
                f"Ignoring invalid cache entry {self.cache_dir}/{cache_key[:2]}/pelt-{cache_key}.json"
            )
            return None

    def get_penalty_and_changepoints(self, traces, penalty=None, num_changepoints=None):
        list_of_lists = type(traces[0]) is list or type(traces[0]) is np.ndarray
        if not list_of_lists:
            traces = [traces]

        data = self.load_cache(traces, penalty, num_changepoints)
        if data:
            for res in data:
                if type(res[1]) is dict:
                    str_keys = list(res[1].keys())
                    for k in str_keys:
                        res[1][int(k)] = res[1].pop(k)
            if list_of_lists:
                return data
            return data[0]

        data = self.calculate_penalty_and_changepoints(
            traces, penalty, num_changepoints
        )
        self.save_cache(traces, penalty, num_changepoints, data)

        if list_of_lists:
            return data
        return data[0]

    def calculate_penalty_and_changepoints(
        self, traces, penalty=None, num_changepoints=None
    ):
        # imported here as ruptures is only used for changepoint detection.
        # This way, dfatool can be used without having ruptures installed as
        # long as --pelt isn't active.
        import ruptures

        if self.stretch > 1:
            traces = list(
                map(
                    lambda trace: np.interp(
                        np.linspace(
                            0, len(trace) - 1, (len(trace) - 1) * self.stretch + 1
                        ),
                        np.arange(len(trace)),
                        trace,
                    ),
                    traces,
                )
            )
        elif self.stretch < -1:
            ds_factor = -self.stretch
            new_traces = list()
            for trace in traces:
                if trace.shape[0] % ds_factor:
                    trace = np.array(
                        list(trace)
                        + [
                            trace[-1]
                            for i in range(ds_factor - (trace.shape[0] % ds_factor))
                        ]
                    )
                new_traces.append(trace.reshape(-1, ds_factor).mean(axis=1))
            traces = new_traces

        algos = list()
        queue = list()
        changepoints_by_penalty_by_trace = list()
        results = list()

        for i in range(len(traces)):
            if self.algo == "dynp":
                # https://centre-borelli.github.io/ruptures-docs/user-guide/detection/dynp/
                algo = ruptures.Dynp(
                    model=self.model, jump=self.jump, min_size=self.min_dist
                )
            else:
                # https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/
                algo = ruptures.Pelt(
                    model=self.model, jump=self.jump, min_size=self.min_dist
                )
            algo = algo.fit(self.norm_signal(traces[i]))
            algos.append(algo)

        for i in range(len(traces)):
            changepoints_by_penalty_by_trace.append(dict())
            if penalty is not None:
                queue.append((i, penalty, algos[i]))
            elif self.algo == "dynp" and num_changepoints is not None:
                queue.append((i, None, algos[i]))
            else:
                for range_penalty in range(self.range_min, self.range_max):
                    queue.append((i, range_penalty, algos[i]))

        if self.with_multiprocessing:
            with Pool() as pool:
                changepoints_by_trace = pool.starmap(PELT_get_changepoints, queue)
        else:
            changepoints_by_trace = map(lambda x: PELT_get_changepoints(*x), queue)

        for i, range_penalty, changepoints in changepoints_by_trace:
            if len(changepoints) and changepoints[-1] == len(traces[i]):
                changepoints.pop()
            if len(changepoints) and changepoints[0] == 0:
                changepoints.pop(0)
            if self.stretch > 1:
                changepoints = list(
                    np.array(
                        np.around(np.array(changepoints) / self.stretch), dtype=np.int
                    )
                )
            elif self.stretch < -1:
                ds_factor = -self.stretch
                changepoints = list(
                    np.array(
                        np.around(np.array(changepoints) * ds_factor), dtype=np.int
                    )
                )
            changepoints_by_penalty_by_trace[i][range_penalty] = changepoints

        for i in range(len(traces)):
            changepoints_by_penalty = changepoints_by_penalty_by_trace[i]
            if penalty is not None:
                results.append((penalty, changepoints_by_penalty))
            elif self.algo == "dynp" and num_changepoints is not None:
                results.append((None, {0: changepoints_by_penalty[None]}))
            else:
                results.append(
                    (
                        self.find_penalty(changepoints_by_penalty),
                        changepoints_by_penalty,
                    )
                )

        return results

    def find_penalty(self, changepoints_by_penalty):
        changepoint_counts = list()
        for i in range(self.range_min, self.range_max):
            changepoint_counts.append(len(changepoints_by_penalty[i]))

        start_index = -1
        end_index = -1
        longest_start = -1
        longest_end = -1
        prev_val = -1
        for i, changepoint_count in enumerate(changepoint_counts):
            if changepoint_count != prev_val:
                end_index = i - 1
                if end_index - start_index > longest_end - longest_start:
                    longest_start = start_index
                    longest_end = end_index
                start_index = i
            if i == len(changepoint_counts) - 1:
                end_index = i
                if end_index - start_index > longest_end - longest_start:
                    longest_start = start_index
                    longest_end = end_index
                start_index = i
            prev_val = changepoint_count
        middle_of_plateau = longest_start + (longest_end - longest_start) // 2

        return self.range_min + middle_of_plateau

    def get_changepoints(self, traces, **kwargs):
        results = self.get_penalty_and_changepoints(traces, **kwargs)
        if type(results) is list:
            return list(map(lambda res: res[1][res[0]], results))
        return results[1][results[0]]

    def get_penalty(self, traces, **kwargs):
        results = self.get_penalty_and_changepoints(traces, **kwargs)
        if type(results) is list:
            return list(map(lambda res: res[0]))
        return res[0]

    def calc_raw_states(
        self,
        timestamps,
        signals,
        changepoints_by_signal,
        num_changepoints,
        opt_model=None,
    ):
        """
        Calculate substates for signals (assumed to belong to a single parameter configuration).

        :returns: (substate_counts, substate_data)
            substates_counts = [number of substates in first changepoints_by_signal entry, number of substates in second entry, ...]
            substate_data = [data for first sub-state, data for second sub-state, ...]
                data = {"duration": [durations of corresponding sub-state in signals[i] where changepoints_by_signal[i] == num_changepoints]}
            Note that len(substate_counts) >= len(substate_data). substate_counts gives the number of sub-states of each signal in signals
                (substate_counts[i] == number of substates in signals[i]). substate_data only contains entries for signals which have
                num_changepoints + 1 substates.

        List of substates with duration and mean power: [(substate 1 duration, substate 1 power), ...]
        """

        substate_data = list()
        substate_counts = list()
        usable_measurements = list()
        expected_substate_count = num_changepoints + 1

        for i, changepoints in enumerate(changepoints_by_signal):
            substates = list()
            start_index = 0
            end_index = 0
            for changepoint in changepoints:
                # start_index of state is end_index of previous one
                # (Transitions are instantaneous)
                start_index = end_index
                end_index = changepoint
                substate = (start_index, end_index)
                substates.append(substate)

            substates.append((end_index, len(signals[i]) - 1))

            substate_counts.append(len(substates))
            if len(substates) == expected_substate_count:
                usable_measurements.append((i, substates))

        if len(usable_measurements) <= len(changepoints_by_signal) * 0.5:
            logger.info(
                f"Only {len(usable_measurements)} of {len(changepoints_by_signal)} measurements have {expected_substate_count} sub-states. Try lowering the jump step size"
            )
        else:
            logger.debug(
                f"{len(usable_measurements)} of {len(changepoints_by_signal)} measurements have {expected_substate_count} sub-states"
            )

        for i in range(expected_substate_count):
            substate_data.append(
                {
                    "duration": list(),
                    "power": list(),
                    "power_std": list(),
                    "signals_index": list(),
                }
            )

        for num_measurement, substates in usable_measurements:
            for i, substate in enumerate(substates):
                power_trace = signals[num_measurement][substate[0] : substate[1]]
                mean_power = np.mean(power_trace)
                std_power = np.std(power_trace)
                duration = (
                    timestamps[num_measurement][substate[1]]
                    - timestamps[num_measurement][substate[0]]
                )
                substate_data[i]["duration"].append(duration)
                substate_data[i]["power"].append(mean_power)
                substate_data[i]["power_std"].append(std_power)
                substate_data[i]["signals_index"].append(num_measurement)

        return substate_counts, substate_data
