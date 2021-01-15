#!/usr/bin/env python3

import logging
import numpy as np
import os
from multiprocessing import Pool

logger = logging.getLogger(__name__)


def PELT_get_changepoints(algo, penalty):
    res = (penalty, algo.predict(pen=penalty))
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
        self.num_samples = None
        self.refinement_threshold = 200e-6  # 200 µW
        self.range_min = 0
        self.range_max = 100
        self.with_multiprocessing = True
        self.__dict__.update(kwargs)

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
            # test
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

    def get_penalty_and_changepoints(self, signal, penalty=None, num_changepoints=None):
        # imported here as ruptures is only used for changepoint detection.
        # This way, dfatool can be used without having ruptures installed as
        # long as --pelt isn't active.
        import ruptures

        if self.num_samples is not None and len(signal) > self.num_samples:
            self.jump = len(signal) // int(self.num_samples)
        else:
            self.jump = 1

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
        algo = algo.fit(self.norm_signal(signal))

        if penalty is not None:
            changepoints = algo.predict(pen=penalty)
            if len(changepoints) and changepoints[-1] == len(signal):
                changepoints.pop()
            if len(changepoints) and changepoints[0] == 0:
                changepoints.pop(0)
            return penalty, changepoints

        if self.algo == "dynp" and num_changepoints is not None:
            changepoints = algo.predict(n_bkps=num_changepoints)
            if len(changepoints) and changepoints[-1] == len(signal):
                changepoints.pop()
            if len(changepoints) and changepoints[0] == 0:
                changepoints.pop(0)
            return None, changepoints

        queue = list()
        for i in range(0, 100):
            queue.append((algo, i))
        if self.with_multiprocessing:
            with Pool() as pool:
                changepoints = pool.starmap(PELT_get_changepoints, queue)
        else:
            changepoints = map(lambda x: PELT_get_changepoints(*x), queue)
        changepoints_by_penalty = dict()
        for res in changepoints:
            if len(res[1]) > 0 and res[1][-1] == len(signal):
                res[1].pop()
            changepoints_by_penalty[res[0]] = res[1]
        changepoint_counts = list()
        for i in range(0, 100):
            changepoint_counts.append(len(changepoints_by_penalty[i]))

        start_index = -1
        end_index = -1
        longest_start = -1
        longest_end = -1
        prev_val = -1
        for i, num_changepoints in enumerate(changepoint_counts):
            if num_changepoints != prev_val:
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
            prev_val = num_changepoints
        middle_of_plateau = longest_start + (longest_start - longest_start) // 2
        changepoints = np.array(changepoints_by_penalty[middle_of_plateau])
        return middle_of_plateau, changepoints

    def get_changepoints(self, signal, **kwargs):
        _, changepoints = self.get_penalty_and_changepoints(signal, **kwargs)
        return changepoints

    def get_penalty(self, signal, **kwargs):
        penalty, _ = self.get_penalty_and_changepoints(signal, **kwargs)
        return penalty

    def calc_raw_states(self, timestamps, signals, penalty, opt_model=None):
        """
        Calculate substates for signals (assumed to be long to a single parameter configuration).

        :returns: List of substates with duration and mean power: [(substate 1 duration, substate 1 power), ...]
        """

        # imported here as ruptures is only used for changepoint detection.
        # This way, dfatool can be used without having ruptures installed as
        # long as --pelt isn't active.
        import ruptures

        substate_data = list()

        raw_states_calc_args = list()
        for num_measurement, measurement in enumerate(signals):
            normed_signal = self.norm_signal(measurement)
            algo = ruptures.Pelt(
                model=self.model, jump=self.jump, min_size=self.min_dist
            ).fit(normed_signal)
            raw_states_calc_args.append((num_measurement, algo, penalty))

        raw_states_list = [None] * len(signals)
        with Pool() as pool:
            raw_states_res = pool.starmap(PELT_get_raw_states, raw_states_calc_args)

        substate_counts = list(map(lambda x: len(x[1]), raw_states_res))
        expected_substate_count = np.argmax(np.bincount(substate_counts))
        usable_measurements = list(
            filter(lambda x: len(x[1]) == expected_substate_count, raw_states_res)
        )
        logger.debug(
            f"    There are {expected_substate_count} substates (std = {np.std(substate_counts)}, {len(usable_measurements)}/{len(raw_states_res)} results are usable)"
        )

        for i in range(expected_substate_count):
            substate_data.append(
                {"duration": list(), "power": list(), "power_std": list()}
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

        return substate_counts, substate_data
