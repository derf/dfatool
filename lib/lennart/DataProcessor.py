#!/usr/bin/env python3
import numpy as np
import logging
import os
import scipy
from bisect import bisect_left, bisect_right

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(
        self, sync_data, et_timestamps, et_power, hw_statechange_indexes=list()
    ):
        """
        Creates DataProcessor object.

        :param sync_data: input timestamps (SigrokResult)
        :param energy_data: List of EnergyTrace datapoints
        """
        self.raw_sync_timestamps = []
        # high-precision LA/Timer timestamps at synchronization events
        self.sync_timestamps = []
        # low-precision energytrace timestamps
        self.et_timestamps = et_timestamps
        # energytrace power values
        self.et_power_values = et_power
        self.hw_statechange_indexes = hw_statechange_indexes
        self.sync_data = sync_data
        self.start_offset = 0

        # TODO determine automatically based on minimum (or p1) power draw over measurement area + X
        # use 0.02 for HFXT runs
        self.power_sync_watt = 0.011
        self.power_sync_len = 0.7
        self.power_sync_max_outliers = 2

    def run(self):
        """
        Main Function to remove unwanted data, get synchronization points, add the offset and add drift.
        :return: None
        """

        # Remove bogus data before / after the measurement

        time_stamp_data = self.sync_data.timestamps
        for x in range(1, len(time_stamp_data)):
            if time_stamp_data[x] - time_stamp_data[x - 1] > 1.3:
                time_stamp_data = time_stamp_data[x:]
                break

        for x in reversed(range(1, len(time_stamp_data))):
            if time_stamp_data[x] - time_stamp_data[x - 1] > 1.3:
                time_stamp_data = time_stamp_data[:x]
                break

        # Each synchronization pulse consists of two LogicAnalyzer pulses, so four
        # entries in time_stamp_data (rising edge, falling edge, rising edge, falling edge).
        # If we have less then twelve entries, we observed no transitions and don't even have
        # valid synchronization data. In this case, we bail out.
        if len(time_stamp_data) < 12:
            raise RuntimeError(
                f"LogicAnalyzer sync data has length {len(time_stamp_data)}, expected >= 12"
            )

        self.raw_sync_timestamps = time_stamp_data

        # NEW
        datasync_timestamps = []
        sync_start = 0
        outliers = 0
        pre_outliers_ts = None
        # TODO only consider the first few and the last few seconds for sync points
        for i, timestamp in enumerate(self.et_timestamps):
            power = self.et_power_values[i]
            if power > 0:
                if power > self.power_sync_watt:
                    if sync_start is None:
                        sync_start = timestamp
                    outliers = 0
                else:
                    # Sync point over or outliers
                    if outliers == 0:
                        pre_outliers_ts = timestamp
                    outliers += 1
                    if outliers > self.power_sync_max_outliers:
                        if sync_start is not None:
                            if (pre_outliers_ts - sync_start) > self.power_sync_len:
                                datasync_timestamps.append(
                                    (sync_start, pre_outliers_ts)
                                )
                            sync_start = None

        if power > self.power_sync_watt:
            if (self.et_timestamps[-1] - sync_start) > self.power_sync_len:
                datasync_timestamps.append((sync_start, pre_outliers_ts))

        # time_stamp_data contains an entry for each level change on the Logic Analyzer input.
        # So, time_stamp_data[0] is the first low-to-high transition, time_stamp_data[2] the second, etc.
        # -> time_stamp_data[2] is the low-to-high transition indicating the end of the first sync pulse
        # -> time_stamp_data[-8] is the low-to-high transition indicating the start of the first after-measurement sync pulse

        start_timestamp = datasync_timestamps[0][1]
        start_offset = start_timestamp - time_stamp_data[2]

        end_timestamp = datasync_timestamps[-2][0]
        end_offset = end_timestamp - (time_stamp_data[-8] + start_offset)
        logger.debug(
            f"Measurement area: ET timestamp range [{start_timestamp}, {end_timestamp}]"
        )
        logger.debug(
            f"Measurement area: LA timestamp range [{time_stamp_data[2]}, {time_stamp_data[-8]}]"
        )
        logger.debug(f"Start/End offsets: {start_offset} / {end_offset}")

        if abs(end_offset) > 10:
            raise RuntimeError(
                f"synchronization end_offset == {end_offset}. It should be no more than a few seconds."
            )

        # adjust start offset
        with_offset = np.array(time_stamp_data) + start_offset
        logger.debug(
            f"Measurement area with offset: LA timestamp range [{with_offset[2]}, {with_offset[-8]}]"
        )

        # adjust stop offset (may be different from start offset due to drift caused by
        # random temperature fluctuations)
        with_drift = self.addDrift(
            with_offset, end_timestamp, end_offset, start_timestamp
        )
        logger.debug(
            f"Measurement area with drift: LA timestamp range [{with_drift[2]}, {with_drift[-8]}]"
        )

        self.sync_timestamps = with_drift

        # adjust intermediate timestamps. There is a small error between consecutive measurements,
        # again due to drift caused by random temperature fluctuation. The error increases with
        # increased distance from synchronization points: It is negligible at the start and end
        # of the measurement and may be quite high around the middle. That's just the bounds, though --
        # you may also have a low error in the middle and error peaks elsewhere.
        # As the start and stop timestamps have already been synchronized, we only adjust
        # actual transition timestamps here.
        if os.getenv("DFATOOL_COMPENSATE_DRIFT"):
            if len(self.hw_statechange_indexes):
                # measurement was performed with EnergyTrace++
                # (i.e., with cpu state annotations)
                with_drift_compensation = self.compensateDriftPlusplus(with_drift[4:-8])
            else:
                with_drift_compensation = self.compensateDrift(with_drift[4:-8])
            self.sync_timestamps[4:-8] = with_drift_compensation

    def addDrift(self, input_timestamps, end_timestamp, end_offset, start_timestamp):
        """
        Add drift to datapoints

        :param input_timestamps: List of timestamps (float list)
        :param end_timestamp: Timestamp of first EnergyTrace datapoint at the second-to-last sync point
        :param end_offset: the time between end_timestamp and the timestamp of synchronisation signal
        :param start_timestamp: Timestamp of last EnergyTrace datapoint at the first sync point
        :return: List of modified timestamps (float list)
        """
        endFactor = 1 + (end_offset / ((end_timestamp - end_offset) - start_timestamp))
        # endFactor assumes that the end of the first sync pulse is at timestamp 0.
        # Then, timestamps with drift := timestamps * endFactor.
        # As this is not the case (the first sync pulse ends at start_timestamp > 0), we shift the data by first
        # removing start_timestamp, then multiplying with endFactor, and then re-adding the start_timestamp.
        sync_timestamps_with_drift = (
            input_timestamps - start_timestamp
        ) * endFactor + start_timestamp
        return sync_timestamps_with_drift

    def compensateDriftPlusplus(self, sync_timestamps):
        """Use hardware state changes reported by EnergyTrace++ to determine transition timestamps."""
        expected_transition_start_timestamps = sync_timestamps[::2]
        compensated_timestamps = list()
        drift = 0
        for i, expected_start_ts in enumerate(expected_transition_start_timestamps):
            expected_end_ts = sync_timestamps[i * 2 + 1]
            et_timestamps_start = bisect_left(
                self.et_timestamps, expected_start_ts - 5e-3
            )
            et_timestamps_end = bisect_right(
                self.et_timestamps, expected_start_ts + 5e-3
            )

            candidate_indexes = list()
            for index in self.hw_statechange_indexes:
                if et_timestamps_start <= index <= et_timestamps_end:
                    candidate_indexes.append(index)

            if len(candidate_indexes) == 2:
                drift = self.et_timestamps[candidate_indexes[0]] - expected_start_ts

            compensated_timestamps.append(expected_start_ts + drift)
            compensated_timestamps.append(expected_end_ts + drift)

        return compensated_timestamps

    def compensateDrift(self, sync_timestamps):
        """Use ruptures (e.g. Pelt, Dynp) to determine transition timestamps."""
        from dfatool.pelt import PELT

        # TODO die Anzahl Changepoints ist a priori bekannt, es könnte mit ruptures.Dynp statt ruptures.Pelt besser funktionieren.
        # Vielleicht sollte man auch "rbf" statt "l1" nutzen.
        # "rbf" und "l2" scheinen ähnlich gut zu funktionieren, l2 ist schneller.
        pelt = PELT(with_multiprocessing=False)
        expected_transition_start_timestamps = sync_timestamps[::2]
        transition_start_candidate_weights = list()
        compensated_timestamps = list()
        drift = 0

        for i, expected_start_ts in enumerate(expected_transition_start_timestamps):
            # assumption: maximum deviation between expected and actual timestamps is 5ms.
            # We use ±10ms to have some contetx for PELT
            et_timestamps_start = bisect_left(
                self.et_timestamps, expected_start_ts - 10e-3
            )
            et_timestamps_end = bisect_right(
                self.et_timestamps, expected_start_ts + 10e-3
            )
            timestamps = self.et_timestamps[et_timestamps_start : et_timestamps_end + 1]
            energy_data = self.et_power_values[
                et_timestamps_start : et_timestamps_end + 1
            ]
            candidate_weight = dict()
            if os.getenv("DFATOOL_DRIFT_COMPENSATION_PENALTY"):
                penalties = (int(os.getenv("DFATOOL_DRIFT_COMPENSATION_PENALTY")),)
            else:
                penalties = (1, 2, 5, 10, 15, 20)
            for penalty in penalties:
                for changepoint in pelt.get_changepoints(energy_data, penalty=penalty):
                    if changepoint in candidate_weight:
                        candidate_weight[changepoint] += 1
                    else:
                        candidate_weight[changepoint] = 1

            transition_start_candidate_weights.append(
                list(
                    map(
                        lambda k: (
                            timestamps[k],
                            timestamps[k] - expected_start_ts,
                            candidate_weight[k],
                        ),
                        candidate_weight.keys(),
                    )
                )
            )

        # Algorithm: Obtain the shortest path in a layered graph made up from
        # transition candidates. Each node represents a transition candidate timestamp, and each layer represents a transition.
        # Each node in layer i contains a directed edge to each node in layer i+1.
        # The edge weight is the drift delta between the two nodes. So, if,
        # node X (transition i, candidate a) has a drift of 5, and node Y
        # (transition i+1, candidate b) has a drift of -2, the weight is 7.
        # The first and last layer of the graph consists of a single node
        # with a drift of 0, representing the start / end synchronization pulse, respectively.

        prev_nodes = [0]
        prev_drifts = [0]
        edge_srcs = list()
        edge_dsts = list()
        csr_weights = list()
        node_drifts = list()

        nodes_by_transition_index = dict()
        transition_by_node = dict()

        for transition_index, candidates in enumerate(
            transition_start_candidate_weights
        ):
            new_nodes = list()
            new_drifts = list()
            i_offset = prev_nodes[-1] + 1
            nodes_by_transition_index[transition_index] = list()
            for new_node_i, (_, new_drift, _) in enumerate(candidates):
                new_node = new_node_i + i_offset
                nodes_by_transition_index[transition_index].append(new_node)
                transition_by_node[new_node] = transition_index
                new_nodes.append(new_node)
                new_drifts.append(new_drift)
                node_drifts.append(new_drift)
                for prev_node_i, prev_node in enumerate(prev_nodes):
                    prev_drift = prev_drifts[prev_node_i]

                    edge_srcs.append(prev_node)
                    edge_dsts.append(new_node)

                    delta_drift = np.abs(prev_drift - new_drift)
                    csr_weights.append(delta_drift)

            prev_nodes = new_nodes
            prev_drifts = new_drifts

        # add an end node for shortest path search
        # (end node == final sync, so drift == 0)
        new_node = prev_nodes[-1] + 1
        for prev_node_i, prev_node in enumerate(prev_nodes):
            prev_drift = prev_drifts[prev_node_i]
            edge_srcs.append(prev_node)
            edge_dsts.append(new_node)
            csr_weights.append(np.abs(prev_drift))

        # Add "skip" edges spanning from transition i to transition i+2
        # and from transition i to transition i+3.
        # These avoid synchronization errors caused by transitions wich are
        # not found by changepiont detection, as long as they are sufficiently rare.
        for transition_index, candidates in enumerate(
            transition_start_candidate_weights
        ):
            if transition_index < 2:
                continue
            for from_i, (_, from_drift, _) in enumerate(
                transition_start_candidate_weights[transition_index - 2]
            ):
                for to_i, (_, to_drift, _) in enumerate(candidates):
                    # Penalize shortcut by the duration of one sample
                    # (~270 us)
                    edge_srcs.append(
                        nodes_by_transition_index[transition_index - 2][from_i]
                    )
                    edge_dsts.append(nodes_by_transition_index[transition_index][to_i])
                    csr_weights.append(np.abs(from_drift - to_drift) + 270e-6)
            if transition_index < 3:
                continue
            for from_i, (_, from_drift, _) in enumerate(
                transition_start_candidate_weights[transition_index - 3]
            ):
                for to_i, (_, to_drift, _) in enumerate(candidates):
                    # Penalize shortcut by the duration of one sample
                    # (~270 us)
                    edge_srcs.append(
                        nodes_by_transition_index[transition_index - 3][from_i]
                    )
                    edge_dsts.append(nodes_by_transition_index[transition_index][to_i])
                    csr_weights.append(np.abs(from_drift - to_drift) + 2 * 270e-6)

        sm = scipy.sparse.csr_matrix(
            (csr_weights, (edge_srcs, edge_dsts)), shape=(new_node + 1, new_node + 1)
        )
        dm, predecessors = scipy.sparse.csgraph.shortest_path(
            sm, return_predecessors=True, indices=0
        )

        nodes = list()
        pred = predecessors[-1]
        while pred > 0:
            nodes.append(pred)
            pred = predecessors[pred]

        nodes = list(reversed(nodes))

        # first and graph nodes are not included in "nodes" as they represent
        # the start/stop sync pulse (and not a transition with sync candidates)

        prev_transition = -1
        for i, node in enumerate(nodes):
            transition = transition_by_node[node]
            drift = node_drifts[node]

            if transition - prev_transition >= 2:
                # previous transition was skipped due to lack of detected changepoints
                prev_drift = node_drifts[nodes[i - 1]]
                mean_drift = np.mean([prev_drift, drift])
                expected_start_ts = (
                    sync_timestamps[(prev_transition + 1) * 2] + mean_drift
                )
                expected_end_ts = (
                    sync_timestamps[(prev_transition + 1) * 2 + 1] + mean_drift
                )
                compensated_timestamps.append(expected_start_ts)
                compensated_timestamps.append(expected_end_ts)
            if transition - prev_transition >= 3:
                # previous transition was skipped due to lack of detected changepoints
                expected_start_ts = (
                    sync_timestamps[(prev_transition + 2) * 2] + mean_drift
                )
                expected_end_ts = (
                    sync_timestamps[(prev_transition + 2) * 2 + 1] + mean_drift
                )
                compensated_timestamps.append(expected_start_ts)
                compensated_timestamps.append(expected_end_ts)

            expected_start_ts = sync_timestamps[transition * 2] + drift
            expected_end_ts = sync_timestamps[transition * 2 + 1] + drift
            compensated_timestamps.append(expected_start_ts)
            compensated_timestamps.append(expected_end_ts)
            prev_transition = transition

        if os.getenv("DFATOOL_EXPORT_DRIFT_COMPENSATION"):
            import json
            from dfatool.utils import NpEncoder

            with open(os.getenv("DFATOOL_EXPORT_DRIFT_COMPENSATION"), "w") as f:
                json.dump(
                    [
                        expected_transition_start_timestamps,
                        transition_start_candidate_weights,
                    ],
                    f,
                    cls=NpEncoder,
                )

        return compensated_timestamps

    def export_sync(self):
        # [1st trans start, 1st trans stop, 2nd trans start, 2nd trans stop, ...]
        sync_timestamps = list()

        for i in range(4, len(self.sync_timestamps) - 8, 2):
            sync_timestamps.append(
                (self.sync_timestamps[i], self.sync_timestamps[i + 1])
            )

        # EnergyTrace timestamps
        timestamps = self.et_timestamps

        # EnergyTrace power values
        power = self.et_power_values

        return {"sync": sync_timestamps, "timestamps": timestamps, "power": power}

    def plot(self, annotateData=None):
        """
        Plots the power usage and the timestamps by logic analyzer

        :param annotateData: List of Strings with labels, only needed if annotated plots are wished
        :return: None
        """

        def calculateRectangleCurve(timestamps, min_value=0, max_value=0.160):
            import numpy as np

            data = []
            for ts in timestamps:
                data.append(ts)
                data.append(ts)

            a = np.empty((len(data),))
            a[0::4] = min_value
            a[1::4] = max_value
            a[2::4] = max_value
            a[3::4] = min_value
            return data, a  # plotting by columns

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        if annotateData:
            annot = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(20, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"),
            )
            annot.set_visible(True)

        rectCurve_with_drift = calculateRectangleCurve(
            self.sync_timestamps, max_value=max(self.et_power_values)
        )

        plt.plot(self.et_timestamps, self.et_power_values, label="Leistung")
        plt.plot(self.et_timestamps, np.gradient(self.et_power_values), label="dP/dt")

        plt.plot(
            rectCurve_with_drift[0],
            rectCurve_with_drift[1],
            "-g",
            label="Synchronisationsignale mit Driftfaktor",
        )

        plt.xlabel("Zeit von EnergyTrace [s]")
        plt.ylabel("Leistung [W]")
        leg = plt.legend()

        def getDataText(x):
            # print(x)
            dl = len(annotateData)
            for i, xt in enumerate(self.sync_timestamps):
                if xt > x and i >= 4 and i - 5 < dl:
                    return f"SoT: {annotateData[i - 5]}"

        def update_annot(x, y, name):
            annot.xy = (x, y)
            text = name

            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            if event.xdata and event.ydata:
                annot.set_visible(False)
                update_annot(event.xdata, event.ydata, getDataText(event.xdata))
                annot.set_visible(True)
                fig.canvas.draw_idle()

        if annotateData:
            fig.canvas.mpl_connect("motion_notify_event", hover)

        plt.show()

    def getStatesdfatool(self, state_sleep, with_traces=False, algorithm=False):
        """
        Calculates the length and energy usage of the states

        :param state_sleep: Length in seconds of one state, needed for cutting out the UART Sending cycle
        :param algorithm: possible usage of accuracy algorithm / not implemented yet
        :returns: returns list of states and transitions, starting with a transition and ending with astate
            Each element is a dict containing:
            * `isa`: 'state' or 'transition'
            * `W_mean`: Mittelwert der Leistungsaufnahme
            * `W_std`: Standardabweichung der Leistungsaufnahme
            * `s`: Dauer
        """
        if algorithm:
            raise NotImplementedError
        end_transition_ts = None
        timestamps_sync_start = 0
        energy_trace_new = list()

        # sync_timestamps[3] is the start of the first (UNINITIALIZED) state (and the end of the benchmark-start sync pulse)
        # sync_timestamps[-8] is the end of the final state and the corresponding UART dump (and the start of the benchmark-end sync pulses)
        self.trigger_high_precision_timestamps = self.sync_timestamps[3:-7]

        self.trigger_edges = list()
        for ts in self.trigger_high_precision_timestamps:
            # Let ts be the trigger timestamp corresponding to the end of a transition.
            # We are looking for an index i such that et_timestamps[i-1] <= ts < et_timestamps[i].
            # Then, et_power_values[i] (the mean power in the interval et_timestamps[i-1] .. et_timestamps[i]) is affected by the transition and
            # et_power_values[i+1] is not affected by it.
            #
            # bisect_right does just what we need; bisect_left would correspond to et_timestamps[i-1] < ts <= et_timestamps[i].
            # Not that this is a moot point in practice, as ts ≠ et_timestamps[j] for almost all j. Also, the resolution of
            # et_timestamps is several decades lower than the resolution of trigger_high_precision_timestamps.
            self.trigger_edges.append(bisect_right(self.et_timestamps, ts))

        # Loop over transitions. We start at the end of the first transition and handle the transition and the following state.
        # We then proceed to the end of the second transition, etc.
        for i in range(2, len(self.trigger_high_precision_timestamps), 2):
            prev_state_start_index = self.trigger_edges[i - 2]
            prev_state_stop_index = self.trigger_edges[i - 1]
            transition_start_index = self.trigger_edges[i - 1]
            transition_stop_index = self.trigger_edges[i]
            state_start_index = self.trigger_edges[i]
            state_stop_index = self.trigger_edges[i + 1]

            # If a transition takes less time than the energytrace measurement interval, its start and stop index may be the same.
            # In this case, et_power_values[transition_start_index] is the only data point affected by the transition.
            # We use the et_power_values slice [transition_start_index, transition_stop_index) to determine the mean power, so we need
            # to increment transition_stop_index by 1 to end at et_power_values[transition_start_index]
            # (as et_power_values[transition_start_index : transition_start_index+1 ] == [et_power_values[transition_start_index])
            if transition_stop_index == transition_start_index:
                transition_stop_index += 1

            prev_state_duration = (
                self.trigger_high_precision_timestamps[i + 1]
                - self.trigger_high_precision_timestamps[i]
            )
            transition_duration = (
                self.trigger_high_precision_timestamps[i]
                - self.trigger_high_precision_timestamps[i - 1]
            )
            state_duration = (
                self.trigger_high_precision_timestamps[i + 1]
                - self.trigger_high_precision_timestamps[i]
            )

            # some states are followed by a UART dump of log data. This causes an increase in CPU energy
            # consumption and is not part of the peripheral behaviour, so it should not be part of the benchmark results.
            # If a case is followed by a UART dump, its duration is longer than the sleep duration between two transitions.
            # In this case, we re-calculate the stop index, and calculate the state duration from coarse energytrace data
            # instead of high-precision sync data
            if (
                self.et_timestamps[prev_state_stop_index]
                - self.et_timestamps[prev_state_start_index]
                > state_sleep
            ):
                prev_state_stop_index = bisect_right(
                    self.et_timestamps,
                    self.et_timestamps[prev_state_start_index] + state_sleep,
                )
                prev_state_duration = (
                    self.et_timestamps[prev_state_stop_index]
                    - self.et_timestamps[prev_state_start_index]
                )

            if (
                self.et_timestamps[state_stop_index]
                - self.et_timestamps[state_start_index]
                > state_sleep
            ):
                state_stop_index = bisect_right(
                    self.et_timestamps,
                    self.et_timestamps[state_start_index] + state_sleep,
                )
                state_duration = (
                    self.et_timestamps[state_stop_index]
                    - self.et_timestamps[state_start_index]
                )

            prev_state_power = self.et_power_values[
                prev_state_start_index:prev_state_stop_index
            ]

            transition_timestamps = self.et_timestamps[
                transition_start_index:transition_stop_index
            ]
            transition_power = self.et_power_values[
                transition_start_index:transition_stop_index
            ]

            state_timestamps = self.et_timestamps[state_start_index:state_stop_index]
            state_power = self.et_power_values[state_start_index:state_stop_index]

            transition = {
                "isa": "transition",
                "W_mean": np.mean(transition_power),
                "W_std": np.std(transition_power),
                "s": transition_duration,
                "count_dp": len(transition_power),
            }
            if with_traces:
                transition["plot"] = (
                    transition_timestamps - transition_timestamps[0],
                    transition_power,
                )

            state = {
                "isa": "state",
                "W_mean": np.mean(state_power),
                "W_std": np.std(state_power),
                "s": state_duration,
            }
            if with_traces:
                state["plot"] = (state_timestamps - state_timestamps[0], state_power)

            transition["W_mean_delta_prev"] = transition["W_mean"] - np.mean(
                prev_state_power
            )
            transition["W_mean_delta_next"] = transition["W_mean"] - state["W_mean"]

            energy_trace_new.append(transition)
            energy_trace_new.append(state)

        return energy_trace_new
