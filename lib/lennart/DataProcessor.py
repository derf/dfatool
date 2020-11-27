#!/usr/bin/env python3
import numpy as np
import logging
from bisect import bisect_right

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, sync_data, energy_data):
        """
        Creates DataProcessor object.

        :param sync_data: input timestamps (SigrokResult)
        :param energy_data: List of EnergyTrace datapoints
        """
        self.raw_sync_timestamps = []
        # high-precision LA/Timer timestamps at synchronization events
        self.sync_timestamps = []
        # low-precision energytrace timestamps
        self.et_timestamps = []
        # energytrace power values
        self.et_power_values = []
        self.sync_data = sync_data
        self.energy_data = energy_data
        self.start_offset = 0

        self.power_sync_watt = 0.011
        self.power_sync_len = 0.7
        self.power_sync_max_outliers = 2

    def run(self):
        """
        Main Function to remove unwanted data, get synchronization points, add the offset and add drift.
        :return: None
        """
        # remove Dirty Data from previously running program (happens if logic Analyzer Measurement starts earlier than
        # the HW Reset from energytrace)
        use_data_after_index = 0
        for x in range(1, len(self.sync_data.timestamps)):
            if self.sync_data.timestamps[x] - self.sync_data.timestamps[x - 1] > 1.3:
                use_data_after_index = x
                break

        time_stamp_data = self.sync_data.timestamps[use_data_after_index:]

        # Each synchronization pulse consists of two LogicAnalyzer pulses, so four
        # entries in time_stamp_data (rising edge, falling edge, rising edge, falling edge).
        # If we have less then twelve entries, we observed no transitions and don't even have
        # valid synchronization data. In this case, we bail out.
        if len(time_stamp_data) < 12:
            raise RuntimeError(
                f"LogicAnalyzer sync data has length {len(time_stamp_data)}, expected >= 12"
            )

        last_data = [0, 0, 0, 0]

        self.raw_sync_timestamps = time_stamp_data

        # NEW
        datasync_timestamps = []
        sync_start = 0
        outliers = 0
        pre_outliers_ts = None
        # TODO only consider the first few and the last few seconds for sync points
        for i, energytrace_dataset in enumerate(self.energy_data):
            usedtime = energytrace_dataset[0] - last_data[0]  # in microseconds
            timestamp = energytrace_dataset[0]
            usedenergy = energytrace_dataset[3] - last_data[3]
            power = usedenergy / usedtime * 1e-3  # in watts
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
                            if (
                                pre_outliers_ts - sync_start
                            ) / 1_000_000 > self.power_sync_len:
                                datasync_timestamps.append(
                                    (
                                        sync_start / 1_000_000,
                                        pre_outliers_ts / 1_000_000,
                                    )
                                )
                            sync_start = None

                last_data = energytrace_dataset

            self.et_timestamps.append(timestamp / 1_000_000)
            self.et_power_values.append(power)

        if power > self.power_sync_watt:
            if (self.energy_data[-1][0] - sync_start) / 1_000_000 > self.power_sync_len:
                datasync_timestamps.append(
                    (sync_start / 1_000_000, pre_outliers_ts / 1_000_000)
                )

        # print(datasync_timestamps)

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

        with_offset = np.array(time_stamp_data) + start_offset
        logger.debug(
            f"Measurement area with offset: LA timestamp range [{with_offset[2]}, {with_offset[-8]}]"
        )

        with_drift = self.addDrift(
            with_offset, end_timestamp, end_offset, start_timestamp
        )
        logger.debug(
            f"Measurement area with drift: LA timestamp range [{with_drift[2]}, {with_drift[-8]}]"
        )

        self.sync_timestamps = with_drift

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
