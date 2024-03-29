#!/usr/bin/env python3

import json
import logging
import numpy as np
import os
from bisect import bisect_right
from dfatool.utils import NpEncoder

logger = logging.getLogger(__name__)


class ExternalTimerSync:
    def __init__(self):
        raise NotImplementedError("must be implemented in sub-class")

    def assert_sync_areas(self, sync_areas):
        # may be implemented in sub-class
        pass

    def compensate_drift(self, data, timestamps, event_timestamps, offline_index=None):
        # adjust intermediate timestamps. There is a small error between consecutive measurements,
        # again due to drift caused by random temperature fluctuation. The error increases with
        # increased distance from synchronization points: It is negligible at the start and end
        # of the measurement and may be quite high around the middle. That's just the bounds, though --
        # you may also have a low error in the middle and error peaks elsewhere.
        # As the start and stop timestamps have already been synchronized, we only adjust
        # actual transition timestamps here.
        if os.getenv("DFATOOL_COMPENSATE_DRIFT"):
            import dfatool.drift

            if len(self.hw_statechange_indexes):
                # measurement was performed with EnergyTrace++
                # (i.e., with cpu state annotations)
                return dfatool.drift.compensate_etplusplus(
                    data,
                    timestamps,
                    event_timestamps,
                    self.hw_statechange_indexes,
                    offline_index=offline_index,
                )

            return dfatool.drift.compensate(
                data, timestamps, event_timestamps, offline_index=offline_index
            )
        return event_timestamps

    # very similar to DataProcessor.getStatesdfatool
    # requires:
    # * self.data (e.g. power readings)
    # * self.timestamps (timstamps in seconds)
    # * self.sync_min_high_count, self.sync_min_low_count: outlier handling in synchronization pulse detection
    # * self.sync_power, self.sync_min_duration, self.sync_max_duration: synchronization pulse parameters. one pulse before the measurement, two pulses afterwards
    # expected_trace must contain online timestamps
    # TODO automatically determine sync_power if it is None
    def analyze_states(self, expected_trace, repeat_id, online_timestamps=None):
        """
        :param online_timestamps: must start at 0, if set
        """
        sync_start = None
        sync_timestamps = list()
        high_count = 0
        low_count = 0
        high_ts = None
        low_ts = None
        for i, timestamp in enumerate(self.timestamps):
            power = self.data[i]
            if power > self.sync_power:
                if high_count == 0:
                    high_ts = timestamp
                high_count += 1
                low_count = 0
            else:
                if low_count == 0:
                    low_ts = timestamp
                high_count = 0
                low_count += 1

            if high_count >= self.sync_min_high_count and sync_start is None:
                sync_start = high_ts
            elif low_count >= self.sync_min_low_count and sync_start is not None:
                if (
                    self.sync_min_duration
                    < low_ts - sync_start
                    < self.sync_max_duration
                ):
                    sync_end = low_ts
                    sync_timestamps.append((sync_start, sync_end))
                sync_start = None

        if len(sync_timestamps) != 3:
            self.errors.append(
                f"Found {len(sync_timestamps)} synchronization pulses, expected three."
            )
            self.errors.append(f"Synchronization pulses == {sync_timestamps}")
            return list()

        self.assert_sync_areas(sync_timestamps)

        start_ts = sync_timestamps[0][1]
        end_ts = sync_timestamps[1][0]

        if online_timestamps is None:
            # start and end of first state
            online_timestamps = [0, expected_trace[0]["start_offset"][repeat_id]]

            # remaining events from the end of the first transition (start of second state) to the end of the last observed state
            try:
                for trace in expected_trace:
                    for word in trace["trace"]:
                        online_timestamps.append(
                            online_timestamps[-1]
                            + word["online_aggregates"]["duration"][repeat_id]
                        )
            except IndexError:
                self.errors.append(
                    f"""offline_index {repeat_id} missing in trace {trace["id"]}"""
                )
                return list()

            online_timestamps = np.array(online_timestamps) * 1e-6
        else:
            online_timestamps = np.array(online_timestamps)

        online_timestamps = (
            online_timestamps
            * ((end_ts - start_ts) / (online_timestamps[-1] - online_timestamps[0]))
            + start_ts
        )

        # drift compensation works on transition boundaries. Exclude start of first state and end of last state.
        # Those are defined to have zero drift anyways.
        online_timestamps[1:-1] = self.compensate_drift(
            self.data, self.timestamps, online_timestamps[1:-1], repeat_id
        )

        trigger_edges = list()
        for ts in online_timestamps:
            trigger_edges.append(bisect_right(self.timestamps, ts))

        energy_trace = list()

        for i in range(2, len(online_timestamps), 2):
            prev_state_start_index = trigger_edges[i - 2]
            prev_state_stop_index = trigger_edges[i - 1]
            transition_start_index = trigger_edges[i - 1]
            transition_stop_index = trigger_edges[i]
            state_start_index = trigger_edges[i]
            state_stop_index = trigger_edges[i + 1]

            # If a transition takes less time than the measurement interval, its start and stop index may be the same.
            # In this case, self.data[transition_start_index] is the only data point affected by the transition.
            # We use the self.data slice [transition_start_index, transition_stop_index) to determine the mean power, so we need
            # to increment transition_stop_index by 1 to end at self.data[transition_start_index]
            # (self.data[transition_start_index : transition_start_index+1 ] == [self.data[transition_start_index])
            if transition_stop_index == transition_start_index:
                transition_stop_index += 1

            prev_state_duration = online_timestamps[i + 1] - online_timestamps[i]
            transition_duration = online_timestamps[i] - online_timestamps[i - 1]
            state_duration = online_timestamps[i + 1] - online_timestamps[i]

            # some states are followed by a UART dump of log data. This causes an increase in CPU energy
            # consumption and is not part of the peripheral behaviour, so it should not be part of the benchmark results.
            # If a case is followed by a UART dump, its duration is longer than the sleep duration between two transitions.
            # In this case, we re-calculate the stop index, and calculate the state duration from coarse energytrace data
            # instead of high-precision sync data
            if (
                self.timestamps[prev_state_stop_index]
                - self.timestamps[prev_state_start_index]
                > self.state_duration
            ):
                prev_state_stop_index = bisect_right(
                    self.timestamps,
                    self.timestamps[prev_state_start_index] + self.state_duration,
                )
                prev_state_duration = (
                    self.timestamps[prev_state_stop_index]
                    - self.timestamps[prev_state_start_index]
                )

            if (
                self.timestamps[state_stop_index] - self.timestamps[state_start_index]
                > self.state_duration
            ):
                state_stop_index = bisect_right(
                    self.timestamps,
                    self.timestamps[state_start_index] + self.state_duration,
                )
                state_duration = (
                    self.timestamps[state_stop_index]
                    - self.timestamps[state_start_index]
                )

            prev_state_power = self.data[prev_state_start_index:prev_state_stop_index]

            transition_timestamps = self.timestamps[
                transition_start_index:transition_stop_index
            ]
            transition_power = self.data[transition_start_index:transition_stop_index]

            state_timestamps = self.timestamps[state_start_index:state_stop_index]
            state_power = self.data[state_start_index:state_stop_index]

            transition = {
                "isa": "transition",
                "W_mean": np.mean(transition_power),
                "W_std": np.std(transition_power),
                "s": transition_duration,
            }
            if self.with_traces:
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
            if self.with_traces:
                state["plot"] = (state_timestamps - state_timestamps[0], state_power)

            transition["W_mean_delta_prev"] = transition["W_mean"] - np.mean(
                prev_state_power
            )
            transition["W_mean_delta_next"] = transition["W_mean"] - state["W_mean"]

            energy_trace.append(transition)
            energy_trace.append(state)

        if os.getenv("DFATOOL_PLOT_SYNC") is not None and repeat_id == int(
            os.getenv("DFATOOL_PLOT_SYNC")
        ):
            self.plot_sync(online_timestamps)  # <- plot traces with sync annotatons
            # self.plot_sync(names) # <- plot annotated traces (with state/transition names)
        # TODO LASYNC -> SYNC
        if os.getenv("DFATOOL_EXPORT_LASYNC") is not None:
            filename = os.getenv("DFATOOL_EXPORT_LASYNC") + f"_{repeat_id}.json"
            with open(filename, "w") as f:
                json.dump(self._export_sync(online_timestamps), f, cls=NpEncoder)
            logger.info("Exported sync timestamps to {filename}")

        return energy_trace

    def _export_sync(self, online_timestamps):
        # [(1st trans start, 1st trans stop), (2nd trans start, 2nd trans stop), ...]
        sync_timestamps = list()

        for i in range(1, len(online_timestamps) - 1, 2):
            sync_timestamps.append((online_timestamps[i], online_timestamps[i + 1]))

        # input timestamps
        timestamps = self.timestamps

        # input data, e.g. power
        data = self.data

        # TODO "power" -> "data"
        return {"sync": sync_timestamps, "timestamps": timestamps, "power": data}

    def plot_sync(self, event_timestamps, annotateData=None):
        """
        Plots the power usage and the timestamps by logic analyzer

        :param annotateData: List of Strings with labels, only needed if annotated plots are wished
        :return: None
        """

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

        plt.plot(self.timestamps, self.data, label="P")
        plt.plot(self.timestamps, np.gradient(self.data), label="dP/dt")
        plt.vlines(
            event_timestamps, 0, max(self.data), colors=["green"], label="Events"
        )

        plt.xlabel("Time [s]")
        plt.ylabel("Power [W]")
        leg = plt.legend()

        def getDataText(x):
            # print(x)
            dl = len(annotateData)
            for i, xt in enumerate(event_timestamps):
                if xt > x and 0 <= i < dl:
                    return f"SoT: {annotateData[i]}"

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
