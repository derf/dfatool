#!/usr/bin/env python3

import numpy as np
from bisect import bisect_right


class ExternalTimerSync:
    def __init__(self):
        raise NotImplementedError("must be implemented in sub-class")

    # very similar to DataProcessor.getStatesdfatool
    # requires:
    # * self.data (e.g. power readings)
    # * self.timestamps (timstamps in seconds)
    # * self.sync_power, self.sync_min_duration: synchronization pulse parameters. one pulse before the measurement, two pulses afterwards
    # expected_trace must contain online timestamps
    def analyze_states(self, expected_trace, repeat_id):
        sync_start = None
        sync_timestamps = list()
        above_count = 0
        below_count = 0
        for i, timestamp in enumerate(self.timestamps):
            power = self.data[i]
            if power > self.sync_power:
                above_count += 1
                below_count = 0
            else:
                above_count = 0
                below_count += 1

            if above_count > 2 and sync_start is None:
                sync_start = timestamp
            elif below_count > 2 and sync_start is not None:
                if timestamp - sync_start > self.sync_min_duration:
                    sync_end = timestamp
                    sync_timestamps.append((sync_start, sync_end))
                sync_start = None
        print(sync_timestamps)

        if len(sync_timestamps) != 3:
            self.errors.append(
                f"Found {len(sync_timestamps)} synchronization pulses, expected three."
            )
            self.errors.append(f"Synchronization pulses == {sync_timestamps}")
            return list()

        start_ts = sync_timestamps[0][1]
        end_ts = sync_timestamps[1][0]

        # start and end of first state
        online_timestamps = [0, expected_trace[0]["start_offset"][repeat_id]]

        # remaining events from the end of the first transition (start of second state) to the end of the last observed state
        for trace in expected_trace:
            for word in trace["trace"]:
                online_timestamps.append(
                    online_timestamps[-1]
                    + word["online_aggregates"]["duration"][repeat_id]
                )

        online_timestamps = np.array(online_timestamps) * 1e-6
        online_timestamps = (
            online_timestamps * ((end_ts - start_ts) / online_timestamps[-1]) + start_ts
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

        return energy_trace
