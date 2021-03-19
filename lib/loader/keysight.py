#!/usr/bin/env python3

import csv
import io
import numpy as np
import struct
import xml.etree.ElementTree as ET

from bisect import bisect_right


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


class DLogChannel:
    def __init__(self, desc_tuple):
        self.slot = desc_tuple[0]
        self.smu = desc_tuple[1]
        self.unit = desc_tuple[2]
        self.data = None

    def __repr__(self):
        return f"""<DLogChannel(slot={self.slot}, smu="{self.smu}", unit="{self.unit}", data={self.data})>"""


class DLog:
    def __init__(
        self,
        voltage: float,
        state_duration: int,
        with_traces=False,
        skip_duration=None,
        limit_duration=None,
    ):
        self.voltage = voltage
        self.state_duration = state_duration
        self.with_traces = with_traces
        self.skip_duration = skip_duration
        self.limit_duration = limit_duration
        self.errors = list()

        self.sync_min_duration = 0.7
        # TODO auto-detect
        self.sync_power = 10e-3

    def load_data(self, content):
        lines = []
        line = ""
        with io.BytesIO(content) as f:
            while line != "</dlog>\n":
                line = f.readline().decode()
                lines.append(line)
            xml_header = "".join(lines)
            raw_header = f.read(8)
            data_offset = f.tell()
            raw_data = f.read()

        xml_header = xml_header.replace("1ua>", "X1ua>")
        xml_header = xml_header.replace("2ua>", "X2ua>")
        dlog = ET.fromstring(xml_header)
        channels = []

        for channel in dlog.findall("channel"):
            channel_id = int(channel.get("id"))
            sense_curr = channel.find("sense_curr").text
            sense_volt = channel.find("sense_volt").text
            model = channel.find("ident").find("model").text
            if sense_volt == "1":
                channels.append((channel_id, model, "V"))
            if sense_curr == "1":
                channels.append((channel_id, model, "A"))

        num_channels = len(channels)

        self.channels = list(map(DLogChannel, channels))
        self.interval = float(dlog.find("frame").find("tint").text)
        self.sense_minmax = int(dlog.find("frame").find("sense_minmax").text)
        self.planned_duration = int(dlog.find("frame").find("time").text)
        self.observed_duration = self.interval * int(len(raw_data) / (4 * num_channels))

        if self.sense_minmax:
            raise RuntimeError(
                "DLog files with 'Log Min/Max' enabled are not supported yet"
            )

        self.timestamps = np.linspace(
            0, self.observed_duration, num=int(len(raw_data) / (4 * num_channels))
        )

        if (
            self.skip_duration is not None
            and self.observed_duration >= self.skip_duration
        ):
            start_offset = 0
            for i, ts in enumerate(self.timestamps):
                if ts >= self.skip_duration:
                    start_offset = i
                    break
            self.timestamps = self.timestamps[start_offset:]
            raw_data = raw_data[start_offset * 4 * num_channels :]

        if (
            self.limit_duration is not None
            and self.observed_duration > self.limit_duration
        ):
            stop_offset = len(self.timestamps) - 1
            for i, ts in enumerate(self.timestamps):
                if ts > self.limit_duration:
                    stop_offset = i
                    break
            self.timestamps = self.timestamps[:stop_offset]
            self.observed_duration = self.timestamps[-1]
            raw_data = raw_data[: stop_offset * 4 * num_channels]

        self.data = np.ndarray(
            shape=(num_channels, int(len(raw_data) / (4 * num_channels))),
            dtype=np.float32,
        )

        iterator = struct.iter_unpack(">f", raw_data)
        channel_offset = 0
        measurement_offset = 0
        for value in iterator:
            if value[0] < -1e6 or value[0] > 1e6:
                print(
                    f"Invalid data value {value[0]} at channel {channel_offset}, measurement {measurement_offset}. Replacing with 0."
                )
                self.data[channel_offset, measurement_offset] = 0
            else:
                self.data[channel_offset, measurement_offset] = value[0]
            if channel_offset + 1 == num_channels:
                channel_offset = 0
                measurement_offset += 1
            else:
                channel_offset += 1

        # An SMU has four slots
        self.slots = [dict(), dict(), dict(), dict()]

        for i, channel in enumerate(self.channels):
            channel.data = self.data[i]
            self.slots[channel.slot - 1][channel.unit] = channel

        assert "A" in self.slots[0]
        self.data = self.slots[0]["A"].data

    def observed_duration_equals_expectation(self):
        return int(self.observed_duration) == self.planned_duration

    # very similar to DataProcessor.getStatesdfatool
    def analyze_states(self, expected_trace, repeat_id):
        sync_start = None
        sync_timestamps = list()
        above_count = 0
        below_count = 0
        for i, timestamp in enumerate(self.timestamps):
            power = self.voltage * self.data[i]
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
