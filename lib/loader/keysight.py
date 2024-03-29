#!/usr/bin/env python3

import csv
import io
import numpy as np
import struct
import xml.etree.ElementTree as ET

from dfatool.loader.generic import ExternalTimerSync


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


class DLog(ExternalTimerSync):
    """Loader for DLog files generated by Keysight power analyzers."""

    def __init__(
        self,
        voltage: float,
        state_duration: int,
        with_traces=False,
        skip_duration=None,
        limit_duration=None,
    ):
        """
        Create a new DLog object

        :param voltage: Voltage in V
        :type voltage: float
        :param state_duration: Expected state duration in ms. Used to detect and ignore UART transmissions in captured energy data.
        :type state_duration: int
        :param with_traces: Provide traces and timestamps, default false
        :type with_traces: bool
        :param skip_duration: Ignore the first `skip_duration` seconds, default None (ignore nothing)
        :type skip_duration: float
        :param limit_duration: Ignore everything after `limit_duration` seconds, default none (ignore nothing)
        :type limit_duration: float
        """
        self.voltage = voltage
        self.state_duration = state_duration
        self.with_traces = with_traces
        self.skip_duration = skip_duration
        self.limit_duration = limit_duration
        self.errors = list()

        self.sync_min_duration = 0.7
        self.sync_max_duration = 1.3
        self.sync_min_low_count = 3
        self.sync_min_high_count = 3

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
        self.data = self.slots[0]["A"].data * self.voltage

    def observed_duration_equals_expectation(self):
        return int(self.observed_duration) == self.planned_duration
