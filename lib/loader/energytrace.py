#!/usr/bin/env python3

import json
import logging
import numpy as np
import os
import re

from dfatool.loader.generic import ExternalTimerSync
from dfatool.utils import NpEncoder, soft_cast_int

logger = logging.getLogger(__name__)

try:
    from dfatool.pubcode import Code128
    import zbar

    zbar_available = True
except ImportError:
    zbar_available = False


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


class EnergyTrace:
    @staticmethod
    def add_offline_aggregates(online_traces, offline_trace, repeat_id):
        # Edits online_traces[*]['trace'][*]['offline']
        # and online_traces[*]['trace'][*]['offline_aggregates'] in place
        # (appends data from offline_trace)
        online_datapoints = []
        for run_idx, run in enumerate(online_traces):
            for trace_part_idx in range(len(run["trace"])):
                online_datapoints.append((run_idx, trace_part_idx))
        for offline_idx, (online_run_idx, online_trace_part_idx) in enumerate(
            online_datapoints
        ):
            try:
                offline_trace_part = offline_trace[offline_idx]
            except IndexError:
                logger.error(f"  offline energy_trace data is shorter than online data")
                logger.error(f"  len(online_datapoints) == {len(online_datapoints)}")
                logger.error(f"  len(energy_trace) == {len(offline_trace)}")
                raise
            online_trace_part = online_traces[online_run_idx]["trace"][
                online_trace_part_idx
            ]

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
                            online_trace_part["parameter"][paramkey][repeat_id]
                        )
                    )
                else:
                    paramvalues.append(
                        soft_cast_int(online_trace_part["parameter"][paramkey])
                    )

            # NB: Unscheduled transitions do not have an 'args' field set.
            # However, they should only be caused by interrupts, and
            # interrupts don't have args anyways.
            if "args" in online_trace_part:
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
                    online_trace_part["offline_support"] = [
                        "power_traces",
                        "timestamps",
                    ]
                    online_trace_part["offline_aggregates"]["power_traces"] = list()
                    online_trace_part["offline_aggregates"]["timestamps"] = list()
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
                offline_aggregates["timestamps"].append(offline_trace_part["plot"][0])

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
            else:
                # TODO this really isn't nice, as W_mean_delta_prev of other setup
                # transitions is probably different. The best solution might be
                # ignoring the first transition when handling delta_prev values
                energy_trace[-1]["W_mean_delta_prev"] = energy_trace[-1]["W_mean"]

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


class EnergyTraceWithLogicAnalyzer(ExternalTimerSync):
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

        self.sync_min_duration = 0.7
        self.sync_min_low_count = 3
        self.sync_min_high_count = 1
        self.sync_power = 0.011

    def load_data(self, log_data):
        la_data = json.loads(log_data[0])
        self.sync_data = la_data["timestamps"]

        (
            self.interval_start_timestamp,
            self.interval_duration,
            self.interval_power,
            self.sample_rate,
            self.hw_statechange_indexes,
        ) = _load_energytrace(log_data[1])

        self.timestamps = self.interval_start_timestamp
        self.data = self.interval_power

        for x in range(1, len(self.sync_data)):
            if self.sync_data[x] - self.sync_data[x - 1] > 1.3:
                self.sync_data = self.sync_data[x:]
                break

        for x in reversed(range(1, len(self.sync_data))):
            if self.sync_data[x] - self.sync_data[x - 1] > 1.3:
                self.sync_data = self.sync_data[:x]
                break

        # Each synchronization pulse consists of two LogicAnalyzer pulses, so four
        # entries in time_stamp_data (rising edge, falling edge, rising edge, falling edge).
        # If we have less then twelve entries, we observed no transitions and don't even have
        # valid synchronization data. In this case, we bail out.
        if len(self.sync_data) < 12:
            raise RuntimeError(
                f"LogicAnalyzer sync data has length {len(time_stamp_data)}, expected >= 12"
            )

        self.online_timestamps = self.sync_data[2:3] + self.sync_data[4:-7]
        self.online_timestamps = (
            np.array(self.online_timestamps) - self.online_timestamps[0]
        )

    def analyze_states(self, expected_trace, repeat_id):
        return super().analyze_states(expected_trace, repeat_id, self.online_timestamps)


class EnergyTraceWithTimer(ExternalTimerSync):
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

        self.sync_min_duration = 0.7
        self.sync_min_low_count = 3
        self.sync_min_high_count = 1
        self.sync_power = 0.011

    def load_data(self, log_data):
        self.sync_data = None
        (
            self.interval_start_timestamp,
            self.interval_duration,
            self.interval_power,
            self.sample_rate,
            self.hw_statechange_indexes,
        ) = _load_energytrace(log_data[0])

        # for analyze_states
        self.timestamps = self.interval_start_timestamp
        self.data = self.interval_power
