#!/usr/bin/env python3

import io
import logging
import numpy as np
import struct
import tarfile

from dfatool.utils import soft_cast_int

logger = logging.getLogger(__name__)

arg_support_enabled = True


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
            if `self.with_traces` is true, it also contains:
            * `plot`: (timestamps [s], power readings [W])
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

    def validate(self, num_triggers, observed_trace, expected_traces, state_duration):
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
        if len(self.errors):
            return False

        # Check trigger count
        sched_trigger_count = 0
        for run in expected_traces:
            sched_trigger_count += len(run["trace"])
        if sched_trigger_count != num_triggers:
            self.errors.append(
                "got {got:d} trigger edges, expected {exp:d}".format(
                    got=num_triggers, exp=sched_trigger_count
                )
            )
            return False
        # Check state durations. Very short or long states can indicate a
        # missed trigger signal which wasn't detected due to duplicate
        # triggers elsewhere
        online_datapoints = []
        for run_idx, run in enumerate(expected_traces):
            for trace_part_idx in range(len(run["trace"])):
                online_datapoints.append((run_idx, trace_part_idx))
        for offline_idx, (online_run_idx, online_trace_part_idx) in enumerate(
            online_datapoints
        ):
            offline_trace_part = observed_trace[offline_idx]
            online_trace_part = expected_traces[online_run_idx]["trace"][
                online_trace_part_idx
            ]

            if online_trace_part["isa"] != offline_trace_part["isa"]:
                self.errors.append(
                    "Offline #{off_idx:d} (online {on_name:s} @ {on_idx:d}/{on_sub:d}) claims to be {off_isa:s}, but should be {on_isa:s}".format(
                        off_idx=offline_idx,
                        on_idx=online_run_idx,
                        on_sub=online_trace_part_idx,
                        on_name=online_trace_part["name"],
                        off_isa=offline_trace_part["isa"],
                        on_isa=online_trace_part["isa"],
                    )
                )
                return False

            # Clipping in UNINITIALIZED (offline_idx == 0) can happen during
            # calibration and is handled by MIMOSA
            if (
                offline_idx != 0
                and offline_trace_part["clip_rate"] != 0
                and not self.ignore_clipping
            ):
                self.errors.append(
                    "Offline #{off_idx:d} (online {on_name:s} @ {on_idx:d}/{on_sub:d}) was clipping {clip:f}% of the time".format(
                        off_idx=offline_idx,
                        on_idx=online_run_idx,
                        on_sub=online_trace_part_idx,
                        on_name=online_trace_part["name"],
                        clip=offline_trace_part["clip_rate"] * 100,
                    )
                )
                return False

            if (
                online_trace_part["isa"] == "state"
                and online_trace_part["name"] != "UNINITIALIZED"
                and len(expected_traces[online_run_idx]["trace"])
                > online_trace_part_idx + 1
            ):
                online_prev_transition = expected_traces[online_run_idx]["trace"][
                    online_trace_part_idx - 1
                ]
                online_next_transition = expected_traces[online_run_idx]["trace"][
                    online_trace_part_idx + 1
                ]
                try:
                    if self._state_is_too_short(
                        online_trace_part,
                        offline_trace_part,
                        state_duration,
                        online_next_transition,
                    ):
                        self.errors.append(
                            "Offline #{off_idx:d} (online {on_name:s} @ {on_idx:d}/{on_sub:d}) is too short (duration = {dur:d} us)".format(
                                off_idx=offline_idx,
                                on_idx=online_run_idx,
                                on_sub=online_trace_part_idx,
                                on_name=online_trace_part["name"],
                                dur=offline_trace_part["us"],
                            )
                        )
                        return False
                    if self._state_is_too_long(
                        online_trace_part,
                        offline_trace_part,
                        state_duration,
                        online_prev_transition,
                    ):
                        self.errors.append(
                            "Offline #{off_idx:d} (online {on_name:s} @ {on_idx:d}/{on_sub:d}) is too long (duration = {dur:d} us)".format(
                                off_idx=offline_idx,
                                on_idx=online_run_idx,
                                on_sub=online_trace_part_idx,
                                on_name=online_trace_part["name"],
                                dur=offline_trace_part["us"],
                            )
                        )
                        return False
                except KeyError:
                    pass
                    # TODO es gibt next_transitions ohne 'plan'
        return True

    @staticmethod
    def add_offline_aggregates(online_traces, offline_trace, repeat_id):
        # Edits online_traces[*]['trace'][*]['offline']
        # and online_traces[*]['trace'][*]['offline_aggregates'] in place
        # (appends data from offline_trace)
        # "offline_aggregates" is the only data used later on by model.py's by_name / by_param dicts
        online_datapoints = []
        for run_idx, run in enumerate(online_traces):
            for trace_part_idx in range(len(run["trace"])):
                online_datapoints.append((run_idx, trace_part_idx))
        for offline_idx, (online_run_idx, online_trace_part_idx) in enumerate(
            online_datapoints
        ):
            offline_trace_part = offline_trace[offline_idx]
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

            if "plot" in offline_trace_part:
                online_trace_part["offline_aggregates"]["power_traces"].append(
                    offline_trace_part["plot"][1]
                )
                online_trace_part["offline_aggregates"]["timestamps"].append(
                    offline_trace_part["plot"][0]
                )
