import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, sync_data, energy_data):
        """
        Creates DataProcessor object.

        :param sync_data: input timestamps (SigrokResult)
        :param energy_data: List of EnergyTrace datapoints
        """
        self.reduced_timestamps = []
        self.modified_timestamps = []
        self.plot_data_x = []
        self.plot_data_y = []
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

        last_data = [0, 0, 0, 0]

        # clean timestamp data, if at the end strange ts got added somehow
        time_stamp_data = self.removeTooFarDatasets(time_stamp_data)

        self.reduced_timestamps = time_stamp_data

        # NEW
        datasync_timestamps = []
        sync_start = 0
        outliers = 0
        pre_outliers_ts = None
        for i, energytrace_dataset in enumerate(self.energy_data):
            usedtime = energytrace_dataset[0] - last_data[0]  # in microseconds
            timestamp = energytrace_dataset[0]
            usedenergy = energytrace_dataset[3] - last_data[3]
            power = usedenergy / usedtime * 10 ** -3  # in watts
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

            self.plot_data_x.append(energytrace_dataset[0] / 1_000_000)
            self.plot_data_y.append(power)

        if power > self.power_sync_watt:
            if (self.energy_data[-1][0] - sync_start) / 1_000_000 > self.power_sync_len:
                datasync_timestamps.append(
                    (sync_start / 1_000_000, pre_outliers_ts / 1_000_000)
                )

        logger.debug(f"Synchronization areas: {datasync_timestamps}")
        # print(time_stamp_data[2])

        start_offset = datasync_timestamps[0][1] - time_stamp_data[2]
        start_timestamp = datasync_timestamps[0][1]

        end_offset = datasync_timestamps[-2][0] - (time_stamp_data[-8] + start_offset)
        end_timestamp = datasync_timestamps[-2][0]
        logger.debug(
            f"Measurement area: LA timestamp range [{start_timestamp}, {end_timestamp}]"
        )
        logger.debug(f"Start/End offsets: {start_offset} / {end_offset}")

        if end_offset > 10:
            logger.warning(
                f"synchronization end_offset == {end_offset}. It should be no more than a few seconds."
            )

        with_offset = self.addOffset(time_stamp_data, start_offset)

        with_drift = self.addDrift(
            with_offset, end_timestamp, end_offset, start_timestamp
        )

        self.modified_timestamps = with_drift

    def addOffset(self, input_timestamps, start_offset):
        """
        Add begin offset at start

        :param input_timestamps: List of timestamps (float list)
        :param start_offset: Timestamp of last EnergyTrace datapoint at the first sync point
        :return: List of modified timestamps (float list)
        """
        modified_timestamps_with_offset = []
        for x in input_timestamps:
            if x + start_offset >= 0:
                modified_timestamps_with_offset.append(x + start_offset)
        return modified_timestamps_with_offset

    def removeTooFarDatasets(self, input_timestamps):
        """
        Removing datasets, that are to far away at ethe end

        :param input_timestamps: List of timestamps (float list)
        :return: List of modified timestamps (float list)
        """
        modified_timestamps = []
        for i, x in enumerate(input_timestamps):
            # print(x - input_timestamps[i - 1], x - input_timestamps[i - 1] < 2.5)
            if x - input_timestamps[i - 1] < 1.6:
                modified_timestamps.append(x)
            else:
                break
        return modified_timestamps

    def addDrift(self, input_timestamps, end_timestamp, end_offset, start_timestamp):
        """
        Add drift to datapoints

        :param input_timestamps: List of timestamps (float list)
        :param end_timestamp: Timestamp of first EnergyTrace datapoint at the second last sync point
        :param end_offset: the time between end_timestamp and the timestamp of synchronisation signal
        :param start_timestamp: Timestamp of last EnergyTrace datapoint at the first sync point
        :return: List of modified timestamps (float list)
        """
        endFactor = (end_timestamp + end_offset - start_timestamp) / (
            end_timestamp - start_timestamp
        )
        modified_timestamps_with_drift = []
        for x in input_timestamps:
            modified_timestamps_with_drift.append(
                ((x - start_timestamp) * endFactor) + start_timestamp
            )

        return modified_timestamps_with_drift

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
            a[1::4] = max_value
            a[2::4] = max_value
            a[3::4] = min_value
            a[4::4] = min_value
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
            self.modified_timestamps, max_value=max(self.plot_data_y)
        )

        plt.plot(self.plot_data_x, self.plot_data_y, label="Leistung")

        plt.plot(
            rectCurve_with_drift[0],
            rectCurve_with_drift[1],
            "-g",
            label="Synchronisationsignale mit Driftfaktor",
        )

        plt.xlabel("Zeit [s]")
        plt.ylabel("Leistung [W]")
        leg = plt.legend()

        def getDataText(x):
            # print(x)
            for i, xt in enumerate(self.modified_timestamps):
                if xt > x:
                    return "Value: %s" % annotateData[i - 5]

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

    def getPowerBetween(self, start, end, state_sleep):  # 0.001469
        """
        calculates the average powerusage in interval
        NOT SIDE EFFECT FREE, DON'T USE IT EVERYWHERE

        :param start: Start timestamp of interval
        :param end: End timestamp of interval
        :param state_sleep: Length in seconds of one state, needed for cutting out the UART Sending cycle
        :return: float with average power usage
        """
        first_index = 0
        all_power = []
        for ind in range(self.start_offset, len(self.plot_data_x)):
            first_index = ind
            if self.plot_data_x[ind] > start:
                break

        nextIndAfterIndex = None
        for ind in range(first_index, len(self.plot_data_x)):
            nextIndAfterIndex = ind
            if (
                self.plot_data_x[ind] > end
                or self.plot_data_x[ind] > start + state_sleep
            ):
                self.start_offset = ind - 1
                break
            all_power.append(self.plot_data_y[ind])

        # TODO Idea remove datapoints that are too far away
        def removeSD_Mean_Values(arr):
            import numpy

            elements = numpy.array(arr)

            mean = numpy.mean(elements, axis=0)
            sd = numpy.std(elements, axis=0)

            return [x for x in arr if (mean - 1 * sd < x < mean + 1.5 * sd)]

        if len(all_power) > 10:
            # all_power = removeSD_Mean_Values(all_power)
            pass
        # TODO algorithm relocate datapoint

        pre_fix_len = len(all_power)
        if len(all_power) == 0:
            # print("PROBLEM")
            all_power.append(self.plot_data_y[nextIndAfterIndex])
        elif len(all_power) == 1:
            # print("OKAY")
            pass
        return pre_fix_len, sum(all_power) / len(all_power)

    def getStatesdfatool(self, state_sleep, algorithm=False):
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

        for ts_index in range(
            0 + timestamps_sync_start, int(len(self.modified_timestamps) / 2)
        ):
            start_transition_ts = self.modified_timestamps[ts_index * 2]
            start_transition_ts_timing = self.reduced_timestamps[ts_index * 2]

            if end_transition_ts is not None:
                count_dp, power = self.getPowerBetween(
                    end_transition_ts, start_transition_ts, state_sleep
                )

                # print("STATE", end_transition_ts * 10 ** 6, start_transition_ts * 10 ** 6, (start_transition_ts - end_transition_ts) * 10 ** 6, power)
                if (
                    (start_transition_ts - end_transition_ts) * 10 ** 6 > 900_000
                    and power > self.power_sync_watt * 0.9
                    and ts_index > 10
                ):
                    # remove last transition and stop (upcoming data only sync)
                    del energy_trace_new[-1]
                    break
                    pass

                state = {
                    "isa": "state",
                    "W_mean": power,
                    "W_std": 0.0001,
                    "s": (
                        start_transition_ts_timing - end_transition_ts_timing
                    ),  # * 10 ** 6,
                }
                energy_trace_new.append(state)

                energy_trace_new[-2]["W_mean_delta_next"] = (
                    energy_trace_new[-2]["W_mean"] - energy_trace_new[-1]["W_mean"]
                )

                # get energy end_transition_ts
            end_transition_ts = self.modified_timestamps[ts_index * 2 + 1]
            count_dp, power = self.getPowerBetween(
                start_transition_ts, end_transition_ts, state_sleep
            )

            # print("TRANS", start_transition_ts * 10 ** 6, end_transition_ts * 10 ** 6, (end_transition_ts - start_transition_ts) * 10 ** 6, power)
            end_transition_ts_timing = self.reduced_timestamps[ts_index * 2 + 1]

            transition = {
                "isa": "transition",
                "W_mean": power,
                "W_std": 0.0001,
                "s": (
                    end_transition_ts_timing - start_transition_ts_timing
                ),  # * 10 ** 6,
                "count_dp": count_dp,
            }

            if (end_transition_ts - start_transition_ts) * 10 ** 6 > 2_000_000:
                # TODO Last data set corrupted? HOT FIX!!!!!!!!!!!! REMOVE LATER
                # for x in range(4):
                #    del energy_trace_new[-1]
                # break
                pass

            energy_trace_new.append(transition)
            # print(start_transition_ts, "-", end_transition_ts, "-", end_transition_ts - start_transition_ts)
        return energy_trace_new
