import numpy as np
from multiprocessing import Pool


def PELT_get_changepoints(algo, penalty):
    res = (penalty, algo.predict(pen=penalty))
    return res


# calculates the raw_states for measurement measurement. num_measurement is used to identify the
# return value
# penalty, model and jump are directly passed to pelt
def PELT_get_raw_states(num_measurement, algo, signal, penalty):
    bkpts = algo.predict(pen=penalty)
    calced_states = list()
    start_time = 0
    end_time = 0
    # calc metrics for all states
    for bkpt in bkpts:
        # start_time of state is end_time of previous one
        # (Transitions are instantaneous)
        start_time = end_time
        end_time = bkpt
        power_vals = signal[start_time:end_time]
        mean_power = np.mean(power_vals)
        std_dev = np.std(power_vals)
        calced_state = (start_time, end_time, mean_power, std_dev)
        calced_states.append(calced_state)
    num = 0
    new_avg_std = 0
    # calc avg std for all states from this measurement
    for s in calced_states:
        # print_info("State " + str(num) + " starts at t=" + str(s[0])
        #            + " and ends at t=" + str(s[1])
        #            + " while using " + str(s[2])
        #            + "uW with  sigma=" + str(s[3]))
        num = num + 1
        new_avg_std = new_avg_std + s[3]
    # check case if no state has been found to avoid crashing
    if len(calced_states) != 0:
        new_avg_std = new_avg_std / len(calced_states)
    else:
        new_avg_std = 0
    change_avg_std = None  # measurement["uW_std"] - new_avg_std
    # print_info("The average standard deviation for the newly found states is "
    #            + str(new_avg_std))
    # print_info("That is a reduction of " + str(change_avg_std))
    return num_measurement, calced_states, new_avg_std, change_avg_std


class PELT:
    def __init__(self, **kwargs):
        self.model = "l1"
        self.jump = 1
        self.min_dist = 10
        self.num_samples = None
        self.refinement_threshold = 200e-6  # µW
        self.range_min = 0
        self.range_max = 100
        self.__dict__.update(kwargs)

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

    def get_penalty_and_changepoints(self, signal):
        # imported here as ruptures is only used for changepoint detection.
        # This way, dfatool can be used without having ruptures installed as
        # long as --pelt isn't active.
        import ruptures

        if self.num_samples is not None and len(signal) > self.num_samples:
            self.jump = len(signal) // int(self.num_samples)
            print(f"jump = {self.jump}")
        else:
            self.jump = 1

        algo = ruptures.Pelt(
            model=self.model, jump=self.jump, min_size=self.min_dist
        ).fit(self.norm_signal(signal))
        queue = list()
        for i in range(0, 100):
            queue.append((algo, i))
        with Pool() as pool:
            changepoints = pool.starmap(PELT_get_changepoints, queue)
        changepoints_by_penalty = dict()
        for res in changepoints:
            if len(res[1]) > 0 and res[1][-1] == len(signal):
                res[1].pop()
            changepoints_by_penalty[res[0]] = res[1]
        num_changepoints = list()
        for i in range(0, 100):
            num_changepoints.append(len(changepoints_by_penalty[i]))

        start_index = -1
        end_index = -1
        longest_start = -1
        longest_end = -1
        prev_val = -1
        for i, num_bkpts in enumerate(num_changepoints):
            if num_bkpts != prev_val:
                end_index = i - 1
                if end_index - start_index > longest_end - longest_start:
                    longest_start = start_index
                    longest_end = end_index
                start_index = i
            if i == len(num_changepoints) - 1:
                end_index = i
                if end_index - start_index > longest_end - longest_start:
                    longest_start = start_index
                    longest_end = end_index
                start_index = i
            prev_val = num_bkpts
        middle_of_plateau = longest_start + (longest_start - longest_start) // 2
        changepoints = np.array(changepoints_by_penalty[middle_of_plateau])
        return middle_of_plateau, changepoints

    def get_changepoints(self, signal):
        _, changepoints = self.get_penalty_and_changepoints(signal)
        return changepoints

    def get_penalty(self, signal):
        penalty, _ = self.get_penalty_and_changepoints(signal)
        return penalty

    def calc_raw_states(self, signals, penalty, opt_model=None):
        # imported here as ruptures is only used for changepoint detection.
        # This way, dfatool can be used without having ruptures installed as
        # long as --pelt isn't active.
        import ruptures

        raw_states_calc_args = list()
        for num_measurement, measurement in enumerate(signals):
            normed_signal = self.norm_signal(measurement)
            algo = ruptures.Pelt(
                model=self.model, jump=self.jump, min_size=self.min_dist
            ).fit(normed_signal)
            raw_states_calc_args.append((num_measurement, algo, normed_signal, penalty))

        raw_states_list = [None] * len(signals)
        with Pool() as pool:
            raw_states_res = pool.starmap(PELT_get_raw_states, raw_states_calc_args)

        # extracting result and putting it in correct order -> index of raw_states_list
        # entry still corresponds with index of measurement in measurements_by_states
        # -> If measurements are discarded the used ones are easily recognized
        for ret_val in raw_states_res:
            num_measurement = ret_val[0]
            raw_states = ret_val[1]
            avg_std = ret_val[2]
            change_avg_std = ret_val[3]
            # FIXME: Wieso gibt mir meine IDE hier eine Warning aus? Der Index müsste doch
            #   int sein oder nicht? Es scheint auch vernünftig zu klappen...
            raw_states_list[num_measurement] = raw_states
            # print(
            #    "The average standard deviation for the newly found states in "
            #    + "measurement No. "
            #    + str(num_measurement)
            #    + " is "
            #    + str(avg_std)
            # )
            # print("That is a reduction of " + str(change_avg_std))
            for i, raw_state in enumerate(raw_states):
                print(
                    f"Measurement #{num_measurement} sub-state #{i}: {raw_state[0]} -> {raw_state[1]}, mean {raw_state[2]}"
                )
            # l_signal = measurements_by_config['offline'][num_measurement]['uW']
            # l_bkpts = [s[1] for s in raw_states]
            # fig, ax = rpt.display(np.array(l_signal), l_bkpts)
            # plt.show()
