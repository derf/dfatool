#!/usr/bin/env python3

import numpy as np
import ruptures
from multiprocessing import Pool

# returns the found changepoints by algo for the specific penalty pen.
# algo should be the return value of Pelt(...).fit(signal)
# Also puts a token in container q to let the progressmeter know the changepoints for penalty pen
# have been calculated.
# used for parallel calculation of changepoints vs penalty
def _get_breakpoints(algo, pen):
    return pen, len(algo.predict(pen=pen))


def find_knee_point(data_x, data_y, S=1.0, curve="convex", direction="decreasing"):
    kneedle = kneed.KneeLocator(data_x, data_y, S=S, curve=curve, direction=direction)
    kneepoint = (kneedle.knee, kneedle.knee_y)
    return kneepoint


def norm_signal(signal, scaler=25):
    max_val = max(signal)
    normed_signal = np.zeros(shape=len(signal))
    for i, signal_i in enumerate(signal):
        normed_signal[i] = signal_i / max_val
        normed_signal[i] = normed_signal[i] * scaler
    return normed_signal


# Scheint Einfluss auf die gefundene Anzahl CHangepoints zu haben. Hrmpf.
# Kann aber auch shitty downsampling sein. Diese Funktion ist _sehr_ quick&dirty.
def coarse_signal(signal, divisor=10):
    ret = list()
    for i in range((len(signal) // divisor)):
        ret.append(np.mean(signal[i * divisor : (i + 1) * divisor]))
    return np.array(ret)


# returns the changepoints found on signal with penalty penalty.
# model, jump and min_dist are directly passed to PELT
def calc_pelt(signal, penalty, model="l1", jump=5, min_dist=2, plotting=False):
    # default params in Function
    if model is None:
        model = "l1"
    if jump is None:
        jump = 5
    if min_dist is None:
        min_dist = 2
    if plotting is None:
        plotting = False
    # change point detection. best fit seemingly with l1. rbf prods. RuntimeErr for pen > 30
    # https://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/costs/index.html
    # model = "l1"   #"l1"  # "l2", "rbf"
    algo = ruptures.Pelt(model=model, jump=jump, min_size=min_dist).fit(signal)

    if penalty is not None:
        bkps = algo.predict(pen=penalty)
        if plotting:
            fig, ax = ruptures.display(signal, bkps)
            plt.show()
        return bkps

    print_error("No Penalty specified.")
    sys.exit(-1)


# calculates the raw_states for measurement measurement. num_measurement is used to identify the
# return value
# penalty, model and jump are directly passed to pelt
def calc_raw_states_func(num_measurement, measurement, penalty, model, jump):
    # extract signal
    signal = np.array(measurement)
    # norm signal to remove dependency on absolute values
    normed_signal = norm_signal(signal)
    # calculate the breakpoints
    bkpts = calc_pelt(normed_signal, penalty, model=model, jump=jump)
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


# parallelize calc over all measurements
def calc_raw_states(arg_list):
    with Pool() as pool:
        # collect results from pool
        result = pool.starmap(calc_raw_states_func, arg_list)
    return result


class PELT:
    def __init__(self, **kwargs):
        # Defaults von Janis
        self.jump = 1
        self.refinement_threshold = 100
        self.range_min = 0
        self.range_max = 100
        self.__dict__.update(kwargs)

    # signals: a set of uW measurements belonging to a single parameter configuration (i.e., a single by_param entry)
    def needs_refinement(self, signals):
        count = 0
        for signal in signals:
            p1, median, p99 = np.percentile(signal, (1, 50, 99))

            if median - p1 > self.refinement_threshold:
                count += 1
            elif p99 - median > self.refinement_threshold:
                count += 1
        refinement_ratio = count / len(signals)
        return refinement_ratio > 0.3

    def get_penalty_value(
        self, signals, model="l1", min_dist=2, range_min=0, range_max=100, S=1.0
    ):
        # Janis macht hier noch kein norm_signal. Mit sieht es aber genau so brauchbar aus.
        # TODO vor der Penaltybestimmung die Auflösung der Daten auf z.B. 1 kHz
        # verringern. Dann geht's deutlich schneller und superkurze
        # Substates interessieren uns ohnehin weniger
        signal = coarse_signal(norm_signal(signals[0]))
        algo = ruptures.Pelt(model=model, jump=self.jump, min_size=min_dist).fit(signal)
        queue = list()
        for i in range(range_min, range_max + 1):
            queue.append((algo, i))
        with Pool() as pool:
            results = pool.starmap(_get_breakpoints, queue)
        pen_val = [x[0] for x in results]
        changepoint_counts = [x[1] for x in results]
        # Scheint unnötig zu sein, da wir ohnehin plateau detection durchführen
        # knee = find_knee_point(pen_val, changepoint_counts, S=S)
        knee = (0,)

        start_index = -1
        end_index = -1
        longest_start = -1
        longest_end = -1
        prev_val = -1
        for i, num_bkpts in enumerate(changepoint_counts[knee[0] :]):
            if num_bkpts != prev_val:
                end_index = i - 1
                if end_index - start_index > longest_end - longest_start:
                    # currently found sequence is the longest found yet
                    longest_start = start_index
                    longest_end = end_index
                start_index = i
            if i == len(changepoint_counts[knee[0] :]) - 1:
                # end sequence with last value
                end_index = i
                # # since it is not guaranteed that this is the end of the plateau, assume the mid
                # # of the plateau was hit.
                # size = end_index - start_index
                # end_index = end_index + size
                # However this is not the clean solution. Better if search interval is widened
                # with range_min and range_max
                if end_index - start_index > longest_end - longest_start:
                    # last found sequence is the longest found yet
                    longest_start = start_index
                    longest_end = end_index
                start_index = i
            prev_val = num_bkpts
        mid_of_plat = longest_start + (longest_end - longest_start) // 2
        knee = (mid_of_plat + knee[0], changepoint_counts[mid_of_plat + knee[0]])

        # modify knee according to options. Defaults to 1 * knee
        knee = (knee[0] * 1, knee[1])
        return knee

    def calc_raw_states(self, signals, penalty, opt_model=None):
        raw_states_calc_args = list()
        for num_measurement, measurement in enumerate(signals):
            raw_states_calc_args.append(
                (num_measurement, measurement, penalty, opt_model, self.jump)
            )

        raw_states_list = [None] * len(signals)
        raw_states_res = calc_raw_states(raw_states_calc_args)

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

    """
    # calculates and returns the necessary penalty for signal. Parallel execution with num_processes many processes
    # jump, min_dist are passed directly to PELT. S is directly passed to kneedle.
    # pen_modifier is used as a factor on the resulting penalty.
    # the interval [range_min, range_max] is used for searching.
    def calculate_penalty_value(
        signal,
        model="l1",
        jump=5,
        min_dist=2,
        range_min=0,
        range_max=100,
        S=1.0,
        pen_modifier=None,
        show_plots=False,
    ):
        # default params in Function
        if model is None:
            model = "l1"
        if jump is None:
            jump = 5
        if min_dist is None:
            min_dist = 2
        if range_min is None:
            range_min = 0
        if range_max is None:
            range_max = 50
        if S is None:
            S = 1.0
        if pen_modifier is None:
            pen_modifier = 1
        # change point detection. best fit seemingly with l1. rbf prods. RuntimeErr for pen > 30
        # https://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/costs/index.html
        # model = "l1"   #"l1"  # "l2", "rbf"
        algo = ruptures.Pelt(model=model, jump=jump, min_size=min_dist).fit(signal)

        ### CALC BKPS WITH DIFF PENALTYS
        if range_max != range_min:
            # building args array for parallelizing
            args = []
            # for displaying progression
            m = Manager()
            q = m.Queue()

            for i in range(range_min, range_max + 1):
                # same calculation for all except other penalty
                args.append((algo, i, q))

            print_info("starting kneepoint calculation.")
            # init Pool with num_proesses
            with Pool() as p:
                # collect results from pool
                result = p.starmap_async(get_bkps, args)
                # monitor loop
                percentage = -100  # Force display of 0%
                i = 0
                while True:
                    if result.ready():
                        break

                    size = q.qsize()
                    last_percentage = percentage
                    percentage = round(size / (range_max - range_min) * 100, 2)
                    if percentage >= last_percentage + 2 or i >= refresh_thresh:
                        print_info("Current progress: " + str(percentage) + "%")
                        i = 0
                    else:
                        i += 1
                    time.sleep(refresh_delay)
                res = result.get()
            print_info("Finished kneepoint calculation.")
            # DECIDE WHICH PENALTY VALUE TO CHOOSE ACCORDING TO ELBOW/KNEE APPROACH
            # split x and y coords to pass to kneedle
            pen_val = [x[0] for x in res]
            fitted_bkps_val = [x[1] for x in res]
            # # plot to look at res
            knee = find_knee_point(pen_val, fitted_bkps_val, S=S)

            # TODO: Find plateau on pen_val vs fitted_bkps_val
            #   scipy.find_peaks() does not find plateaus if they extend through the end of the data.
            #   to counter that, add one extremely large value to the right side of the data
            #   after negating it is extremely small -> Almost certainly smaller than the
            #   found plateau therefore the plateau does not extend through the border
            #   -> scipy.find_peaks finds it. Choose value from within that plateau.
            # fitted_bkps_val.append(100000000)
            # TODO: Approaching over find_peaks might not work if the initial decrease step to the
            #   "correct" number of changepoints and additional decrease steps e.g. underfitting
            #   take place within the given penalty interval. find_peak only finds plateaus
            #   of peaks. If the number of chpts decreases after the wanted plateau the condition
            #   for local peaks is not satisfied anymore. Therefore this approach will only work
            #   if the plateau extends over the right border of the penalty interval.
            # peaks, peak_plateaus = find_peaks(- np.array(fitted_bkps_val), plateau_size=1)
            # Since the data is monotonously decreasing only one plateau can be found.

            # assuming the plateau is constant, i.e. no noise. OK to assume this here, since num_bkpts
            # is monotonously decreasing. If the number of bkpts decreases inside a considered
            # plateau, it means that the stable configuration is not yet met. -> Search further
            start_index = -1
            end_index = -1
            longest_start = -1
            longest_end = -1
            prev_val = -1
            for i, num_bkpts in enumerate(fitted_bkps_val[knee[0] :]):
                if num_bkpts != prev_val:
                    end_index = i - 1
                    if end_index - start_index > longest_end - longest_start:
                        # currently found sequence is the longest found yet
                        longest_start = start_index
                        longest_end = end_index
                    start_index = i
                if i == len(fitted_bkps_val[knee[0] :]) - 1:
                    # end sequence with last value
                    end_index = i
                    # # since it is not guaranteed that this is the end of the plateau, assume the mid
                    # # of the plateau was hit.
                    # size = end_index - start_index
                    # end_index = end_index + size
                    # However this is not the clean solution. Better if search interval is widened
                    # with range_min and range_max
                    if end_index - start_index > longest_end - longest_start:
                        # last found sequence is the longest found yet
                        longest_start = start_index
                        longest_end = end_index
                    start_index = i
                prev_val = num_bkpts
            if show_plots:
                plt.xlabel("Penalty")
                plt.ylabel("Number of Changepoints")
                plt.plot(pen_val, fitted_bkps_val)
                plt.vlines(
                    longest_start + knee[0], 0, max(fitted_bkps_val), linestyles="dashed"
                )
                plt.vlines(
                    longest_end + knee[0], 0, max(fitted_bkps_val), linestyles="dashed"
                )
                plt.show()
            # choosing pen from plateau
            mid_of_plat = longest_start + (longest_end - longest_start) // 2
            knee = (mid_of_plat + knee[0], fitted_bkps_val[mid_of_plat + knee[0]])

            # modify knee according to options. Defaults to 1 * knee
            knee = (knee[0] * pen_modifier, knee[1])

        else:
            # range_min == range_max. has the same effect as pen_override
            knee = (range_min, None)
        print_info(str(knee[0]) + " has been selected as penalty.")
        if knee[0] is not None:
            return knee

        print_error(
            "With the current thresh-hold S="
            + str(S)
            + " it is not possible to select a penalty value."
        )
        sys.exit(-1)
    """
