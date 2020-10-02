#!/usr/bin/env python3
import json
import os
import time
import sys
import getopt
import re
import pprint
from multiprocessing import Pool, Manager, cpu_count
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import ruptures as rpt
import numpy as np

from dfatool.functions import analytic
from dfatool.loader import RawData
from dfatool import parameters
from dfatool.model import ParallelParamFit, PTAModel
from dfatool.utils import by_name_to_by_param

# from scipy.cluster.hierarchy import dendrogram, linkage # for graphical display

# py bin\Proof_Of_Concept_PELT.py --filename="..\data\TX.json" --jump=1 --pen_override=28 --refinement_thresh=100
# py bin\Proof_Of_Concept_PELT.py --filename="..\data\TX.json" --jump=1 --pen_override=28 --refinement_thresh=100 --cache_dicts --cache_loc="..\data\TX2_cache"
from dfatool.validation import CrossValidator


# helper functions. Not used
def plot_data_from_json(filename, trace_num, x_axis, y_axis):
    with open(filename, "r") as file:
        tx_data = json.load(file)
    print(tx_data[trace_num]["parameter"])
    plt.plot(tx_data[trace_num]["offline"][0]["uW"])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()


def plot_data_vs_mean(signal, x_axis, y_axis):
    plt.plot(signal)
    average = np.mean(signal)
    plt.hlines(average, 0, len(signal))
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()


def plot_data_vs_data_vs_means(signal1, signal2, x_axis, y_axis):
    plt.plot(signal1)
    lens = max(len(signal1), len(signal2))
    average = np.mean(signal1)
    plt.hlines(average, 0, lens, color="red")
    plt.vlines(len(signal1), 0, 100000, color="red", linestyles="dashed")
    plt.plot(signal2)
    average = np.mean(signal2)
    plt.hlines(average, 0, lens, color="green")
    plt.vlines(len(signal2), 0, 100000, color="green", linestyles="dashed")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()


# returns the found changepoints by algo for the specific penalty pen.
# algo should be the return value of Pelt(...).fit(signal)
# Also puts a token in container q to let the progressmeter know the changepoints for penalty pen
# have been calculated.
# used for parallel calculation of changepoints vs penalty
def get_bkps(algo, pen, q):
    res = pen, len(algo.predict(pen=pen))
    q.put(pen)
    return res


# Wrapper for kneedle
def find_knee_point(data_x, data_y, S=1.0, curve="convex", direction="decreasing"):
    kneedle = KneeLocator(data_x, data_y, S=S, curve=curve, direction=direction)
    kneepoint = (kneedle.knee, kneedle.knee_y)
    return kneepoint


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
    algo = rpt.Pelt(model=model, jump=jump, min_size=min_dist).fit(signal)

    if penalty is not None:
        bkps = algo.predict(pen=penalty)
        if plotting:
            fig, ax = rpt.display(signal, bkps)
            plt.show()
        return bkps

    print_error("No Penalty specified.")
    sys.exit(-1)


# calculates and returns the necessary penalty for signal. Parallel execution with num_processes many processes
# jump, min_dist are passed directly to PELT. S is directly passed to kneedle.
# pen_modifier is used as a factor on the resulting penalty.
# the interval [range_min, range_max] is used for searching.
# refresh_delay and refresh_thresh are used to configure the progress "bar".
def calculate_penalty_value(
    signal,
    model="l1",
    jump=5,
    min_dist=2,
    range_min=0,
    range_max=50,
    num_processes=8,
    refresh_delay=1,
    refresh_thresh=5,
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
    if num_processes is None:
        num_processes = 8
    if refresh_delay is None:
        refresh_delay = 1
    if refresh_thresh is None:
        refresh_thresh = 5
    if S is None:
        S = 1.0
    if pen_modifier is None:
        pen_modifier = 1
    # change point detection. best fit seemingly with l1. rbf prods. RuntimeErr for pen > 30
    # https://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/costs/index.html
    # model = "l1"   #"l1"  # "l2", "rbf"
    algo = rpt.Pelt(model=model, jump=jump, min_size=min_dist).fit(signal)

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
        with Pool(min(num_processes, len(args))) as p:
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


# calculates the raw_states for measurement measurement. num_measurement is used to identify the
# return value
# penalty, model and jump are directly passed to pelt
def calc_raw_states_func(num_measurement, measurement, penalty, model, jump):
    # extract signal
    signal = np.array(measurement["uW"])
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
    change_avg_std = measurement["uW_std"] - new_avg_std
    # print_info("The average standard deviation for the newly found states is "
    #            + str(new_avg_std))
    # print_info("That is a reduction of " + str(change_avg_std))
    return num_measurement, calced_states, new_avg_std, change_avg_std


# parallelize calc over all measurements
def calc_raw_states(arg_list, num_processes=8):
    m = Manager()
    with Pool(processes=min(num_processes, len(arg_list))) as p:
        # collect results from pool
        result = p.starmap(calc_raw_states_func, arg_list)
    return result


# Very short benchmark yielded approx. 3 times the speed of solution not using sort
# checks the percentiles if refinement is necessary
def needs_refinement(signal, thresh):
    sorted_signal = sorted(signal)
    length_of_signal = len(signal)
    percentile_size = int()
    percentile_size = length_of_signal // 100
    lower_percentile = sorted_signal[0:percentile_size]
    upper_percentile = sorted_signal[
        length_of_signal - percentile_size : length_of_signal
    ]
    lower_percentile_mean = np.mean(lower_percentile)
    upper_percentile_mean = np.mean(upper_percentile)
    median = np.median(sorted_signal)
    dist = median - lower_percentile_mean
    if dist > thresh:
        return True
    dist = upper_percentile_mean - median
    if dist > thresh:
        return True
    return False


# helper functions for user output
# TODO: maybe switch with python logging feature
def print_info(str_to_prt):
    str_lst = str_to_prt.split(sep="\n")
    for str_prt in str_lst:
        print("[INFO]" + str_prt)


def print_warning(str_to_prt):
    str_lst = str_to_prt.split(sep="\n")
    for str_prt in str_lst:
        print("[WARNING]" + str_prt)


def print_error(str_to_prt):
    str_lst = str_to_prt.split(sep="\n")
    for str_prt in str_lst:
        print("[ERROR]" + str_prt, file=sys.stderr)


# norms the signal and apply scaler to all values as a factor
def norm_signal(signal, scaler=25):
    max_val = max(signal)
    normed_signal = np.zeros(shape=len(signal))
    for i, signal_i in enumerate(signal):
        normed_signal[i] = signal_i / max_val
        normed_signal[i] = normed_signal[i] * scaler
    return normed_signal


# norms the values to prepare them for clustering
def norm_values_to_cluster(values_to_cluster):
    new_vals = np.array(values_to_cluster)
    num_samples = len(values_to_cluster)
    num_params = len(values_to_cluster[0])
    for i in range(num_params):
        param_vals = []
        for sample in new_vals:
            param_vals.append(sample[i])
        max_val = np.max(np.abs(param_vals))
        for num_sample, sample in enumerate(new_vals):
            values_to_cluster[num_sample][i] = sample[i] / max_val
    return new_vals


# finds state_num using state name
def get_state_num(state_name, distinct_states):
    for state_num, states in enumerate(distinct_states):
        if state_name in states:
            return state_num
    return -1


if __name__ == "__main__":
    # OPTION RECOGNITION
    opt = dict()

    optspec = (
        "filename= "
        "v "
        "model= "
        "jump= "
        "min_dist= "
        "range_min= "
        "range_max= "
        "num_processes= "
        "refresh_delay= "
        "refresh_thresh= "
        "S= "
        "pen_override= "
        "pen_modifier= "
        "plotting= "
        "refinement_thresh= "
        "cache_dicts "
        "cache_loc= "
    )
    opt_filename = None
    opt_verbose = False
    opt_model = None
    opt_jump = None
    opt_min_dist = None
    opt_range_min = None
    opt_range_max = None
    opt_num_processes = cpu_count()
    opt_refresh_delay = None
    opt_refresh_thresh = None
    opt_S = None
    opt_pen_override = None
    opt_pen_modifier = None
    opt_plotting = False
    opt_refinement_thresh = None
    opt_cache_loc = None
    try:
        raw_opts, args = getopt.getopt(sys.argv[1:], "", optspec.split(" "))

        for option, parameter in raw_opts:
            optname = re.sub(r"^--", "", option)
            opt[optname] = parameter

        if "filename" not in opt:
            print_error("No file specified!")
            sys.exit(-1)
        else:
            opt_filename = opt["filename"]
        if "v" in opt:
            opt_verbose = True
            opt_plotting = True
        if "model" in opt:
            opt_model = opt["model"]
        if "jump" in opt:
            try:
                opt_jump = int(opt["jump"])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(-1)
        if "min_dist" in opt:
            try:
                opt_min_dist = int(opt["min_dist"])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(-1)
        if "range_min" in opt:
            try:
                opt_range_min = int(opt["range_min"])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(-1)
        if "range_max" in opt:
            try:
                opt_range_max = int(opt["range_max"])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(-1)
        if "num_processes" in opt:
            try:
                opt_num_processes = int(opt["num_processes"])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(-1)
        if "refresh_delay" in opt:
            try:
                opt_refresh_delay = int(opt["refresh_delay"])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(-1)
        if "refresh_thresh" in opt:
            try:
                opt_refresh_thresh = int(opt["refresh_thresh"])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(-1)
        if "S" in opt:
            try:
                opt_S = float(opt["S"])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(-1)
        if "pen_override" in opt:
            try:
                opt_pen_override = int(opt["pen_override"])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(-1)
        if "pen_modifier" in opt:
            try:
                opt_pen_modifier = float(opt["pen_modifier"])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(-1)
        if "refinement_thresh" in opt:
            try:
                opt_refinement_thresh = int(opt["refinement_thresh"])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(-1)
        if "cache_dicts" in opt:
            if "cache_loc" in opt:
                opt_cache_loc = opt["cache_loc"]
            else:
                print_error('If "cache_dicts" is set, "cache_loc" must be provided.')
                sys.exit(-1)
    except getopt.GetoptError as err:
        print(err, file=sys.stderr)
        sys.exit(-1)
    filepath = os.path.dirname(opt_filename)
    # OPENING DATA
    if ".json" in opt_filename:
        # open file with trace data from json
        print_info(
            "Will only refine the state which is present in "
            + opt_filename
            + " if necessary."
        )
        with open(opt_filename, "r") as f:
            configurations = json.load(f)

        # for i in range(0, 7):
        #     signal = np.array(configurations[i]['offline'][0]['uW'])
        #     plt.plot(signal)
        # plt.xlabel('Time [us]')
        # plt.ylabel('Power [mW]')
        # plt.show()
        # sys.exit()

        # resulting_sequence_list = []
        # search for param_names, by_param and by_name files
        # cachingopts
        by_param_file = None
        by_name_file = None
        param_names_file = None
        from_cache = False
        not_accurate = False
        if opt_cache_loc is not None:
            flag = False
            by_name_loc = os.path.join(opt_cache_loc, "by_name.txt")
            by_param_loc = os.path.join(opt_cache_loc, "by_param.txt")
            param_names_loc = os.path.join(opt_cache_loc, "param_names.txt")
            if os.path.isfile(by_name_loc) and os.path.getsize(by_name_loc) > 0:
                by_name_file = open(by_name_loc, "r")
            else:
                print_error("In " + opt_cache_loc + " is no by_name.txt.")
                flag = True
            if os.path.isfile(by_param_loc) and os.path.getsize(by_param_loc) > 0:
                by_param_file = open(by_param_loc, "r")
            else:
                print_error("In " + opt_cache_loc + " is no by_param.txt.")
                flag = True
            if os.path.isfile(param_names_loc) and os.path.getsize(param_names_loc) > 0:
                param_names_file = open(param_names_loc, "r")
            else:
                print_error("In " + opt_cache_loc + " is no param_names.txt.")
                flag = True
            if flag:
                print_info("The cache will be build.")
            else:
                print_warning(
                    'THE OPTION "cache_dicts" IS FOR DEBUGGING PURPOSES ONLY! '
                    "\nDO NOT USE FOR REGULAR APPLICATIONS!"
                    "\nThis will possibly not be maintained in further development."
                )
                from_cache = True
        big_state_name = configurations[0]["name"]
        if None in (by_param_file, by_name_file, param_names_file):
            state_durations_by_config = []
            state_consumptions_by_config = []
            # loop through all traces check if refinement is necessary and if necessary refine it.
            for num_config, measurements_by_config in enumerate(configurations):
                # loop through all occurrences of the looked at state
                print_info(
                    "Looking at state '"
                    + measurements_by_config["name"]
                    + "' with params: "
                    + str(measurements_by_config["parameter"])
                    + "("
                    + str(num_config + 1)
                    + "/"
                    + str(len(configurations))
                    + ")"
                )
                num_needs_refine = 0
                print_info("Checking if refinement is necessary...")
                for measurement in measurements_by_config["offline"]:
                    # loop through measurements of particular state
                    # an check if state needs refinement
                    signal = measurement["uW"]
                    # mean = measurement['uW_mean']
                    if needs_refinement(signal, opt_refinement_thresh):
                        num_needs_refine = num_needs_refine + 1
                if num_needs_refine == 0:
                    print_info(
                        "No refinement necessary for state '"
                        + measurements_by_config["name"]
                        + "' with params: "
                        + str(measurements_by_config["parameter"])
                    )
                elif num_needs_refine < len(measurements_by_config["offline"]) / 2:
                    print_info(
                        "No refinement necessary for state '"
                        + measurements_by_config["name"]
                        + "' with params: "
                        + str(measurements_by_config["parameter"])
                    )
                    print_warning(
                        "However this decision was not unanimously. This could hint a poor"
                        "measurement quality."
                    )
                else:
                    if num_needs_refine != len(measurements_by_config["parameter"]):
                        print_warning(
                            "However this decision was not unanimously. This could hint a poor"
                            "measurement quality."
                        )
                    # assume that all measurements of the same param configuration are fundamentally
                    # similar -> calculate penalty for first measurement, use it for all
                    if opt_pen_override is None:
                        signal = np.array(measurements_by_config["offline"][0]["uW"])
                        normed_signal = norm_signal(signal)
                        penalty = calculate_penalty_value(
                            normed_signal,
                            model=opt_model,
                            range_min=opt_range_min,
                            range_max=opt_range_max,
                            num_processes=opt_num_processes,
                            jump=opt_jump,
                            S=opt_S,
                            pen_modifier=opt_pen_modifier,
                        )
                        penalty = penalty[0]
                    else:
                        penalty = opt_pen_override
                    # build arguments for parallel excecution
                    print_info("Starting raw_states calculation.")
                    raw_states_calc_args = []
                    for num_measurement, measurement in enumerate(
                        measurements_by_config["offline"]
                    ):
                        raw_states_calc_args.append(
                            (num_measurement, measurement, penalty, opt_model, opt_jump)
                        )

                    raw_states_list = [None] * len(measurements_by_config["offline"])
                    raw_states_res = calc_raw_states(
                        raw_states_calc_args, opt_num_processes
                    )
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
                        print_info(
                            "The average standard deviation for the newly found states in "
                            + "measurement No. "
                            + str(num_measurement)
                            + " is "
                            + str(avg_std)
                        )
                        print_info("That is a reduction of " + str(change_avg_std))
                        # l_signal = measurements_by_config['offline'][num_measurement]['uW']
                        # l_bkpts = [s[1] for s in raw_states]
                        # fig, ax = rpt.display(np.array(l_signal), l_bkpts)
                        # plt.show()
                    print_info("Finished raw_states calculation.")
                    num_states_array = [int()] * len(raw_states_list)
                    i = 0
                    for i, x in enumerate(raw_states_list):
                        num_states_array[i] = len(x)
                    avg_num_states = np.mean(num_states_array)
                    num_states_dev = np.std(num_states_array)
                    print_info(
                        "On average "
                        + str(avg_num_states)
                        + " States have been found. The standard deviation"
                        + " is "
                        + str(num_states_dev)
                    )
                    # TODO: MAGIC NUMBER
                    if num_states_dev > 1:
                        print_warning(
                            "The number of states varies strongly across measurements."
                            " Consider choosing a larger range for penalty detection."
                            " It is also possible, that the processed data is not accurate"
                            " enough to produce proper results."
                        )
                        time.sleep(5)
                    # TODO: Wie bekomme ich da jetzt raus, was die Wahrheit ist?
                    #   Einfach Durchschnitt nehmen?
                    #   Preliminary decision: Further on only use the traces, which have the most
                    #   frequent state count
                    counts = np.bincount(num_states_array)
                    num_raw_states = np.argmax(counts)
                    print_info(
                        "Choose " + str(num_raw_states) + " as number of raw_states."
                    )
                    if num_raw_states == 1:
                        print_info(
                            "Upon further inspection it is clear that no refinement is necessary."
                            " The macromodel is usable for this configuration."
                        )
                        continue
                    # iterate through all found breakpoints and determine start and end points as well
                    # as power consumption
                    num_measurements = len(raw_states_list)
                    states_duration_list = [list()] * num_raw_states
                    states_consumption_list = [list()] * num_raw_states
                    for num_elem, _ in enumerate(states_duration_list):
                        states_duration_list[num_elem] = [0] * num_measurements
                        states_consumption_list[num_elem] = [0] * num_measurements
                    num_used_measurements = 0
                    for num_measurement, raw_states in enumerate(raw_states_list):
                        if len(raw_states) == num_raw_states:
                            num_used_measurements = num_used_measurements + 1
                            for num_state, s in enumerate(raw_states):
                                states_duration_list[num_state][num_measurement] = (
                                    s[1] - s[0]
                                )
                                states_consumption_list[num_state][num_measurement] = s[
                                    2
                                ]
                            # calced_state = (start_time, end_time, mean_power, std_dev)
                            # for num_state, s in enumerate(raw_states):
                            #     state_duration = s[1] - s[0]
                            #     state_consumption = s[2]
                            #     states_duration_list[num_state] = \
                            #         states_duration_list[num_state] + state_duration
                            #     states_consumption_list[num_state] = \
                            #         states_consumption_list[num_state] + state_consumption
                        else:
                            print_info(
                                "Discarding measurement No. "
                                + str(num_measurement)
                                + " because it did not recognize the number of "
                                "raw_states correctly."
                            )
                        # l_signal = measurements_by_config['offline'][num_measurement]['uW']
                        # l_bkpts = [s[1] for s in raw_states]
                        # fig, ax = rpt.display(np.array(l_signal), l_bkpts)
                        # plt.show()
                    # for i, x in enumerate(states_duration_list):
                    #     states_duration_list[i] = x / num_used_measurements
                    # for i, x in enumerate(states_consumption_list):
                    #     states_consumption_list[i] = x / num_used_measurements
                    if num_used_measurements != len(raw_states_list):
                        if num_used_measurements / len(raw_states_list) <= 0.5:
                            print_warning(
                                "Only used "
                                + str(num_used_measurements)
                                + "/"
                                + str(len(raw_states_list))
                                + " Measurements for refinement. "
                                + "Others did not recognize number of states correctly."
                                + "\nYou should verify the integrity of the measurements."
                            )
                        else:
                            print_info(
                                "Used "
                                + str(num_used_measurements)
                                + "/"
                                + str(len(raw_states_list))
                                + " Measurements for refinement."
                                + " Others did not recognize number of states correctly."
                            )
                        num_used_measurements = i
                    else:
                        print_info("Used all available measurements.")

                    state_durations_by_config.append((num_config, states_duration_list))
                    state_consumptions_by_config.append(
                        (num_config, states_consumption_list)
                    )

            # combine all state durations and consumptions to parametrized model
            if len(state_durations_by_config) == 0:
                print(
                    "No refinement necessary for this state. The macromodel is usable."
                )
                sys.exit(1)
            if len(state_durations_by_config) / len(configurations) > 1 / 2 and len(
                state_durations_by_config
            ) != len(configurations):
                print_warning(
                    "Some measurements(>50%) need to be refined, however that is not true for"
                    " all measurements. This hints a correlation between the structure of"
                    " the underlying automaton and parameters. Only the ones which need to"
                    " be refined will be refined. THE RESULT WILL NOT ACCURATELY DEPICT "
                    " THE REAL WORLD."
                )
                not_accurate = True
            if len(state_durations_by_config) / len(configurations) < 1 / 2:
                print_warning(
                    "Some measurements(<50%) need to be refined, however that is not true for"
                    " all measurements. This hints a correlation between the structure of"
                    " the underlying automaton and parameters. Or a poor quality of measurements."
                    " No Refinement will be done."
                )
                sys.exit(-1)
            # this is only necessary because at this state only linear automatons can be modeled.
            num_states_array = [int()] * len(state_consumptions_by_config)
            for i, (_, states_consumption_list) in enumerate(
                state_consumptions_by_config
            ):
                num_states_array[i] = len(states_consumption_list)
            counts = np.bincount(num_states_array)
            num_raw_states = np.argmax(counts)
            usable_configs = len(state_consumptions_by_config)
            # param_list identical for each raw_state
            param_list = []
            param_names = configurations[0]["offline_aggregates"]["paramkeys"][0]
            print_info("param_names: " + str(param_names))
            for num_config, states_consumption_list in state_consumptions_by_config:
                if len(states_consumption_list) != num_raw_states:
                    print_warning(
                        "Config No."
                        + str(num_config)
                        + " not usable yet due to different "
                        + "number of states. This hints a correlation between parameters and "
                        + "the structure of the resulting automaton. This will be possibly"
                        + " supported in a future version of this tool. HOWEVER AT THE MOMENT"
                        " THIS WILL LEAD TO INACCURATE RESULTS!"
                    )
                    not_accurate = True
                    usable_configs = usable_configs - 1
                else:
                    param_list.extend(
                        configurations[num_config]["offline_aggregates"]["param"]
                    )
            print_info("param_list: " + str(param_list))

            if usable_configs == len(state_consumptions_by_config):
                print_info("All configs usable.")
            else:
                print_info("Using only " + str(usable_configs) + " Configs.")
            # build by_name
            by_name = {}
            usable_configs_2 = len(state_consumptions_by_config)
            for i in range(num_raw_states):
                consumptions_for_state = []
                durations_for_state = []
                for j, (_, states_consumption_list) in enumerate(
                    state_consumptions_by_config
                ):
                    if len(states_consumption_list) == num_raw_states:
                        consumptions_for_state.extend(states_consumption_list[i])
                        durations_for_state.extend(state_durations_by_config[j][1][i])
                    else:
                        not_accurate = True
                        usable_configs_2 = usable_configs_2 - 1
                if usable_configs_2 != usable_configs:
                    print_error(
                        "an zwei unterschiedlichen Stellen wurden unterschiedlich viele "
                        "Messungen rausgeworfen. Bei Janis beschweren."
                    )
                state_name = "state_" + str(i)
                state_dict = {
                    "param": param_list,
                    "power": consumptions_for_state,
                    "duration": durations_for_state,
                    "attributes": ["power", "duration"],
                    # Da kein "richtiger" Automat generiert wird, gibt es auch keine Transitionen
                    "isa": "state",
                }
                by_name[state_name] = state_dict
            by_param = by_name_to_by_param(by_name)
            if opt_cache_loc is not None:
                by_name_loc = os.path.join(opt_cache_loc, "by_name.txt")
                by_param_loc = os.path.join(opt_cache_loc, "by_param.txt")
                param_names_loc = os.path.join(opt_cache_loc, "param_names.txt")
                f = open(by_name_loc, "w")
                f.write(str(by_name))
                f.close()
                f = open(by_param_loc, "w")
                f.write(str(by_param))
                f.close()
                f = open(param_names_loc, "w")
                f.write(str(param_names))
                f.close()
        else:
            by_name_text = str(by_name_file.read())
            by_name = eval(by_name_text)
            by_param_text = str(by_param_file.read())
            by_param = eval(by_param_text)
            param_names_text = str(param_names_file.read())
            param_names = eval(param_names_text)

        # t = 0
        # last_pow = 0
        # for key in by_name.keys():
        #     end_t = t + np.mean(by_name[key]["duration"])
        #     power = np.mean(by_name[key]["power"])
        #     plt.vlines(t, min(last_pow, power), max(last_pow, power))
        #     plt.hlines(power, t, end_t)
        #     t = end_t
        #     last_pow = power
        # plt.show()
        stats = parameters.ParamStats(by_name, by_param, param_names, dict())
        paramfit = ParallelParamFit(by_param)
        for state_name in by_name.keys():
            for num_param, param_name in enumerate(param_names):
                if stats.depends_on_param(state_name, "power", param_name):
                    paramfit.enqueue(state_name, "power", num_param, param_name)
                if stats.depends_on_param(state_name, "duration", param_name):
                    paramfit.enqueue(state_name, "duration", num_param, param_name)
                print_info(
                    "State "
                    + state_name
                    + "s power depends on param "
                    + param_name
                    + ":"
                    + str(stats.depends_on_param(state_name, "power", param_name))
                )
                print_info(
                    "State "
                    + state_name
                    + "s duration depends on param "
                    + param_name
                    + ":"
                    + str(stats.depends_on_param(state_name, "duration", param_name))
                )
        paramfit.fit()
        fit_res_dur_dict = {}
        fit_res_pow_dict = {}
        # fit functions and check if successful
        for state_name in by_name.keys():
            fit_power = paramfit.get_result(state_name, "power")
            fit_duration = paramfit.get_result(state_name, "duration")
            combined_fit_power = analytic.function_powerset(fit_power, param_names, 0)
            combined_fit_duration = analytic.function_powerset(
                fit_duration, param_names, 0
            )
            combined_fit_power.fit(by_param, state_name, "power")
            if not combined_fit_power.fit_success:
                print_warning(
                    "Fitting(power) for state " + state_name + " was not succesful!"
                )
            combined_fit_duration.fit(by_param, state_name, "duration")
            if not combined_fit_duration.fit_success:
                print_warning(
                    "Fitting(duration) for state " + state_name + " was not succesful!"
                )
            fit_res_pow_dict[state_name] = combined_fit_power
            fit_res_dur_dict[state_name] = combined_fit_duration
        # only raw_states with the same number of function parameters can be similar
        num_param_pow_dict = {}
        num_param_dur_dict = {}
        # print found substate_results
        for state_name in by_name.keys():
            model_function = str(fit_res_pow_dict[state_name].model_function)
            model_args = fit_res_pow_dict[state_name].model_args
            num_param_pow_dict[state_name] = len(model_args)
            for num_arg, arg in enumerate(model_args):
                replace_string = "regression_arg(" + str(num_arg) + ")"
                model_function = model_function.replace(replace_string, str(arg))
            print_info("Power-Function for state " + state_name + ": " + model_function)
        for state_name in by_name.keys():
            model_function = str(fit_res_dur_dict[state_name].model_function)
            model_args = fit_res_dur_dict[state_name].model_args
            num_param_dur_dict[state_name] = len(model_args)
            for num_arg, arg in enumerate(model_args):
                replace_string = "regression_arg(" + str(num_arg) + ")"
                model_function = model_function.replace(replace_string, str(arg))
            print_info(
                "Duration-Function for state " + state_name + ": " + model_function
            )
        # sort states in buckets for clustering
        similar_raw_state_buckets = {}
        for state_name in by_name.keys():
            pow_model_function = str(fit_res_pow_dict[state_name].model_function)
            dur_model_function = str(fit_res_dur_dict[state_name].model_function)
            key_tuple = (pow_model_function, dur_model_function)
            if key_tuple not in similar_raw_state_buckets:
                similar_raw_state_buckets[key_tuple] = []
            similar_raw_state_buckets[key_tuple].append(state_name)

        # cluster for each Key-Tuple using the function parameters
        distinct_states = []
        for key_tuple in similar_raw_state_buckets.keys():
            print_info(
                "Key-Tuple "
                + str(key_tuple)
                + ": "
                + str(similar_raw_state_buckets[key_tuple])
            )
            similar_states = similar_raw_state_buckets[key_tuple]
            if len(similar_states) > 1:
                # only necessary to cluster if more than one raw_state has the same function
                # configuration
                # functions are identical -> num_params and used params are identical
                num_params = (
                    num_param_dur_dict[similar_states[0]]
                    + num_param_pow_dict[similar_states[0]]
                )
                values_to_cluster = np.zeros((len(similar_states), num_params))
                for num_state, state_name in enumerate(similar_states):
                    dur_params = fit_res_dur_dict[state_name].model_args
                    pow_params = fit_res_pow_dict[state_name].model_args
                    j = 0
                    for param in pow_params:
                        values_to_cluster[num_state][j] = param
                        j = j + 1
                    for param in dur_params:
                        values_to_cluster[num_state][j] = param
                        j = j + 1
                normed_vals_to_cluster = norm_values_to_cluster(values_to_cluster)
                cluster = AgglomerativeClustering(
                    n_clusters=None,
                    compute_full_tree=True,
                    affinity="euclidean",
                    linkage="ward",
                    # TODO: Magic Number. Beim Evaluieren finetunen
                    distance_threshold=1,
                )
                cluster.fit_predict(values_to_cluster)
                cluster_labels = cluster.labels_
                print_info("Cluster labels:\n" + str(cluster_labels))
                if cluster.n_clusters_ > 1:
                    # more than one distinct state found -> seperation of raw_states necessary
                    distinct_state_dict = {}
                    for num_state, label in enumerate(cluster_labels):
                        if label not in distinct_state_dict.keys():
                            distinct_state_dict[label] = []
                        distinct_state_dict[label].append(similar_states[num_state])
                    for distinct_state_key in distinct_state_dict.keys():
                        distinct_states.append(distinct_state_dict[distinct_state_key])
                else:
                    # all raw_states make up this state
                    distinct_states.append(similar_states)
            else:
                distinct_states.append(similar_states)
        for num_state, distinct_state in enumerate(distinct_states):
            print("State " + str(num_state) + ": " + str(distinct_state))
        num_raw_states = len(by_name.keys())
        resulting_sequence = [int] * num_raw_states
        for i in range(num_raw_states):
            # apply the projection from raw_states to states
            state_name = "state_" + str(i)
            state_num = get_state_num(state_name, distinct_states)
            if state_num == -1:
                print_error(
                    "Critical Error when creating the resulting sequence. raw_state state_"
                    + str(i)
                    + " could not be mapped to a state."
                )
                sys.exit(-1)
            resulting_sequence[i] = state_num
        print("Resulting sequence is: " + str(resulting_sequence))
        # if from_cache:
        #     print_warning(
        #         "YOU USED THE OPTION \"cache_dicts\". THIS IS FOR DEBUGGING PURPOSES ONLY!"
        #         "\nTHE SCRIPT WILL NOW STOP PREMATURELY,"
        #         "SINCE DATA FOR FURTHER COMPUTATION IS MISSING!")
        #     sys.exit(0)
        # parameterize all new states
        new_by_name = {}
        for num_state, distinct_state in enumerate(distinct_states):
            state_name = "State_" + str(num_state)
            consumptions_for_state = []
            durations_for_state = []
            param_list = []
            for raw_state in distinct_state:
                original_state_dict = by_name[raw_state]
                param_list.extend(original_state_dict["param"])
                consumptions_for_state.extend(original_state_dict["power"])
                durations_for_state.extend(original_state_dict["duration"])
            new_state_dict = {
                "param": param_list,
                "power": consumptions_for_state,
                "duration": durations_for_state,
                "attributes": ["power", "duration"],
                # Da kein richtiger Automat generiert wird, gibt es auch keine Transitionen
                "isa": "state",
            }
            new_by_name[state_name] = new_state_dict
        new_by_param = by_name_to_by_param(new_by_name)
        new_stats = parameters.ParamStats(
            new_by_name, new_by_param, param_names, dict()
        )
        new_paramfit = ParallelParamFit(new_by_param)
        for state_name in new_by_name.keys():
            for num_param, param_name in enumerate(param_names):
                if new_stats.depends_on_param(state_name, "power", param_name):
                    new_paramfit.enqueue(state_name, "power", num_param, param_name)
                if new_stats.depends_on_param(state_name, "duration", param_name):
                    new_paramfit.enqueue(state_name, "duration", num_param, param_name)
                print_info(
                    "State "
                    + state_name
                    + "s power depends on param "
                    + param_name
                    + ":"
                    + str(new_stats.depends_on_param(state_name, "power", param_name))
                )
                print_info(
                    "State "
                    + state_name
                    + "s duration depends on param "
                    + param_name
                    + ":"
                    + str(
                        new_stats.depends_on_param(state_name, "duration", param_name)
                    )
                )
        new_paramfit.fit()
        new_fit_res_dur_dict = {}
        new_fit_res_pow_dict = {}
        for state_name in new_by_name.keys():
            fit_power = new_paramfit.get_result(state_name, "power")
            fit_duration = new_paramfit.get_result(state_name, "duration")
            combined_fit_power = analytic.function_powerset(fit_power, param_names, 0)
            combined_fit_duration = analytic.function_powerset(
                fit_duration, param_names, 0
            )
            combined_fit_power.fit(new_by_param, state_name, "power")
            if not combined_fit_power.fit_success:
                print_warning(
                    "Fitting(power) for state " + state_name + " was not succesful!"
                )
            combined_fit_duration.fit(new_by_param, state_name, "duration")
            if not combined_fit_duration.fit_success:
                print_warning(
                    "Fitting(duration) for state " + state_name + " was not succesful!"
                )
            new_fit_res_pow_dict[state_name] = combined_fit_power
            new_fit_res_dur_dict[state_name] = combined_fit_duration
        # output results
        result_loc = os.path.join(filepath, "result" + big_state_name + ".txt")
        with open(result_loc, "w") as f:
            f.write("Resulting Sequence: " + str(resulting_sequence))
            f.write("\n\n")
            for state_name in new_by_name.keys():
                model_function = str(new_fit_res_pow_dict[state_name].model_function)
                model_args = new_fit_res_pow_dict[state_name].model_args
                for num_arg, arg in enumerate(model_args):
                    replace_string = "regression_arg(" + str(num_arg) + ")"
                    model_function = model_function.replace(replace_string, str(arg))
                print("Power-Function for state " + state_name + ": " + model_function)
                f.write(
                    "Power-Function for state "
                    + state_name
                    + ": "
                    + model_function
                    + "\n"
                )
            f.write("\n\n")
            for state_name in new_by_name.keys():
                model_function = str(new_fit_res_dur_dict[state_name].model_function)
                model_args = new_fit_res_dur_dict[state_name].model_args
                for num_arg, arg in enumerate(model_args):
                    replace_string = "regression_arg(" + str(num_arg) + ")"
                    model_function = model_function.replace(replace_string, str(arg))
                print(
                    "Duration-Function for state " + state_name + ": " + model_function
                )
                f.write(
                    "Duration-Function for state "
                    + state_name
                    + ": "
                    + model_function
                    + "\n"
                )
            if not_accurate:
                print_warning(
                    "THIS RESULT IS NOT ACCURATE. SEE WARNINGLOG TO GET A BETTER UNDERSTANDING"
                    " WHY."
                )
                f.write(
                    "THIS RESULT IS NOT ACCURATE. SEE WARNINGLOG TO GET A BETTER UNDERSTANDING"
                    " WHY."
                )

    #         Removed clustering at this point, since it provided too much difficulties
    #           at the current state. Clustering is still used, but at another point of execution.
    #           Now parametrization is done first. raw_states are grouped by their using a dict
    #           where the key is [power_function, duration_dunction]. Then all raw_states from
    #           each bucket are clustered by their parameters
    #         i = 0
    #         cluster_labels_list = []
    #         num_cluster_list = []
    #         for num_trace, raw_states in enumerate(raw_states_list):
    #             # iterate through raw states from measurements
    #             if len(raw_states) == num_raw_states:
    #                 # build array with power values to cluster these
    #                 value_to_cluster = np.zeros((num_raw_states, 2))
    #                 j = 0
    #                 for s in raw_states:
    #                     value_to_cluster[j][0] = s[2]
    #                     value_to_cluster[j][1] = 0
    #                     j = j + 1
    #                 # linked = linkage(value_to_cluster, 'single')
    #                 #
    #                 # labelList = range(1, 11)
    #                 #
    #                 # plt.figure(figsize=(10, 7))
    #                 # dendrogram(linked,
    #                 #            orientation='top',
    #                 #            distance_sort='descending',
    #                 #            show_leaf_counts=True)
    #                 # plt.show()
    #                 # TODO: Automatic detection of number of clusters. Aktuell noch MAGIC NUMBER
    #                 #   im distance_threshold
    #                 cluster = AgglomerativeClustering(n_clusters=None, compute_full_tree=True,
    #                                                   affinity='euclidean',
    #                                                   linkage='ward',
    #                                                   distance_threshold=opt_refinement_thresh * 100)
    #                 # cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean',
    #                 #                                   linkage='ward')
    #                 cluster.fit_predict(value_to_cluster)
    #                 # print_info("Cluster labels:\n" + str(cluster.labels_))
    #                 # plt.scatter(value_to_cluster[:, 0], value_to_cluster[:, 1], c=cluster.labels_, cmap='rainbow')
    #                 # plt.show()
    #                 cluster_labels_list.append((num_trace, cluster.labels_))
    #                 num_cluster_list.append((num_trace, cluster.n_clusters_))
    #                 i = i + 1
    #             else:
    #                 print_info("Discarding measurement No. " + str(num_trace) + " because it "
    #                            + "did not recognize the number of raw_states correctly.")
    #         num_used_measurements = len(raw_states_list)
    #         if i != len(raw_states_list):
    #             if i / len(raw_states_list) <= 0.5:
    #                 print_warning("Only used " + str(i) + "/" + str(len(raw_states_list))
    #                               + " Measurements for refinement. "
    #                                 "Others did not recognize number of states correctly."
    #                                 "\nYou should verify the integrity of the measurements.")
    #             else:
    #                 print_info("Used " + str(i) + "/" + str(len(raw_states_list))
    #                            + " Measurements for refinement. "
    #                              "Others did not recognize number of states correctly.")
    #             num_used_measurements = i
    #             # TODO: DEBUG Kram
    #             sys.exit(0)
    #         else:
    #             print_info("Used all available measurements.")
    #
    #         num_states = np.argmax(np.bincount([elem[1] for elem in num_cluster_list]))
    #         avg_per_state_list = [None] * len(cluster_labels_list)
    #         used_clusters = 0
    #         for number, (num_trace, labels) in enumerate(cluster_labels_list):
    #             if num_cluster_list[number][1] == num_states:
    #                 avg_per_state = [0] * num_states
    #                 count_per_state = [0] * num_states
    #                 raw_states = raw_states_list[num_trace]
    #                 for num_label, label in enumerate(labels):
    #                     count_per_state[label] = count_per_state[label] + 1
    #                     avg_per_state[label] = avg_per_state[label] + raw_states[num_label][2]
    #                 for i, _ in enumerate(avg_per_state):
    #                     avg_per_state[i] = avg_per_state[i] / count_per_state[i]
    #                 avg_per_state_list[number] = avg_per_state
    #                 used_clusters = used_clusters + 1
    #             else:
    #                 # hopefully this does not happen regularly
    #                 print_info("Discarding measurement " + str(number)
    #                            + " because the clustering yielded not matching results.")
    #                 num_used_measurements = num_used_measurements - 1
    #         if num_used_measurements == 0:
    #             print_error("Something went terribly wrong. Discarded all measurements.")
    #             # continue
    #             sys.exit(-1)
    #         # flattend version for clustering:
    #         values_to_cluster = np.zeros((num_states * used_clusters, 2))
    #         index = 0
    #         for avg_per_state in avg_per_state_list:
    #             if avg_per_state is not None:
    #                 for avg in avg_per_state:
    #                     values_to_cluster[index][0] = avg
    #                     values_to_cluster[index][1] = 0
    #                     index = index + 1
    #         # plt.scatter(values_to_cluster[:, 0], values_to_cluster[:, 1])
    #         # plt.show()
    #         cluster = AgglomerativeClustering(n_clusters=num_states)
    #         cluster.fit_predict(values_to_cluster)
    #         # Aktuell hast du hier ein plattes Array mit labels. Jetzt also das wieder auf die
    #         # ursprünglichen Labels abbilden, die dann verändern mit den hier gefundenen Labels.
    #         # Alle identischen Zustände haben identische Labels. Dann vllt bei resulting
    #         # sequence ausgeben, wie groß die übereinstimmung bei der Stateabfolge ist.
    #         new_labels_list = []
    #         new_labels = []
    #         i = 0
    #         for label in cluster.labels_:
    #             new_labels.append(label)
    #             i = i + 1
    #             if i == num_states:
    #                 new_labels_list.append(new_labels)
    #                 new_labels = []
    #                 i = 0
    #         # only the selected measurements are present in new_labels.
    #         # new_labels_index should not be incremented, if not selected_measurement is skipped
    #         new_labels_index = 0
    #         # cluster_labels_list contains all measurements -> if measurement is skipped
    #         # still increment the index
    #         index = 0
    #         for elem in avg_per_state_list:
    #             if elem is not None:
    #                 for number, label in enumerate(cluster_labels_list[index][1]):
    #                     cluster_labels_list[index][1][number] = \
    #                         new_labels_list[new_labels_index][label]
    #                 new_labels_index = new_labels_index + 1
    #             else:
    #                 # override not selected measurement labels to avoid choosing the wrong ones.
    #                 for number, label in enumerate(cluster_labels_list[index][1]):
    #                     cluster_labels_list[index][1][number] = -1
    #             index = index + 1
    #         resulting_sequence = [None] * num_raw_states
    #         i = 0
    #         confidence = 0
    #         for x in resulting_sequence:
    #             j = 0
    #             test_list = []
    #             for arr in [elem[1] for elem in cluster_labels_list]:
    #                 if num_cluster_list[j][1] != num_states:
    #                     j = j + 1
    #                 else:
    #                     if -1 in arr:
    #                         print_error("Bei Janis beschweren! Fehler beim Umbenennen der"
    #                                     " Zustände wahrscheinlich.")
    #                         sys.exit(-1)
    #                     test_list.append(arr[i])
    #                     j = j + 1
    #             bincount = np.bincount(test_list)
    #             resulting_sequence[i] = np.argmax(bincount)
    #             confidence = confidence + bincount[resulting_sequence[i]] / np.sum(bincount)
    #             i = i + 1
    #         confidence = confidence / len(resulting_sequence)
    #         print_info("Confidence of resulting sequence is " + str(confidence)
    #                    + " while using " + str(num_used_measurements) + "/"
    #                    + str(len(raw_states_list)) + " measurements.")
    #         #print(resulting_sequence)
    #         resulting_sequence_list.append((num_config, resulting_sequence))
    # # TODO: Was jetzt? Hier habe ich jetzt pro Konfiguration eine Zustandsfolge. Daraus Automat
    # #   erzeugen. Aber wie? Oder erst parametrisieren? Eigentlich brauche ich vorher die
    # #   Loops. Wie erkenne ich die? Es können beliebig viele Loops an beliebigen Stellen
    # #   auftreten.
    # # TODO: Die Zustandsfolgen werden sich nicht einfach in isomorphe(-einzelne wegfallende bzw.
    # #   hinzukommende Zustände) Automaten übersetzten lassen. Basiert alles auf dem Problem:
    # #   wie erkenne ich, dass zwei Zustände die selben sind und nicht nur einfach eine ähnliche
    # #   Leistungsaufnahme haben?! Vllt Zustände 2D clustern? 1Dim = Leistungsaufnahme,
    # #   2Dim=Dauer? Zumindest innerhalb einer Paramkonfiguration sollte sich die Dauer eines
    # #   Zustands ja nicht mehr ändern. Kann sicherlich immernoch Falschclustering erzeugen...
    # for num_config, sequence in resulting_sequence_list:
    #     print_info("NO. config:" + str(num_config))
    #     print_info(sequence)
    #
    #
    #
    #

    elif ".tar" in opt_filename:
        # open with dfatool
        raw_data_args = list()
        raw_data_args.append(opt_filename)
        raw_data = RawData(raw_data_args, with_traces=True)
        print_info(
            "Preprocessing file. Depending on its size, this could take a while."
        )
        preprocessed_data = raw_data.get_preprocessed_data()
        print_info("File fully preprocessed")
        # TODO: Mal schauen, wie ich das mache. Erstmal nur mit json. Ist erstmal raus. Wird nicht
        #   umgesetzt.
        print_error(
            "Not implemented yet. Please generate .json files first with dfatool and use"
            " those."
        )
    else:
        print_error("Unknown dataformat")
        sys.exit(-1)
