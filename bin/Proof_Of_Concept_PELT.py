import json
import time
import sys
import getopt
import re
from multiprocessing import Pool, Manager
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import ruptures as rpt
import numpy as np
from dfatool.dfatool import RawData


# from scipy.cluster.hierarchy import dendrogram, linkage # for graphical display

# py bin\Proof_Of_Concept_PELT.py --filename="..\data\TX.json" --jump=1 --pen_override=10 --refinement_thresh=100


def plot_data_from_json(filename, trace_num, x_axis, y_axis):
    with open(filename, 'r') as file:
        tx_data = json.load(file)
    print(tx_data[trace_num]['parameter'])
    plt.plot(tx_data[trace_num]['offline'][0]['uW'])
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
    plt.hlines(average, 0, lens, color='red')
    plt.vlines(len(signal1), 0, 100000, color='red', linestyles='dashed')
    plt.plot(signal2)
    average = np.mean(signal2)
    plt.hlines(average, 0, lens, color='green')
    plt.vlines(len(signal2), 0, 100000, color='green', linestyles='dashed')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()


def get_bkps(algo, pen, q):
    res = pen, len(algo.predict(pen=pen))
    q.put(pen)
    return res


def find_knee_point(data_x, data_y, S=1.0, curve='convex', direction='decreasing'):
    kneedle = KneeLocator(data_x, data_y, S=S, curve=curve, direction=direction)
    kneepoint = (kneedle.knee, kneedle.knee_y)
    return kneepoint


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
    sys.exit()


def calculate_penalty_value(signal, model="l1", jump=5, min_dist=2, range_min=0, range_max=50,
                            num_processes=8, refresh_delay=1, refresh_thresh=5, S=1.0,
                            pen_modifier=None):
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
            args.append((algo, i, q))

        print_info("starting kneepoint calculation.")
        # init Pool with num_proesses
        with Pool(num_processes) as p:
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
        # scipy.find_peaks() does not find plateaus if they extend through the end of the data.
        # to counter that, add one extremely large value to the right side of the data
        # after negating it is extremely small -> Almost certainly smaller than the
        # found plateau therefore the plateau does not extend through the border -> scipy.find_peaks
        # finds it. Choose value from within that plateau.
        # fitted_bkps_val.append(100000000)
        # TODO: Approaching over find_peaks might not work if the initial decrease step to the
        #   "correct" number of changepoints and additional decrease steps e.g. underfitting
        #   take place within the given penalty interval. find_peak only finds plateaus
        #   of peaks. If the number of chpts decreases after the wanted plateau the condition
        #   for local peaks is not satisfied anymore. Therefore this approach will only work
        #   if the plateau extends over the right border of the penalty interval.
        # peaks, peak_plateaus = find_peaks(- np.array(fitted_bkps_val), plateau_size=1)
        # Since the data is monotonously decreasing only one plateau can be found.

        # assuming the plateau is constant
        start_index = -1
        end_index = -1
        longest_start = -1
        longest_end = -1
        prev_val = -1
        for i, num_bkpts in enumerate(fitted_bkps_val[knee[0]:]):
            if num_bkpts != prev_val:
                end_index = i - 1
                if end_index - start_index > longest_end - longest_start:
                    # currently found sequence is the longest found yet
                    longest_start = start_index
                    longest_end = end_index
                start_index = i
            if i == len(fitted_bkps_val[knee[0]:]) - 1:
                # end sequence with last value
                end_index = i
                if end_index - start_index > longest_end - longest_start:
                    # last found sequence is the longest found yet
                    longest_start = start_index
                    longest_end = end_index
                start_index = i
            prev_val = num_bkpts
        # plt.xlabel('Penalty')
        # plt.ylabel('Number of Changepoints')
        # plt.plot(pen_val, fitted_bkps_val)
        # plt.vlines(longest_start + knee[0], 0, max(fitted_bkps_val), linestyles='dashed')
        # plt.vlines(longest_end + knee[0], 0, max(fitted_bkps_val), linestyles='dashed')
        # plt.show()
        # choosing pen from plateau
        mid_of_plat = longest_start + (longest_end - longest_start) // 2
        knee = (mid_of_plat + knee[0], fitted_bkps_val[mid_of_plat + knee[0]])

        # modify knee according to options. Defaults to 1 * knee
        knee = (knee[0] * pen_modifier, knee[1])

    else:
        # range_min == range_max. has the same effect as pen_override
        knee = (range_min, None)
    print_info(str(knee[0]) + " has been selected as kneepoint.")
    if knee[0] is not None:
        return knee

    print_error("With the current thresh-hold S=" + str(S)
                + " it is not possible to select a penalty value.")
    sys.exit()


# very short benchmark yielded approx. 1/3 of speed compared to solution with sorting
# def needs_refinement_no_sort(signal, mean, thresh):
#     # linear search for the top 10%/ bottom 10%
#     # should be sufficient
#     length_of_signal = len(signal)
#     percentile_size = int()
#     percentile_size = length_of_signal // 100
#     upper_percentile = [None] * percentile_size
#     lower_percentile = [None] * percentile_size
#     fill_index_upper = percentile_size - 1
#     fill_index_lower = percentile_size - 1
#     index_smallest_val = fill_index_upper
#     index_largest_val = fill_index_lower
#
#     for x in signal:
#         if x > mean:
#             # will be in upper percentile
#             if fill_index_upper >= 0:
#                 upper_percentile[fill_index_upper] = x
#                 if x < upper_percentile[index_smallest_val]:
#                     index_smallest_val = fill_index_upper
#                 fill_index_upper = fill_index_upper - 1
#                 continue
#
#             if x > upper_percentile[index_smallest_val]:
#                 # replace smallest val. Find next smallest val
#                 upper_percentile[index_smallest_val] = x
#                 index_smallest_val = 0
#                 i = 0
#                 for y in upper_percentile:
#                     if upper_percentile[i] < upper_percentile[index_smallest_val]:
#                         index_smallest_val = i
#                     i = i + 1
#
#         else:
#             if fill_index_lower >= 0:
#                 lower_percentile[fill_index_lower] = x
#                 if x > lower_percentile[index_largest_val]:
#                     index_largest_val = fill_index_upper
#                 fill_index_lower = fill_index_lower - 1
#                 continue
#             if x < lower_percentile[index_largest_val]:
#                 # replace smallest val. Find next smallest val
#                 lower_percentile[index_largest_val] = x
#                 index_largest_val = 0
#                 i = 0
#                 for y in lower_percentile:
#                     if lower_percentile[i] > lower_percentile[index_largest_val]:
#                         index_largest_val = i
#                     i = i + 1
#
#     # should have the percentiles
#     lower_percentile_mean = np.mean(lower_percentile)
#     upper_percentile_mean = np.mean(upper_percentile)
#     dist = mean - lower_percentile_mean
#     if dist > thresh:
#         return True
#     dist = upper_percentile_mean - mean
#     if dist > thresh:
#         return True
#     return False


# raw_states_calc_args.append((num_measurement, measurement, penalty, opt_model
#                                                  , opt_jump))
def calc_raw_states_func(num_trace, measurement, penalty, model, jump):
    signal = np.array(measurement['uW'])
    normed_signal = norm_signal(signal)
    bkpts = calc_pelt(normed_signal, penalty, model=model, jump=jump)
    calced_states = list()
    start_time = 0
    end_time = 0
    for bkpt in bkpts:
        # start_time of state is end_time of previous one
        # (Transitions are instantaneous)
        start_time = end_time
        end_time = bkpt
        power_vals = signal[start_time: end_time]
        mean_power = np.mean(power_vals)
        std_dev = np.std(power_vals)
        calced_state = (start_time, end_time, mean_power, std_dev)
        calced_states.append(calced_state)
    num = 0
    new_avg_std = 0
    for s in calced_states:
        # print_info("State " + str(num) + " starts at t=" + str(s[0])
        #            + " and ends at t=" + str(s[1])
        #            + " while using " + str(s[2])
        #            + "uW with  sigma=" + str(s[3]))
        num = num + 1
        new_avg_std = new_avg_std + s[3]
    new_avg_std = new_avg_std / len(calced_states)
    change_avg_std = measurement['uW_std'] - new_avg_std
    # print_info("The average standard deviation for the newly found states is "
    #            + str(new_avg_std))
    # print_info("That is a reduction of " + str(change_avg_std))
    return num_trace, calced_states, new_avg_std, change_avg_std


def calc_raw_states(arg_list, num_processes=8):
    m = Manager()
    with Pool(processes=num_processes) as p:
        # collect results from pool
        result = p.starmap(calc_raw_states_func, arg_list)
    return result


# Very short benchmark yielded approx. 3 times the speed of solution not using sort
# TODO: Decide whether median is really the better baseline than mean
def needs_refinement(signal, thresh):
    sorted_signal = sorted(signal)
    length_of_signal = len(signal)
    percentile_size = int()
    percentile_size = length_of_signal // 100
    lower_percentile = sorted_signal[0:percentile_size]
    upper_percentile = sorted_signal[length_of_signal - percentile_size: length_of_signal]
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


def print_info(str_to_prt):
    str_lst = str_to_prt.split(sep='\n')
    for str_prt in str_lst:
        print("[INFO]" + str_prt)


def print_warning(str_to_prt):
    str_lst = str_to_prt.split(sep='\n')
    for str_prt in str_lst:
        print("[WARNING]" + str_prt)


def print_error(str_to_prt):
    str_lst = str_to_prt.split(sep='\n')
    for str_prt in str_lst:
        print("[ERROR]" + str_prt, file=sys.stderr)


def norm_signal(signal):
    # TODO: maybe refine normalisation of signal
    normed_signal = np.zeros(shape=len(signal))
    for i, signal_i in enumerate(signal):
        normed_signal[i] = signal_i / 1000
    return normed_signal


if __name__ == '__main__':
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
    )
    opt_filename = None
    opt_verbose = False
    opt_model = None
    opt_jump = None
    opt_min_dist = None
    opt_range_min = None
    opt_range_max = None
    opt_num_processes = None
    opt_refresh_delay = None
    opt_refresh_thresh = None
    opt_S = None
    opt_pen_override = None
    opt_pen_modifier = None
    opt_plotting = False
    opt_refinement_thresh = None
    try:
        raw_opts, args = getopt.getopt(sys.argv[1:], "", optspec.split(" "))

        for option, parameter in raw_opts:
            optname = re.sub(r"^--", "", option)
            opt[optname] = parameter

        if 'filename' not in opt:
            print_error("No file specified!")
            sys.exit(2)
        else:
            opt_filename = opt['filename']
        if 'v' in opt:
            opt_verbose = True
            opt_plotting = True
        if 'model' in opt:
            opt_model = opt['model']
        if 'jump' in opt:
            try:
                opt_jump = int(opt['jump'])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(2)
        if 'min_dist' in opt:
            try:
                opt_min_dist = int(opt['min_dist'])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(2)
        if 'range_min' in opt:
            try:
                opt_range_min = int(opt['range_min'])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(2)
        if 'range_max' in opt:
            try:
                opt_range_max = int(opt['range_max'])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(2)
        if 'num_processes' in opt:
            try:
                opt_num_processes = int(opt['num_processes'])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(2)
        if 'refresh_delay' in opt:
            try:
                opt_refresh_delay = int(opt['refresh_delay'])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(2)
        if 'refresh_thresh' in opt:
            try:
                opt_refresh_thresh = int(opt['refresh_thresh'])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(2)
        if 'S' in opt:
            try:
                opt_S = float(opt['S'])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(2)
        if 'pen_override' in opt:
            try:
                opt_pen_override = int(opt['pen_override'])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(2)
        if 'pen_modifier' in opt:
            try:
                opt_pen_modifier = float(opt['pen_modifier'])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(2)
        if 'refinement_thresh' in opt:
            try:
                opt_refinement_thresh = int(opt['refinement_thresh'])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(2)
    except getopt.GetoptError as err:
        print(err, file=sys.stderr)
        sys.exit(2)

    # OPENING DATA
    if ".json" in opt_filename:
        # open file with trace data from json
        print_info(
            " Will only refine the state which is present in " + opt_filename + " if necessary.")
        with open(opt_filename, 'r') as f:
            states = json.load(f)
        # loop through all traces check if refinement is necessary
        print_info("Checking if refinement is necessary...")
        for measurements_by_state in states:
            # loop through all occurrences of the looked at state
            print_info("Looking at state '" + measurements_by_state['name'] + "' with params: "
                       + str(measurements_by_state['parameter']))
            refine = False
            for measurement in measurements_by_state['offline']:
                # loop through measurements of particular state
                # an check if state needs refinement
                signal = measurement['uW']
                # mean = measurement['uW_mean']
                # TODO: Decide if median is really the better baseline than mean
                if needs_refinement(signal, opt_refinement_thresh) and not refine:
                    print_info("Refinement is necessary!")
                    refine = True
            if not refine:
                print_info("No refinement necessary for state '" + measurements_by_state['name']
                           + "' with params: " + str(measurements_by_state['parameter']))
            else:
                # assume that all measurements of the same param configuration are fundamentally
                # similar -> calculate penalty for first measurement, use it for all
                if opt_pen_override is None:
                    signal = np.array(measurements_by_state['offline'][0]['uW'])
                    normed_signal = norm_signal(signal)
                    penalty = calculate_penalty_value(normed_signal, model=opt_model,
                                                      range_min=opt_range_min,
                                                      range_max=opt_range_max,
                                                      num_processes=opt_num_processes,
                                                      jump=opt_jump, S=opt_S,
                                                      pen_modifier=opt_pen_modifier)
                    penalty = penalty[0]
                else:
                    penalty = opt_pen_override
                # build arguments for parallel excecution
                print_info("Starting raw_states calculation.")
                raw_states_calc_args = []
                for num_measurement, measurement in enumerate(measurements_by_state['offline']):
                    raw_states_calc_args.append((num_measurement, measurement, penalty,
                                                 opt_model, opt_jump))

                raw_states_list = [None] * len(measurements_by_state['offline'])
                raw_states_res = calc_raw_states(raw_states_calc_args, opt_num_processes)
                # extracting result and putting it in correct order -> index of raw_states_list
                # entry still corresponds with index of measurement in measurements_by_states
                # -> If measurements are discarded the correct ones are easily recognized
                for ret_val in raw_states_res:
                    num_trace = ret_val[0]
                    raw_states = ret_val[1]
                    avg_std = ret_val[2]
                    change_avg_std = ret_val[3]
                    # TODO: Wieso gibt mir meine IDE hier eine Warning aus? Der Index müsste doch
                    #   int sein oder nicht? Es scheint auch vernünftig zu klappen...
                    raw_states_list[num_trace] = raw_states
                    print_info("The average standard deviation for the newly found states in "
                               + "measurement No. " + str(num_trace) + " is " + str(avg_std))
                    print_info("That is a reduction of " + str(change_avg_std))
                print_info("Finished raw_states calculation.")
                num_states_array = [int()] * len(raw_states_list)
                i = 0
                for i, x in enumerate(raw_states_list):
                    num_states_array[i] = len(x)
                avg_num_states = np.mean(num_states_array)
                num_states_dev = np.std(num_states_array)
                print_info("On average " + str(avg_num_states)
                           + " States have been found. The standard deviation"
                           + " is " + str(num_states_dev))
                # TODO: MAGIC NUMBER
                if num_states_dev > 1:
                    print_warning("The number of states varies strongly across measurements."
                                  " Consider choosing a larger value for S or using the "
                                  "pen_modifier option.")
                    time.sleep(5)
                # TODO: Wie bekomme ich da jetzt raus, was die Wahrheit ist?
                # Einfach Durchschnitt nehmen?
                # Preliminary decision: Further on only use the traces, which have the most frequent state count
                counts = np.bincount(num_states_array)
                num_raw_states = np.argmax(counts)
                print_info("Choose " + str(num_raw_states) + " as number of raw_states.")
                i = 0
                cluster_labels_list = []
                num_cluster_list = []
                for num_trace, raw_states in enumerate(raw_states_list):
                    # iterate through raw states from measurements
                    if len(raw_states) == num_raw_states:
                        # build array with power values to cluster these
                        value_to_cluster = np.zeros((num_raw_states, 2))
                        j = 0
                        for s in raw_states:
                            value_to_cluster[j][0] = s[2]
                            value_to_cluster[j][1] = 0
                            j = j + 1
                        # linked = linkage(value_to_cluster, 'single')
                        #
                        # labelList = range(1, 11)
                        #
                        # plt.figure(figsize=(10, 7))
                        # dendrogram(linked,
                        #            orientation='top',
                        #            distance_sort='descending',
                        #            show_leaf_counts=True)
                        # plt.show()
                        # TODO: Automatic detection of number of clusters. Aktuell noch MAGIC NUMBER
                        #   im distance_threshold
                        cluster = AgglomerativeClustering(n_clusters=None, compute_full_tree=True,
                                                          affinity='euclidean',
                                                          linkage='ward',
                                                          distance_threshold=opt_refinement_thresh * 100)
                        # cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean',
                        #                                   linkage='ward')
                        cluster.fit_predict(value_to_cluster)
                        # print_info("Cluster labels:\n" + str(cluster.labels_))
                        # plt.scatter(value_to_cluster[:, 0], value_to_cluster[:, 1], c=cluster.labels_, cmap='rainbow')
                        # plt.show()
                        # TODO: Problem: Der Algorithmus nummeriert die Zustände nicht immer gleich... also bspw.:
                        # mal ist das tatsächliche Transmit mit 1 belabelt und mal mit 3
                        cluster_labels_list.append(cluster.labels_)
                        num_cluster_list.append(cluster.n_clusters_)
                        i = i + 1
                    else:
                        print_info("Discarding measurement No. " + str(num_trace) + " because it "
                                   + "did not recognize the number of raw_states correctly.")
                if i != len(raw_states_list):
                    if i / len(raw_states_list) <= 0.5:
                        print_warning("Only used " + str(i) + "/" + str(len(raw_states_list))
                                      + " Measurements for refinement. "
                                        "Others did not recognize number of states correctly."
                                        "\nYou should verify the integrity of the measurements.")
                    else:
                        print_info("Used " + str(i) + "/" + str(len(raw_states_list))
                                   + " Measurements for refinement. "
                                     "Others did not recognize number of states correctly.")
                    # TODO: DEBUG Kram
                    sys.exit()
                else:
                    print_info("Used all available measurements.")

                num_states = np.argmax(np.bincount(num_cluster_list))
                resulting_sequence = [None] * num_raw_states
                i = 0
                for x in resulting_sequence:
                    j = 0
                    test_list = []
                    for arr in cluster_labels_list:
                        if num_cluster_list[j] != num_states:
                            j = j + 1
                        else:
                            test_list.append(arr[i])
                            j = j + 1
                    resulting_sequence[i] = np.argmax(np.bincount(test_list))
                    i = i + 1
                print(resulting_sequence)

    elif ".tar" in opt_filename:
        # open with dfatool
        raw_data_args = list()
        raw_data_args.append(opt_filename)
        raw_data = RawData(
            raw_data_args, with_traces=True
        )
        print_info("Preprocessing file. Depending on its size, this could take a while.")
        preprocessed_data = raw_data.get_preprocessed_data()
        print_info("File fully preprocessed")

        # TODO: Mal schauen, wie ich das mache. Erstmal nur mit json
    else:
        print_error("Unknown dataformat")
        sys.exit(2)

    # print(tx_data[1]['parameter'])
    # # parse json to array for PELT
    # signal = np.array(tx_data[1]['offline'][0]['uW'])
    #
    # for i in range(0, len(signal)):
    #     signal[i] = signal[i]/1000
    # bkps = calc_pelt(signal, model=opt_model, range_max=opt_range_max, num_processes=opt_num_processes, jump=opt_jump, S=opt_S)
    # fig, ax = rpt.display(signal, bkps)
    # plt.xlabel('Time [us]')
    # plt.ylabel('Power [mW]')
    # plt.show()
