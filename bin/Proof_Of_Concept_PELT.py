def plot_data_from_json(filename, trace_num, xaxis, yaxis):
    import matplotlib.pyplot as plt
    import json
    with open(filename, 'r') as f:
        tx_data = json.load(f)
    print(tx_data[trace_num]['parameter'])
    plt.plot(tx_data[trace_num]['offline'][0]['uW'])
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.show()


def plot_data_vs_mean(signal, xaxis, yaxis):
    import matplotlib.pyplot as plt
    from statistics import mean
    plt.plot(signal)
    average = mean(signal)
    plt.hlines(average, 0, len(signal))
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.show()


def plot_data_vs_data_vs_means(signal1, signal2, xaxis, yaxis):
    import matplotlib.pyplot as plt
    from statistics import mean
    plt.plot(signal1)
    lens = max(len(signal1), len(signal2))
    average = mean(signal1)
    plt.hlines(average, 0, lens, color='red')
    plt.vlines(len(signal1), 0, 100000, color='red', linestyles='dashed')
    plt.plot(signal2)
    average = mean(signal2)
    plt.hlines(average, 0, lens, color='green')
    plt.vlines(len(signal2), 0, 100000, color='green', linestyles='dashed')
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.show()


def get_bkps(algo, pen, q):
    res = pen, len(algo.predict(pen=pen))
    q.put(pen)
    return res


def find_knee_point(data_x, data_y, S=1.0, curve='convex', direction='decreasing', plotting=False):
    from kneed import KneeLocator
    kneedle = KneeLocator(data_x, data_y, S=S, curve=curve, direction=direction)
    if plotting:
        kneedle.plot_knee()
    kneepoint = (kneedle.knee, kneedle.knee_y)
    return kneepoint


def calc_PELT(signal, model='l1', jump=5, min_dist=2, range_min=1, range_max=50, num_processes=8, refresh_delay=1,
              refresh_thresh=5, S=1.0, pen_override=None, plotting=False):
    import ruptures as rpt
    import time
    import matplotlib.pylab as plt
    from multiprocessing import Pool, Manager

    # default params in Function
    if model is None:
        model = 'l1'
    if jump is None:
        jump = 5
    if min_dist is None:
        min_dist = 2
    if range_min is None:
        range_min = 1
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
    if plotting is None:
        plotting = False

    # change point detection. best fit seemingly with l1. rbf prods. RuntimeErr for pen > 30
    # https://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/costs/index.html
    # model = "l1"   #"l1"  # "l2", "rbf"
    algo = rpt.Pelt(model=model, jump=jump, min_size=min_dist).fit(signal)

    ### CALC BKPS WITH DIFF PENALTYS
    if pen_override is None:
        # building args array for parallelizing
        args = []
        # for displaying progression
        m = Manager()
        q = m.Queue()

        for i in range(range_min, range_max):
            args.append((algo, i, q))

        print('starting kneepoint calculation')
        # init Pool with num_proesses
        with Pool(num_processes) as p:
            # collect results from pool
            result = p.starmap_async(get_bkps, args)
            # monitor loop
            last_percentage = -1
            percentage = -100  # Force display of 0%
            i = 0
            while True:
                if result.ready():
                    break
                else:
                    size = q.qsize()
                    last_percentage = percentage
                    percentage = round(size / (range_max - range_min) * 100, 2)
                    if percentage >= last_percentage + 2 or i >= refresh_thresh:
                        print('Current progress: ' + str(percentage) + '%')
                        i = 0
                    else:
                        i += 1
                    time.sleep(refresh_delay)
            res = result.get()

        # DECIDE WHICH PENALTY VALUE TO CHOOSE ACCORDING TO ELBOW/KNEE APPROACH
        # split x and y coords to pass to kneedle
        pen_val = [x[0] for x in res]
        fittet_bkps_val = [x[1] for x in res]
        # # plot to look at res

        knee = find_knee_point(pen_val, fittet_bkps_val, S=S, plotting=plotting)
        plt.xlabel('Penalty')
        plt.ylabel('Number of Changepoints')
        plt.plot(pen_val, fittet_bkps_val)
        plt.vlines(knee[0], 0, max(fittet_bkps_val), linestyles='dashed')
        print("knee: " + str(knee[0]))
        plt.show()
    else:
        # use forced pen value for plotting
        knee = (pen_override, None)


    #plt.plot(pen_val, fittet_bkps_val)
    if knee[0] is not None:
        bkps = algo.predict(pen=knee[0])
        if plotting:
            fig, ax = rpt.display(signal, bkps)
            plt.show()
        return bkps
    else:
        print('With the current thresh-hold S=' + str(S) + ' it is not possible to select a penalty value.')


# very short benchmark yielded approx. 1/3 of speed compared to solution with sorting
def needs_refinement_no_sort(signal, mean, thresh):
    # linear search for the top 10%/ bottom 10%
    # should be sufficient
    length_of_signal = len(signal)
    percentile_size = int()
    percentile_size = length_of_signal // 100
    upper_percentile = [None] * percentile_size
    lower_percentile = [None] * percentile_size
    fill_index_upper = percentile_size - 1
    fill_index_lower = percentile_size - 1
    index_smallest_val = fill_index_upper
    index_largest_val = fill_index_lower

    for x in signal:
        if x > mean:
            # will be in upper percentile
            if fill_index_upper >= 0:
                upper_percentile[fill_index_upper] = x
                if x < upper_percentile[index_smallest_val]:
                    index_smallest_val = fill_index_upper
                fill_index_upper = fill_index_upper - 1
                continue

            if x > upper_percentile[index_smallest_val]:
                # replace smallest val. Find next smallest val
                upper_percentile[index_smallest_val] = x
                index_smallest_val = 0
                i = 0
                for y in upper_percentile:
                    if upper_percentile[i] < upper_percentile[index_smallest_val]:
                        index_smallest_val = i
                    i = i + 1

        else:
            if fill_index_lower >= 0:
                lower_percentile[fill_index_lower] = x
                if x > lower_percentile[index_largest_val]:
                    index_largest_val = fill_index_upper
                fill_index_lower = fill_index_lower - 1
                continue
            if x < lower_percentile[index_largest_val]:
                # replace smallest val. Find next smallest val
                lower_percentile[index_largest_val] = x
                index_largest_val = 0
                i = 0
                for y in lower_percentile:
                    if lower_percentile[i] > lower_percentile[index_largest_val]:
                        index_largest_val = i
                    i = i + 1

    # should have the percentiles
    lower_percentile_mean = np.mean(lower_percentile)
    upper_percentile_mean = np.mean(upper_percentile)
    dist = mean - lower_percentile_mean
    if dist > thresh:
        return True
    dist = upper_percentile_mean - mean
    if dist > thresh:
        return True
    return False


# Very short benchmark yielded approx. 3 times the speed of solution not using sort
def needs_refinement_sort(signal, thresh):
    sorted_signal = sorted(signal)
    length_of_signal = len(signal)
    percentile_size = int()
    percentile_size = length_of_signal // 100
    lower_percentile = sorted_signal[0:percentile_size]
    upper_percentile = sorted_signal[length_of_signal - percentile_size : length_of_signal]
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


if __name__ == '__main__':
    import numpy as np
    import json
    import ruptures as rpt
    import matplotlib.pylab as plt
    import sys
    import getopt
    import re
    from dfatool.dfatool import RawData
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
    opt_plotting = False
    opt_refinement_thresh = None
    try:
        raw_opts, args = getopt.getopt(sys.argv[1:], "", optspec.split(" "))

        for option, parameter in raw_opts:
            optname = re.sub(r"^--", "", option)
            opt[optname] = parameter

        if 'filename' not in opt:
            print("No file specified!", file=sys.stderr)
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
        if 'refinement_thresh' in opt:
            try:
                opt_refinement_thresh = int(opt['refinement_thresh'])
            except ValueError as verr:
                print(verr, file=sys.stderr)
                sys.exit(2)
    except getopt.GetoptError as err:
        print(err, file=sys.stderr)
        sys.exit(2)

    #OPENING DATA
    import time
    if ".json" in opt_filename:
        # open file with trace data from json
        print("[INFO] Will only refine the state which is present in " + opt_filename + " if necessary.")
        with open(opt_filename, 'r') as f:
            states = json.load(f)
        # loop through all traces check if refinement is necessary
        print("Checking if refinement is necessary...")
        res = False
        for measurements_by_state in states:
            # loop through all occurrences of the looked at state
            print("Looking at state '" + measurements_by_state['name'] + "'")
            for measurement in measurements_by_state['offline']:
                # loop through measurements of particular state
                # an check if state needs refinement
                signal = measurement['uW']
                # mean = measurement['uW_mean']
                # TODO: Decide if median is really the better baseline than mean
                if needs_refinement_sort(signal, opt_refinement_thresh):
                    print("Refinement is necessary!")
                    break
    elif ".tar" in opt_filename:
        # open with dfatool
        raw_data_args = list()
        raw_data_args.append(opt_filename)
        raw_data = RawData(
            raw_data_args, with_traces=True
        )
        print("Preprocessing file. Depending on its size, this could take a while.")
        preprocessed_data = raw_data.get_preprocessed_data()
        print("File fully preprocessed")

        # TODO: Mal schauen, wie ich das mache. Erstmal nur mit json
    else:
        print("Unknown dataformat", file=sys.stderr)
        sys.exit(2)

    # print(tx_data[1]['parameter'])
    # # parse json to array for PELT
    # signal = np.array(tx_data[1]['offline'][0]['uW'])
    #
    # for i in range(0, len(signal)):
    #     signal[i] = signal[i]/1000
    # bkps = calc_PELT(signal, model=opt_model, range_max=opt_range_max, num_processes=opt_num_processes, jump=opt_jump, S=opt_S)
    # fig, ax = rpt.display(signal, bkps)
    # plt.xlabel('Time [us]')
    # plt.ylabel('Power [mW]')
    # plt.show()
