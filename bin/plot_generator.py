import getopt
import sys
import re
import os
import numpy as np
import pprint
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # OPTION RECOGNITION
    opt = dict()

    optspec = "bench_filename= " "result_filename= "
    opt_bench_filename = None
    opt_result_filename = None
    try:
        raw_opts, args = getopt.getopt(sys.argv[1:], "", optspec.split(" "))

        for option, parameter in raw_opts:
            optname = re.sub(r"^--", "", option)
            opt[optname] = parameter
    except getopt.GetoptError as err:
        print(err, file=sys.stderr)
        sys.exit(-1)

    if "bench_filename" in opt:
        opt_bench_filename = opt["bench_filename"]
    else:

        sys.exit(-1)
    if "result_filename" in opt:
        opt_result_filename = opt["result_filename"]
    else:
        print("wth")
        sys.exit(-1)

    with open(opt_bench_filename, "r") as f:
        configurations = json.load(f)
    with open(opt_result_filename, "r") as f:
        sequence_line = f.readline()
        begin_sequence = sequence_line.rfind("Resulting Sequence: ") + 20

        if begin_sequence < 20:
            print("nicht gefunden!")
            sys.exit(-1)
        sequence_substr = sequence_line[begin_sequence:]
        resulting_sequence = eval(sequence_substr)
        new_line = f.readline()
        while new_line == "\n":
            new_line = f.readline()
        function_line = new_line
        pow_function_dict = dict()
        while function_line != "\n":
            state_name_pos = function_line.find("Power-Function for state ") + 25
            state_name_end = function_line.find(":")
            state_name = function_line[state_name_pos:state_name_end]
            function_string = function_line[state_name_end + 1 : -1]
            pow_function_dict[state_name] = function_string
            function_line = f.readline()
        new_line = "\n"
        while new_line == "\n":
            new_line = f.readline()
        function_line = new_line
        dur_function_dict = dict()
        while (
            function_line != "\n"
            and function_line != ""
            and "THIS RESULT IS NOT ACCURATE." not in function_line
        ):
            state_name_pos = function_line.find("Duration-Function for state ") + 28
            state_name_end = function_line.find(":")
            state_name = function_line[state_name_pos:state_name_end]
            function_string = function_line[state_name_end + 1 : -1]
            dur_function_dict[state_name] = function_string
            function_line = f.readline()

    param_names = configurations[0]["offline_aggregates"]["paramkeys"][0]

    for num_fig in range(0, min(4, len(configurations))):
        rand_config_no = np.random.randint(0, len(configurations), 1)[0]
        rand_conf = configurations[rand_config_no]
        rand_signal = np.array(rand_conf["offline"][0]["uW"])
        rand_param = rand_conf["offline_aggregates"]["param"][0]
        rand_max_pow = max(rand_signal)
        # pprint.pprint(rand_param)
        pretty_rand_param = pprint.pformat(rand_param)
        print(
            str(param_names)
            + "("
            + str(rand_config_no)
            + ")"
            + "\n"
            + pretty_rand_param
        )
        time = 0
        next_time = 0
        rand_stepper = 0
        pow = 0
        resulting_coords = list()
        while rand_stepper < len(resulting_sequence):
            curr_state = resulting_sequence[rand_stepper]
            curr_state_name = "State_" + str(curr_state)
            curr_pow_func = pow_function_dict[curr_state_name]
            curr_dur_func = dur_function_dict[curr_state_name]
            for num_param, name in enumerate(param_names):
                replace_string = "parameter(" + name + ")"
                curr_pow_func = curr_pow_func.replace(
                    replace_string, str(rand_param[num_param])
                )
                curr_dur_func = curr_dur_func.replace(
                    replace_string, str(rand_param[num_param])
                )
            pow = eval(curr_pow_func)
            dur = eval(curr_dur_func)
            next_time = time + dur
            start_coord = (time, pow)
            end_coord = (next_time, pow)
            resulting_coords.append(start_coord)
            resulting_coords.append(end_coord)
            rand_stepper = rand_stepper + 1
            time = next_time

        with open("res_conf_" + str(num_fig) + "_signal.txt", "w") as f:
            f.write("x,y\n")
            for x, y in enumerate(rand_signal):
                f.write(str(x) + "," + str(y) + "\n")
        with open("res_conf_" + str(num_fig) + "_fit.txt", "w") as f:
            f.write("x,y\n")
            for x, y in resulting_coords:
                f.write(str(x) + "," + str(y) + "\n")
        plt.plot(rand_signal)
        plt.plot([x for x, y in resulting_coords], [y for x, y in resulting_coords])
        plt.savefig("res_conf_" + str(num_fig) + "_pic.pdf", format="pdf", dpi=300)
        plt.clf()
