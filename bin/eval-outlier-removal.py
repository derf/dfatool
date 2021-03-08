#!/usr/bin/env python3

import getopt
import re
import sys
from dfatool.loader import RawData, pta_trace_to_aggregate
from dfatool.model import PTAModel

opt = dict()


def model_quality_table(result_lists, info_list):
    for state_or_tran in result_lists[0].keys():
        for key in result_lists[0][state_or_tran].keys():
            buf = "{:20s} {:15s}".format(state_or_tran, key)
            for i, results in enumerate(result_lists):
                results = results
                info = info_list[i]
                buf += "  |||  "
                if info == None or info(state_or_tran, key):
                    result = results[state_or_tran][key]
                    if "smape" in result:
                        buf += "{:6.2f}% / {:9.0f}".format(
                            result["smape"], result["mae"]
                        )
                    else:
                        buf += "{:6}    {:9.0f}".format("", result["mae"])
                else:
                    buf += "{:6}----{:9}".format("", "")
            print(buf)


def combo_model_quality_table(result_lists, info_list):
    for state_or_tran in result_lists[0][0].keys():
        for key in result_lists[0][0][state_or_tran].keys():
            for sub_result_lists in result_lists:
                buf = "{:20s} {:15s}".format(state_or_tran, key)
                for i, results in enumerate(sub_result_lists):
                    info = info_list[i]
                    buf += "  |||  "
                    if info == None or info(state_or_tran, key):
                        result = results[state_or_tran][key]
                        if "smape" in result:
                            buf += "{:6.2f}% / {:9.0f}".format(
                                result["smape"], result["mae"]
                            )
                        else:
                            buf += "{:6}    {:9.0f}".format("", result["mae"])
                    else:
                        buf += "{:6}----{:9}".format("", "")
                print(buf)


if __name__ == "__main__":

    ignored_trace_indexes = []
    discard_outliers = None

    try:
        raw_opts, args = getopt.getopt(
            sys.argv[1:], "", "plot ignored-trace-indexes= discard-outliers=".split(" ")
        )

        for option, parameter in raw_opts:
            optname = re.sub(r"^--", "", option)
            opt[optname] = parameter

            if "ignored-trace-indexes" in opt:
                ignored_trace_indexes = list(
                    map(int, opt["ignored-trace-indexes"].split(","))
                )
                if 0 in ignored_trace_indexes:
                    print("[E] arguments to --ignored-trace-indexes start from 1")

            if "discard-outliers" in opt:
                discard_outliers = float(opt["discard-outliers"])

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    raw_data = RawData(args)

    preprocessed_data = raw_data.get_preprocessed_data()
    by_name, parameters, arg_count = pta_trace_to_aggregate(
        preprocessed_data, ignored_trace_indexes
    )
    m1 = PTAModel(
        by_name,
        parameters,
        arg_count,
        traces=preprocessed_data,
        ignore_trace_indexes=ignored_trace_indexes,
    )
    m2 = PTAModel(
        by_name,
        parameters,
        arg_count,
        traces=preprocessed_data,
        ignore_trace_indexes=ignored_trace_indexes,
        discard_outliers=discard_outliers,
    )

    print("--- simple static model ---")
    static_m1 = m1.get_static()
    static_m2 = m2.get_static()
    # for state in model.states:
    #    print('{:10s}: {:.0f} µW  ({:.2f})'.format(
    #        state,
    #        static_model(state, 'power'),
    #        model.generic_param_dependence_ratio(state, 'power')))
    #    for param in model.parameters():
    #        print('{:10s}  dependence on {:15s}: {:.2f}'.format(
    #            '',
    #            param,
    #            model.param_dependence_ratio(state, 'power', param)))
    # for trans in model.transitions:
    #    print('{:10s}: {:.0f} / {:.0f} / {:.0f} pJ  ({:.2f} / {:.2f} / {:.2f})'.format(
    #        trans, static_model(trans, 'energy'),
    #        static_model(trans, 'rel_energy_prev'),
    #        static_model(trans, 'rel_energy_next'),
    #        model.generic_param_dependence_ratio(trans, 'energy'),
    #        model.generic_param_dependence_ratio(trans, 'rel_energy_prev'),
    #        model.generic_param_dependence_ratio(trans, 'rel_energy_next')))
    #    print('{:10s}: {:.0f} µs'.format(trans, static_model(trans, 'duration')))
    static_q1 = m1.assess(static_m1)
    static_q2 = m2.assess(static_m2)
    static_q12 = m1.assess(static_m2)

    print("--- LUT ---")
    lut_m1 = m1.get_param_lut()
    lut_m2 = m2.get_param_lut()
    lut_q1 = m1.assess(lut_m1)
    lut_q2 = m2.assess(lut_m2)
    lut_q12 = m1.assess(lut_m2)

    print("--- param model ---")
    param_m1, param_i1 = m1.get_fitted()
    for state in m1.states:
        for attribute in ["power"]:
            if param_i1(state, attribute):
                print(
                    "{:10s}: {}".format(
                        state, param_i1(state, attribute)["function"].model_function
                    )
                )
                print(
                    "{:10s}  {}".format(
                        "", param_i1(state, attribute)["function"].model_args
                    )
                )
    for trans in m1.transitions:
        for attribute in [
            "energy",
            "rel_energy_prev",
            "rel_energy_next",
            "duration",
            "timeout",
        ]:
            if param_i1(trans, attribute):
                print(
                    "{:10s}: {:10s}: {}".format(
                        trans,
                        attribute,
                        param_i1(trans, attribute)["function"].model_function,
                    )
                )
                print(
                    "{:10s}  {:10s}  {}".format(
                        "", "", param_i1(trans, attribute)["function"].model_args
                    )
                )
    param_m2, param_i2 = m2.get_fitted()
    for state in m2.states:
        for attribute in ["power"]:
            if param_i2(state, attribute):
                print(
                    "{:10s}: {}".format(
                        state, param_i2(state, attribute)["function"].model_function
                    )
                )
                print(
                    "{:10s}  {}".format(
                        "", param_i2(state, attribute)["function"].model_args
                    )
                )
    for trans in m2.transitions:
        for attribute in [
            "energy",
            "rel_energy_prev",
            "rel_energy_next",
            "duration",
            "timeout",
        ]:
            if param_i2(trans, attribute):
                print(
                    "{:10s}: {:10s}: {}".format(
                        trans,
                        attribute,
                        param_i2(trans, attribute)["function"].model_function,
                    )
                )
                print(
                    "{:10s}  {:10s}  {}".format(
                        "", "", param_i2(trans, attribute)["function"].model_args
                    )
                )

    analytic_q1 = m1.assess(param_m1)
    analytic_q2 = m2.assess(param_m2)
    analytic_q12 = m1.assess(param_m2)
    model_quality_table([static_q1, analytic_q1, lut_q1], [None, param_i1, None])
    model_quality_table([static_q2, analytic_q2, lut_q2], [None, param_i2, None])
    model_quality_table([static_q12, analytic_q12, lut_q12], [None, param_i2, None])
    combo_model_quality_table(
        [
            [static_q1, analytic_q1, lut_q1],
            [static_q2, analytic_q2, lut_q2],
            [static_q12, analytic_q12, lut_q12],
        ],
        [None, param_i1, None],
    )

    sys.exit(0)
