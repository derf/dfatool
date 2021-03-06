#!/usr/bin/env python3

import sys
from dfatool.automata import PTA
from dfatool.utils import human_readable
from dfatool.lex import TimedSequence, TimedWord, Workload

args = sys.argv[1:]

loops = dict()
ptafiles = list()
loop_names = set()


def simulate_word(timedword):
    prev_state = "UNINITIALIZED"
    prev_param = None
    ret = dict()
    for trace_part in timedword:
        print("Trace Part {}".format(trace_part))
        if type(trace_part) is TimedWord:
            result = pta.simulate(
                trace_part, orig_state=prev_state, orig_param=prev_param
            )
        elif type(trace_part) is Workload:
            result = pta.simulate(
                trace_part.word, orig_state=prev_state, orig_param=prev_param
            )
            if prev_state != result.end_state:
                print(
                    "Warning: loop starts in state {}, but terminates in {}".format(
                        prev_state, result.end_state.name
                    )
                )
            if prev_param != result.parameters:
                print(
                    "Warning: loop starts with parameters {}, but terminates with {}".format(
                        prev_param, result.parameters
                    )
                )
            ret[trace_part.name] = result
            loop_names.add(trace_part.name)

        print("    Duration: " + human_readable(result.duration, "s"))
        if result.duration_mae:
            print(
                u"    ± {}  /  {:.0f}%".format(
                    human_readable(result.duration_mae, "s"), result.duration_mape
                )
            )
        print("    Energy: " + human_readable(result.energy, "J"))
        if result.energy_mae:
            print(
                u"    ± {}  /  {:.0f}%".format(
                    human_readable(result.energy_mae, "J"), result.energy_mape
                )
            )
        print("    Mean Power: " + human_readable(result.mean_power, "W"))
        print("")

        prev_state = result.end_state
        prev_param = result.parameters

    return ret


for i in range(len(args) // 2):
    ptafile, raw_word = args[i * 2], args[i * 2 + 1]
    ptafiles.append(ptafile)
    pta = PTA.from_file(ptafile)
    timedword = TimedSequence(raw_word)
    print("Input: {}\n".format(timedword))
    loops[ptafile] = simulate_word(timedword)

for loop_name in sorted(loop_names):
    result_set = list()
    total_power = 0
    for ptafile in sorted(ptafiles):
        if loop_name in loops[ptafile]:
            result_set.append(loops[ptafile][loop_name])
            total_power += loops[ptafile][loop_name].mean_power
    print(
        "{}: total mean power is {}".format(loop_name, human_readable(total_power, "W"))
    )
    for i, result in enumerate(result_set):
        print(
            "    {:.0f}% {} (period: {})".format(
                result.mean_power * 100 / total_power,
                ptafiles[i],
                human_readable(result.duration, "s"),
            )
        )
