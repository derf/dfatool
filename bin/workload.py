#!/usr/bin/env python3

from automata import PTA
import sys
from utils import human_readable
from lex import TimedSequence, TimedWord, Workload

ptafile, raw_word = sys.argv[1:]

# TODO loops im raw_word:
# init(); repeat { foo(); sleep(5m); bar(); ... } o.ä.
# - Zeitangaben mit Einheit in sleep
# - Ausgabe in Gesamt, Init und Schleifeninhalt aufdröseln

pta = PTA.from_file(ptafile)
timedword = TimedSequence(raw_word)

print('Input: {}\n'.format(timedword))

prev_state = 'UNINITIALIZED'
prev_param = None
for trace_part in timedword:
    print('Trace Part {}'.format(trace_part))
    if type(trace_part) is TimedWord:
        result = pta.simulate(trace_part, orig_state=prev_state, orig_param=prev_param)
    elif type(trace_part) is Workload:
        result = pta.simulate(trace_part.word, orig_state=prev_state, orig_param=prev_param)
        if prev_state != result.end_state:
            print('Warning: loop starts in state {}, but terminates in {}'.format(prev_state, result.end_state.name))
        if prev_param != result.parameters:
            print('Warning: loop starts with parameters {}, but terminates with {}'.format(prev_param, result.parameters))

    print('    Duration: ' + human_readable(result.duration, 's'))
    if result.duration_mae:
        print(u'    ± {}  /  {:.0f}%'.format(human_readable(result.duration_mae, 's'), result.duration_mape))
    print('    Energy: ' + human_readable(result.energy, 'J'))
    if result.energy_mae:
        print(u'    ± {}  /  {:.0f}%'.format(human_readable(result.energy_mae, 'J'), result.energy_mape))
    print('    Mean Power: ' + human_readable(result.mean_power, 'W'))
    print('')

    prev_state = result.end_state
    prev_param = result.parameters
