#!/usr/bin/env python3

from automata import PTA
import re
import sys
from utils import soft_cast_int, human_readable

ptafile, raw_word = sys.argv[1:]

pta = PTA.from_file(ptafile)

trace = list()
for raw_symbol in raw_word.split(';'):
    match = re.fullmatch(r' *([^(]+)\((.*)\) *', raw_symbol)
    if match:
        function_name = match.group(1).strip()
        if match.group(2) == '':
            raw_args = list()
        else:
            raw_args = match.group(2).split(',')
        if function_name == 'sleep':
            function_name = None
        word = [function_name]
        for raw_arg in raw_args:
            word.append(soft_cast_int(raw_arg.strip()))
        trace.append(word)

print(trace)
result = pta.simulate(trace)

print('Duration: ' + human_readable(result.duration, 's'))
if result.duration_mae:
    print(u'    ± {}  /  {:.0f}%'.format(human_readable(result.duration_mae, 's'), result.duration_mape))
print('Energy: ' + human_readable(result.energy, 'J'))
if result.energy_mae:
    print(u'    ± {}  /  {:.0f}%'.format(human_readable(result.energy_mae, 'J'), result.energy_mape))
print('Mean Power: ' + human_readable(result.mean_power, 'W'))
