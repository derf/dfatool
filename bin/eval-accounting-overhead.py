#!/usr/bin/env python3
"""
eval-accounting-overhead -- evaluate overhead of various accounting methods and energy/power/timestamp integer sizes

Usage:
PYTHONPATH=lib bin/eval-accounting-overhead.py <files ...>

Data Generation:
for accounting in static_state_immediate 'static_state' 'static_statetransition_immediate' 'static_statetransition'; do for intsize in uint16_t uint32_t uint64_t; do PYTHONPATH=~/var/ess/aemr/dfatool/lib ~/var/ess/aemr/dfatool/bin/generate-dfa-benchmark.py --timer-pin=GPIO::p1_0 --sleep=30 --repeat=10 --depth=10 --arch=msp430fr5994lp --app=test_benchmark --trace-filter='setup,setAutoAck,write,getEnergy,$' --timing --dummy= --accounting=${accounting},ts_type=${intsize},power_type=${intsize},energy_type=${intsize} model/driver/nrf24l01.dfa ~/var/projects/multipass/src/app/test_benchmark/main.cc; done; done

Feed the resulting files into this script, output is one line per file
providing overhead per transition and getEnergy overhead

"""

from dfatool.dfatool import AnalyticModel, TimingData, pta_trace_to_aggregate
import json
import sys

for filename in sys.argv[1:]:
    with open(filename, 'r') as f:
        measurement = json.load(f)
    raw_data = TimingData([filename])
    preprocessed_data = raw_data.get_preprocessed_data()
    by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
    model = AnalyticModel(by_name, parameters, arg_count)
    static_model = model.get_static()
    if 'setup' in model.names:
        transition_duration = static_model('setup', 'duration')
    elif 'init' in model.names:
        transition_duration = static_model('init', 'duration')
    get_energy_duration = static_model('getEnergy', 'duration')

    print('{:60s}: {:.0f} / {:.0f} µs'.format(measurement['opt']['accounting'], transition_duration, get_energy_duration))
