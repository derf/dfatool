#!/usr/bin/env python3

import numpy as np
from dfatool import PTAModel, RawData, soft_cast_int

ignored_trace_indexes = None

files = '../data/20170125_125433_cc1200.tar ../data/20170125_142420_cc1200.tar ../data/20170125_144957_cc1200.tar ../data/20170125_151149_cc1200.tar ../data/20170125_151824_cc1200.tar ../data/20170125_154019_cc1200.tar'.split(' ')
#files = '../data/20170116_124500_LM75x.tar ../data/20170116_131306_LM75x.tar'.split(' ')

raw_data = RawData(files)
preprocessed_data = raw_data.get_preprocessed_data()
model = PTAModel(preprocessed_data, ignore_trace_indexes = ignored_trace_indexes)
