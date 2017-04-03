#!/usr/bin/env python3

import json
import numpy as np
import os
from scipy.cluster.vq import kmeans2
import struct
import sys
import tarfile
from dfatool import running_mean, MIMOSA

voltage = float(sys.argv[1])
shunt = float(sys.argv[2])
filename = sys.argv[3]

mim = MIMOSA(voltage, shunt)

charges, triggers = mim.load_data(filename)
trigidx = mim.trigger_edges(triggers)
triggers = []
cal_edges = mim.calibration_edges(running_mean(mim.currents_nocal(charges[0:trigidx[0]]), 10))
calfunc, caldata = mim.calibration_function(charges, cal_edges)
vcalfunc = np.vectorize(calfunc, otypes=[np.float64])

json_out = {
    'triggers' : len(trigidx),
    'first_trig' : trigidx[0] * 10,
    'calibration' : caldata,
    'trace' : mim.analyze_states(charges, trigidx, vcalfunc)
}

basename, _ = os.path.splitext(filename)

# TODO also look for interesting gradients inside each state

with open(basename + ".json", "w") as f:
    json.dump(json_out, f)
    f.close()

#print(kmeans2(charges[:firstidx], np.array([130 * ua_step, 3.6 / 987 * 1000000, 3.6 / 99300 * 1000000])))
