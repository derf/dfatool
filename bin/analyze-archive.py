#!/usr/bin/env python3

import json
import numpy as np
import os
from scipy.cluster.vq import kmeans2
import struct
import sys
import tarfile
from dfatool import EnergyModel, RawData

if __name__ == '__main__':
    filename = sys.argv[1]
    raw_data = RawData(filename)

    preprocessed_data = raw_data.get_preprocessed_data()
    model = EnergyModel(preprocessed_data)
    static_model = model.get_static()

    print('--- simple static model ---')
    for state in model.states():
        print('{:10s}: {:.0f} µW'.format(state, static_model(state, 'power')))
    for trans in model.transitions():
        print('{:10s}: {:.0f} / {:.0f} / {:.0f} pJ'.format(
            trans, static_model(trans, 'energy'),
            static_model(trans, 'rel_energy_prev'),
            static_model(trans, 'rel_energy_next')))

    model.assess(model.get_static())
    sys.exit(0)
