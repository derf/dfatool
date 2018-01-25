#!/usr/bin/env python3

import json
import numpy as np
import os
from scipy.cluster.vq import kmeans2
import struct
import sys
import tarfile
from dfatool import Analysis, RawData

if __name__ == '__main__':
    filename = sys.argv[1]
    raw_data = RawData(filename)

    preprocessed_data = raw_data.get_preprocessed_data()
    print(preprocessed_data)
    foo = Analysis(preprocessed_data)
    res = foo.analyze()
    print(res)
    for key in res.keys():
        print(key)
        for subkey in res[key].keys():
            if subkey != 'isa' and len(res[key][subkey]) > 0:
                print('  {:s}: {:f}'.format(subkey, np.mean(res[key][subkey])))
    sys.exit(0)
